# MLX KV Cache Architecture Analysis for TurboQuant Integration

**Author:** Jinho Jang (eric@jangq.ai)
**Date:** 2026-03-24
**Scope:** Complete reverse-engineering of MLX KV cache subsystem for TurboQuant hook points

---

## 1. Cache Class Hierarchy

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/cache.py`

```
_BaseCache (abstract)
├── ConcatenateKVCache       — simplest, naive concatenation
├── KVCache                  — step-based pre-allocation (DEFAULT)
├── QuantizedKVCache         — mx.quantize() per-step, uniform bits
├── RotatingKVCache          — sliding window with rotation
├── ChunkedKVCache           — chunk-based with front trimming
├── ArraysCache              — generic array store (for SSM state)
├── CacheList                — composite cache for hybrid models
├── BatchKVCache             — batched KV with left-padding support
└── BatchRotatingKVCache     — batched sliding window
```

### Cache Factory: `make_prompt_cache()`

```python
def make_prompt_cache(model, max_kv_size=None):
    if hasattr(model, "make_cache"):
        return model.make_cache()         # hybrid models override this
    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)]
    else:
        return [KVCache() for _ in range(num_layers)]
```

Key insight: Models with `make_cache()` method (Jamba, GraniteMoeHybrid, Mamba, Mamba2) bypass the default and can mix cache types per layer. This is exactly the hook point TurboQuant should use.

---

## 2. KVCache — The Default Cache (CRITICAL PATH)

### Storage Layout

Shape: `(B, n_kv_heads, seq_len, head_dim)` — always float16/bfloat16 (matches model dtype).

```python
class KVCache(_BaseCache):
    step = 256  # pre-allocation granularity

    def __init__(self):
        self.keys = None      # mx.array or None
        self.values = None    # mx.array or None
        self.offset = 0       # current sequence position
```

### Step-Based Allocation Strategy

```python
def update_and_fetch(self, keys, values):
    prev = self.offset
    if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
        B, n_kv_heads, _, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        n_steps = (self.step + keys.shape[2] - 1) // self.step
        k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
        v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
        new_k = mx.zeros(k_shape, keys.dtype)
        new_v = mx.zeros(v_shape, values.dtype)
        if self.keys is not None:
            if prev % self.step != 0:
                self.keys = self.keys[..., :prev, :]
                self.values = self.values[..., :prev, :]
            self.keys = mx.concatenate([self.keys, new_k], axis=2)
            self.values = mx.concatenate([self.values, new_v], axis=2)
        else:
            self.keys, self.values = new_k, new_v

    self.offset += keys.shape[2]
    self.keys[..., prev : self.offset, :] = keys
    self.values[..., prev : self.offset, :] = values
    return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
```

**Allocation pattern:** Pre-allocate in chunks of `step=256` tokens. When buffer is full, allocate another 256 and concatenate. Returns sliced view `[..., :offset, :]` — attention always sees only valid tokens.

### Memory Per Layer (float16)

```
memory_per_layer = 2 * B * n_kv_heads * seq_len * head_dim * 2  bytes
                 = 4 * B * n_kv_heads * seq_len * head_dim      bytes
```

Example — DeepSeek-V3 (236B) at 4096 tokens, B=1:
- n_kv_heads = 128 (after MLA expansion), head_dim = 192
- Per layer: 4 * 1 * 128 * 4096 * 192 = 402 MB
- 61 layers: ~24 GB just for KV cache

Example — Qwen3-MoE (30B) at 4096 tokens, B=1:
- n_kv_heads = 4 (GQA), head_dim = 128
- Per layer: 4 * 1 * 4 * 4096 * 128 = 8.4 MB
- 48 layers: ~403 MB

### `to_quantized()` — The Conversion Hook

```python
def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
    quant_cache = QuantizedKVCache(group_size=group_size, bits=bits)
    quant_cache.offset = self.offset
    if self.keys is not None:
        quant_cache.keys = mx.quantize(self.keys, group_size=group_size, bits=bits)
        quant_cache.values = mx.quantize(self.values, group_size=group_size, bits=bits)
    return quant_cache
```

**CRITICAL:** This is called once from `maybe_quantize_kv_cache()` after prefill. It converts an existing `KVCache` into a `QuantizedKVCache`. TurboQuant can override this method to apply mixed-precision quantization instead of uniform bits.

---

## 3. QuantizedKVCache — Current Quantization (REPLACEMENT TARGET)

### How mx.quantize Works

`mx.quantize(x, group_size=64, bits=8)` returns a tuple of 3 arrays:
1. **Quantized data** — uint32 packed, shape `(..., dim // el_per_int)` where `el_per_int = 32 // bits`
2. **Scales** — same dtype as input, shape `(..., dim // group_size)`
3. **Biases** — same dtype as input, shape `(..., dim // group_size)`

This is **affine quantization**: `dequant = scale * quant_val + bias`

### Storage Format

```python
class QuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None      # tuple of (data, scales, biases) or None
        self.values = None    # tuple of (data, scales, biases) or None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits
```

`self.keys` and `self.values` are each a **tuple of 3 arrays** `(quantized_data, scales, biases)`.

### Quantize-on-Write

```python
def update_and_fetch(self, keys, values):
    B, n_kv_heads, num_steps, k_head_dim = keys.shape
    v_head_dim = values.shape[-1]
    prev = self.offset

    if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
        el_per_int = 8 * mx.uint32.size // self.bits
        new_steps = (self.step + num_steps - 1) // self.step * self.step
        shape = (B, n_kv_heads, new_steps)

        def init_quant(dim):
            return (
                mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
            )
        # ... allocation / expansion ...

    self.offset += num_steps

    keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
    values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
    for i in range(len(self.keys)):
        self.keys[i][..., prev : self.offset, :] = keys[i]
        self.values[i][..., prev : self.offset, :] = values[i]

    return tree_map(lambda x: x[..., : self.offset, :], (self.keys, self.values))
```

**Key observations:**
1. `mx.quantize()` is called on every new K/V insertion — not just once
2. Same bits for ALL heads, ALL layers, keys AND values
3. Returns the quantized tuple (not dequantized) — attention must handle this format
4. **`self.bits` attribute** is checked by `scaled_dot_product_attention` to dispatch to quantized path

### Memory Per Layer (quantized, 8-bit, group_size=64)

```
per_component = B * n_kv_heads * seq_len * (head_dim/4 + 2 * head_dim/64 * 2)
              = B * n_kv_heads * seq_len * (head_dim/4 + head_dim/16)
              ≈ B * n_kv_heads * seq_len * head_dim * 5/16   (for 8-bit)
```

For 4-bit:
```
              = B * n_kv_heads * seq_len * (head_dim/8 + head_dim/16)
              ≈ B * n_kv_heads * seq_len * head_dim * 3/16
```

Compared to float16 baseline (head_dim * 2 bytes = head_dim/8 in uint32 units):
- **8-bit:** ~5/16 vs 2 bytes = 62.5% reduction
- **4-bit:** ~3/16 vs 2 bytes = 81.25% reduction

---

## 4. Quantized SDPA — The Attention Consumer

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/base.py`

### Dispatch Logic

```python
def scaled_dot_product_attention(queries, keys, values, cache, scale, mask, sinks=None):
    if hasattr(cache, "bits"):
        # Quantized path
        return quantized_scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask,
            group_size=cache.group_size, bits=cache.bits,
        )
    else:
        # Normal path — uses mx.fast SDPA (Metal optimized)
        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask, sinks=sinks,
        )
```

**CRITICAL:** The dispatch checks `hasattr(cache, "bits")`. Any TurboQuant cache must have a `bits` attribute to trigger the quantized path. However, the quantized path uses a single `bits` value for ALL heads.

### Quantized Attention Implementation

```python
def quantized_scaled_dot_product_attention(
    queries, q_keys, q_values, scale, mask, group_size=64, bits=8):

    B, n_q_heads, L, D = queries.shape
    n_kv_heads = q_keys[0].shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    queries *= scale

    if n_repeats > 1:
        queries = mx.reshape(queries, (B, n_kv_heads, n_repeats, L, D))
        q_keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_keys)
        q_values = tree_map(lambda x: mx.expand_dims(x, axis=-3), q_values)

    # Q @ K^T using quantized matmul
    scores = mx.quantized_matmul(
        queries, *q_keys, transpose=True, group_size=group_size, bits=bits
    )
    # ... masking + softmax ...
    # scores @ V using quantized matmul
    out = mx.quantized_matmul(
        scores, *q_values, transpose=False, group_size=group_size, bits=bits
    )
    return out
```

**Key insight:** `mx.quantized_matmul` takes the 3-tuple `(data, scales, biases)` from quantized arrays and performs the matmul without explicit dequantization. This is a Metal-optimized kernel. TurboQuant must either:
1. Use the same format and pass through to `mx.quantized_matmul`
2. Replace the entire SDPA with a custom implementation
3. Dequantize before calling `mx.fast.scaled_dot_product_attention`

---

## 5. RotatingKVCache — Sliding Window

```python
class RotatingKVCache(_BaseCache):
    step = 256

    def __init__(self, max_size, keep=0):
        self.keep = keep          # tokens to always keep (beginning of sequence)
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = max_size
        self._idx = 0             # current write position in circular buffer
```

**Two modes:**
- `_update_concat` (for prefill, multi-token): Puts cache in temporal order, trims old tokens, appends new
- `_update_in_place` (for decode, single token): Circular buffer write at `_idx`, wraps to `keep` position

**Quantization status:** `to_quantized()` raises `NotImplementedError("RotatingKVCache Quantization NYI")`.

**TurboQuant implication:** If we want to support sliding window models, we need to implement quantized rotating cache ourselves.

---

## 6. DeepSeek V2/V3 — MLA (Multi-head Latent Attention) Cache

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/deepseek_v2.py`

### Architecture

MLA compresses KV through low-rank projection:
- `kv_lora_rank = 512` (compressed latent dimension)
- `qk_rope_head_dim = 64` (RoPE component, NOT compressed)
- `qk_nope_head_dim = 128` (non-RoPE key dimension)
- `v_head_dim = 128` (value dimension)

### What Gets Cached

```python
def __call__(self, x, mask=None, cache=None):
    # Compress KV
    compressed_kv = self.kv_a_proj_with_mqa(x)  # → (B, L, kv_lora_rank + qk_rope_head_dim)
    compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)

    # Decompress for attention
    kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))  # → full K_nope + V
    kv = kv.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
    k_nope, values = mx.split(kv, [self.qk_nope_head_dim], axis=-1)

    # Apply RoPE to PE component
    k_pe = self.rope(k_pe, cache.offset)
    k_pe = mx.repeat(k_pe, self.num_heads, axis=1)

    # Cache the DECOMPRESSED K and V (not the compressed latent!)
    keys, values = cache.update_and_fetch(
        mx.concatenate([k_nope, k_pe], axis=-1), values
    )
```

**CRITICAL FINDING:** Despite MLA being a compression technique, the MLX implementation caches the DECOMPRESSED keys and values — the full `(n_heads, seq_len, q_head_dim)` for keys and `(n_heads, seq_len, v_head_dim)` for values. The latent compression is NOT preserved in cache.

This means:
- Keys cached: `(B, num_heads, seq_len, qk_nope_head_dim + qk_rope_head_dim)` = `(B, 128, seq, 192)`
- Values cached: `(B, num_heads, seq_len, v_head_dim)` = `(B, 128, seq, 128)`
- Uses standard `KVCache.update_and_fetch()` — same as any MHA model

**TurboQuant opportunity:** MLA models have the HIGHEST cache memory pressure (128 heads instead of GQA's 4-8). Quantizing the cache here gives the biggest absolute savings. But it uses the standard KVCache path, so TurboQuant hooks apply directly.

### Memory Calculation (DeepSeek-V3, 4096 tokens, B=1)

```
Keys:   1 * 128 * 4096 * 192 * 2 = 201 MB per layer
Values: 1 * 128 * 4096 * 128 * 2 = 134 MB per layer
Total:  335 MB per layer * 61 layers = ~20 GB
```

With 4-bit quantization: ~20 GB * 3/16 / 2 ≈ ~1.9 GB (10x reduction)

---

## 7. Mamba/SSM Models — No KV Cache

### Mamba 1 (Pure SSM)

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/mamba.py`

Uses `ArraysCache(size=2)` per layer storing:
- `cache[0]` = conv_state: `(B, conv_kernel-1, intermediate_size)` — convolution buffer
- `cache[1]` = ssm_state: `(B, intermediate_size, state_size)` — SSM recurrent state

```python
def make_cache(self):
    return [ArraysCache(size=2) for _ in range(len(self.layers))]
```

**No KV cache at all.** State is fixed-size regardless of sequence length. TurboQuant has nothing to quantize here.

### Mamba 2

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/mamba2.py`

Same pattern with `ArraysCache(size=2)`:
- `cache[0]` = conv_state
- `cache[1]` = SSM state (managed by Metal kernel via `ssm.py`)

Uses a custom Metal kernel for single-step inference (`ssm_update_kernel`) and SSD-SSM attention-like computation for prefill (`ssm_attn`).

### SSM Metal Kernel

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/ssm.py`

The `ssm_update_kernel` is a custom Metal kernel that:
1. Takes `state_in` as input
2. Computes `state = dA * state_in + dB * x` per head per state dimension
3. Outputs `state_out` and `y`
4. State shape: `(B*H, Dh, Ds)` — batch*heads by head_dim by state_dim

**No quantization opportunity in SSM state** — it's a dense recurrent state that's continuously modified, not accumulated like KV.

---

## 8. Hybrid Models — Mixed Cache Types

### Jamba (Attention + Mamba SSM + MoE)

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/jamba.py`

```python
def make_cache(self):
    caches = []
    for layer in self.model.layers:
        if layer.is_attn:
            caches.append(KVCache())          # standard KV for attention layers
        else:
            caches.append(ArraysCache(size=2))  # SSM state for mamba layers
    return caches
```

Layer assignment:
```python
layers_block_type = [
    "attention" if i % attn_layer_period == attn_layer_offset else "mamba"
    for i in range(num_hidden_layers)
]
```

**TurboQuant implication:** Only attention layers have KV caches to quantize. Must detect layer type and skip SSM layers. The `make_cache()` pattern is the right hook — override it to return `TurboQuantKVCache()` for attention layers and `ArraysCache()` for SSM layers.

### GraniteMoeHybrid (Attention + Mamba2 + MoE)

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/granitemoehybrid.py`

Same pattern:
```python
def make_cache(self):
    caches = []
    for layer in self.layers:
        if layer.layer_type == "mamba":
            caches.append(ArraysCache(size=2))
        elif layer.layer_type == "attention":
            caches.append(KVCache())
    return caches
```

---

## 9. GQA Cache (Qwen3-MoE, Standard Transformer)

Source: `.venv/lib/python3.14/site-packages/mlx_lm/models/qwen3_moe.py`

### Attention with GQA

```python
class Attention(nn.Module):
    def __init__(self, args, layer_idx):
        self.n_heads = args.num_attention_heads      # e.g., 32
        self.n_kv_heads = args.num_key_value_heads   # e.g., 4 (GQA ratio 8:1)
        head_dim = args.head_dim                      # e.g., 128

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)  # smaller!
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)  # smaller!
        # ... QK norms, RoPE ...
```

Cache call:
```python
    keys, values = cache.update_and_fetch(keys, values)
    # keys shape: (B, n_kv_heads, seq, head_dim) = (B, 4, seq, 128)
    # values shape: (B, n_kv_heads, seq, head_dim) = (B, 4, seq, 128)
```

**GQA reduces cache size by kv_ratio** (e.g., 8x for 32 heads / 4 kv_heads). Uses standard `KVCache` — no special cache class needed.

### Memory Calculation (Qwen3-MoE 30B, 4096 tokens, B=1)

```
Keys:   1 * 4 * 4096 * 128 * 2 = 4.2 MB per layer
Values: 1 * 4 * 4096 * 128 * 2 = 4.2 MB per layer
Total:  8.4 MB per layer * 48 layers = ~403 MB
```

GQA models already have small caches — TurboQuant benefit is proportionally less but still useful for long contexts.

---

## 10. Cache Conversion Flow — `generate_step()`

Source: `.venv/lib/python3.14/site-packages/mlx_lm/generate.py`

```python
def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)
```

Called during `generate_step()`:
```python
def generate_step(prompt, model, *, ..., kv_bits=None, kv_group_size=64,
                  quantized_kv_start=0, ...):
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )
    # ... prefill loop ...
    quantize_cache_fn(prompt_cache)  # converts KVCache → QuantizedKVCache
    # ... decode loop (all new KV insertions auto-quantized by QuantizedKVCache) ...
```

**Flow:**
1. Prefill uses `KVCache` (full precision)
2. After prefill, `maybe_quantize_kv_cache()` converts to `QuantizedKVCache`
3. Decode tokens are quantized on-write by `QuantizedKVCache.update_and_fetch()`

---

## 11. TurboQuant Integration Points

### Strategy A: Override `to_quantized()` (Minimal Change)

Replace `KVCache.to_quantized()` to produce a `TurboQuantKVCache` instead of `QuantizedKVCache`.

```python
# Hook point: KVCache.to_quantized()
def to_quantized(self, group_size=64, bits=4, turbo_config=None):
    if turbo_config:
        return TurboQuantKVCache.from_kv_cache(self, turbo_config)
    # ... original behavior ...
```

**Pros:** Works with existing `generate_step()` flow, no changes to generate.py needed.
**Cons:** Must monkey-patch or subclass KVCache.

### Strategy B: Custom `make_cache()` (Per-Model)

Override `make_cache()` in the Model class to return `TurboQuantKVCache` directly.

```python
def make_cache(self):
    return [TurboQuantKVCache(layer_config=self.turbo_config[i])
            for i in range(len(self.layers))]
```

**Pros:** Per-layer configuration from the start (no conversion step needed).
**Cons:** Must modify each model file or monkey-patch Model.make_cache.

### Strategy C: Replace `maybe_quantize_kv_cache()` (Cleanest)

Replace the conversion function in generate.py:

```python
def turbo_quantize_kv_cache(prompt_cache, turbo_config):
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized"):
            layer_config = turbo_config.get_layer_config(e)
            prompt_cache[e] = TurboQuantKVCache.from_kv_cache(c, layer_config)
```

**Pros:** Single point of control, per-layer config, no model changes.
**Cons:** Requires patching generate.py or wrapping generate_step().

### RECOMMENDED: Strategy C with Strategy A as fallback

Strategy C gives per-layer mixed-precision control. Strategy A handles cases where models use custom `to_quantized()` calls.

---

## 12. What TurboQuantKVCache Must Implement

### Required Interface (from _BaseCache)

```python
class TurboQuantKVCache(_BaseCache):
    # MUST have .bits attribute for SDPA dispatch
    bits = None          # needs custom handling — see below
    group_size = 64

    # Core methods
    def update_and_fetch(self, keys, values) → (q_keys, q_values)
    def make_mask(self, N, **kwargs) → mask

    # State management
    @property
    def state(self) → serializable state
    @property
    def meta_state(self) → metadata for save/load
    def is_trimmable(self) → bool
    def trim(self, n) → int

    # Properties
    @property
    def offset(self) → int
    @property
    def nbytes(self) → int
    def empty(self) → bool
```

### The `bits` Problem

The current SDPA dispatch checks `hasattr(cache, "bits")` and passes a single `bits` to `mx.quantized_matmul`. TurboQuant needs mixed bits per head or per layer.

**Options:**
1. **Uniform bits per layer, vary across layers** — easiest, set `self.bits` per-layer TurboQuantKVCache instance. Works with existing SDPA.
2. **Mixed bits within a layer (keys vs values)** — need custom SDPA that calls `mx.quantized_matmul` twice with different bits.
3. **Mixed bits per head** — need completely custom attention, splitting heads by quantization level.

**Recommendation:** Start with option 1 (per-layer bits). This alone gives significant savings — e.g., early layers 8-bit, middle layers 4-bit, late layers 4-bit. No SDPA changes needed.

### Critical Method: `update_and_fetch()`

```python
def update_and_fetch(self, keys, values):
    """
    keys:   (B, n_kv_heads, num_new_tokens, k_head_dim) — float16
    values: (B, n_kv_heads, num_new_tokens, v_head_dim) — float16

    Returns:
        q_keys:   tuple(data, scales, biases) — quantized
        q_values: tuple(data, scales, biases) — quantized

    TurboQuant modifications:
    - Could use different bits for keys vs values
    - Could use different group_size
    - Could apply importance-weighted quantization
    - Could keep recent tokens in higher precision
    """
    # Quantize with TurboQuant scheme
    k_quant = mx.quantize(keys, group_size=self.k_group_size, bits=self.k_bits)
    v_quant = mx.quantize(values, group_size=self.v_group_size, bits=self.v_bits)
    # ... store and return ...
```

---

## 13. mx.quantize / mx.dequantize / mx.quantized_matmul Reference

### mx.quantize(x, group_size=64, bits=8)

- Input: any float array
- Output: `(data: uint32, scales: same_dtype, biases: same_dtype)`
- Method: Per-group affine quantization
- `data` packs multiple values into uint32: `el_per_int = 32 // bits`
- Supported bits: 2, 4, 8

### mx.quantized_matmul(x, data, scales, biases, transpose, group_size, bits)

- Performs `x @ dequant(data, scales, biases)` or `x @ dequant(data, scales, biases)^T`
- Metal-optimized kernel — avoids materializing the full dequantized matrix
- **This is why quantized KV cache is fast** — no dequant step

### Bit Packing Layout

```
bits=8:  4 values per uint32  (8 * 4 = 32)
bits=4:  8 values per uint32  (4 * 8 = 32)
bits=2:  16 values per uint32 (2 * 16 = 32)
```

For head_dim=128, bits=4: data is `(B, H, seq, 128/8)` = `(B, H, seq, 16)` uint32 values.

---

## 14. Quantization Sensitivity Analysis

### What to Quantize Aggressively (4-bit or 2-bit)

1. **Values in middle layers** — empirically least sensitive to quantization
2. **Keys in layers with GQA** — already compressed, low dimensionality
3. **Middle attention layers** — less critical than first/last layers

### What to Keep at Higher Precision (8-bit or float16)

1. **First 2-3 layers** — attend to BOS/system prompt, high impact
2. **Last 2-3 layers** — directly affect logits
3. **Keys with RoPE** — position information is sensitive to quantization error
4. **Attention sink tokens** — first few KV entries that all tokens attend to

### Layer-Aware Configuration Sketch

```python
turboquant_config = {
    "layers": {
        # First 3 layers: keep float16
        range(0, 3): {"bits": None},  # no quantization
        # Early layers: 8-bit
        range(3, 10): {"bits": 8, "group_size": 64},
        # Middle layers: 4-bit
        range(10, num_layers - 3): {"bits": 4, "group_size": 64},
        # Last 3 layers: 8-bit
        range(num_layers - 3, num_layers): {"bits": 8, "group_size": 64},
    },
    "sink_tokens": 4,  # keep first 4 tokens at full precision
}
```

---

## 15. Summary of Hook Points

| Hook Point | File | Method/Function | What TurboQuant Does |
|---|---|---|---|
| Cache creation | cache.py | `make_prompt_cache()` | Return TurboQuantKVCache per layer |
| Model cache | model.py | `Model.make_cache()` | Override for hybrid models |
| KV quantization | cache.py | `KVCache.to_quantized()` | Return TurboQuantKVCache instead of QuantizedKVCache |
| Conversion trigger | generate.py | `maybe_quantize_kv_cache()` | Replace with per-layer config |
| SDPA dispatch | base.py | `scaled_dot_product_attention()` | Check `cache.bits` — must be set on TurboQuantKVCache |
| Quantized matmul | base.py | `quantized_scaled_dot_product_attention()` | May need replacement for mixed K/V bits |
| On-write quant | cache.py | `QuantizedKVCache.update_and_fetch()` | Replace `mx.quantize()` with TurboQuant scheme |

---

## 16. Files Referenced

All paths relative to `.venv/lib/python3.14/site-packages/mlx_lm/`:

| File | Contains |
|---|---|
| `models/cache.py` | All cache classes: KVCache, QuantizedKVCache, RotatingKVCache, ArraysCache, BatchKVCache, etc. |
| `models/base.py` | `scaled_dot_product_attention()`, `quantized_scaled_dot_product_attention()`, mask creation |
| `models/deepseek_v2.py` | MLA attention — caches decompressed K/V, not latent |
| `models/mamba.py` | Mamba 1 — ArraysCache for conv + SSM state, no KV |
| `models/mamba2.py` | Mamba 2 — ArraysCache + Metal SSM kernel |
| `models/ssm.py` | SSM update kernels (Metal + Python fallback) |
| `models/jamba.py` | Hybrid attention+SSM — mixed KVCache + ArraysCache |
| `models/granitemoehybrid.py` | Hybrid attention+Mamba2+MoE — mixed KVCache + ArraysCache |
| `models/qwen3_moe.py` | Standard GQA attention with MoE FFN |
| `generate.py` | `generate_step()`, `maybe_quantize_kv_cache()` — the conversion flow |
