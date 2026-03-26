# TurboQuant Edge Cases & Architecture Analysis
**Author:** Jinho Jang (eric@jangq.ai)
**Date:** 2026-03-24

---

## 1. Attention Type Detection Matrix

| Model | Type | Layers | KV Heads | Head Dim | Experts | Detection Method | Cache@4K |
|-------|------|--------|----------|----------|---------|-----------------|---------|
| Qwen3.5-27B | GQA + SSM | 64 | 4 | 256 | 0 | `text_config.layer_types[]` | 1.1 GB |
| MiniMax M2.5 | GQA | 62 | 8 | 128 | 256 | all attention (no hybrid) | 1.0 GB |
| Qwen3.5-35B-A3B | GQA + SSM | 40 | 2 | 256 | 0 | `text_config.layer_types[]` | 0.3 GB |
| Qwen3.5-122B-A10B | GQA + SSM | 48 | 2 | 256 | 0 | `text_config.layer_types[]` | 0.4 GB |
| Nemotron Cascade | GQA + Mamba | 52 | 2 | 128 | 128 | `hybrid_override_pattern` | 0.2 GB |
| Mistral 4 119B | MLA + MoE | 36 | 128* | 192/128 | 128 | `kv_lora_rank > 0` | 2.4 GB |

*Mistral 4 has 32 num_kv_heads but MLA expands to 128 effective heads after decompression.

## 2. Edge Cases & How TurboQuant Handles Them

### 2a. Hybrid SSM + Attention (Qwen3.5, Nemotron)
**Issue:** Only 25-55% of layers have KV cache. SSM layers have cumulative state (ArraysCache).
**Detection:**
- Qwen3.5: `text_config.layer_types` = `["linear_attention", ..., "full_attention", ...]`
- Nemotron: `hybrid_override_pattern` = `"MEMEM*EMEMEM*..."` (M=Mamba, E=attention)
**Handling:** `make_turboquant_cache()` creates TurboQuantKVCache for attention layers,
ArraysCache for SSM layers. VERIFIED WORKING on Qwen3.5-27B.

### 2b. MLA (Multi-head Latent Attention) — Mistral 4
**Issue:** Keys have different dim (192 = nope_128 + rope_64) than values (128).
Asymmetric key/value dimensions. Also, 128 effective KV heads (biggest cache).
**Detection:** `kv_lora_rank > 0` or `model_type == "mistral4"`
**Handling:** Separate TurboQuantEncoder for keys (dim=192) and values (dim=128).
Block Hadamard handles 192 = 128 + 64 decomposition. VERIFIED in unit tests.
**Status:** Needs Mac Studio mlx-lm patches for E2E test.

### 2c. 256-Expert MoE (MiniMax M2.5)
**Issue:** Gate routing is precision-sensitive. MoE gate uses sigmoid + bias (not softmax).
Group size must be >= 128 for 256 experts to prevent NaN.
**Detection:** `n_routed_experts >= 256`
**Handling:** TurboQuant only affects KV cache, NOT gate weights. Gate weights are
already handled separately by JANG loader (float16 passthrough or 8-bit quantized).
MoE routing is in MLP, not attention — TurboQuant doesn't touch it.
**Status:** Should work (GQA attention, standard cache), needs E2E verification.

### 2d. Large Head Dimensions (256-dim in Qwen3.5)
**Issue:** head_dim=256 is larger than typical 128. Hadamard on 256 is a power-of-2 (fine).
Codebook computation for dim=256 has narrower distribution (N(0, 1/256)).
**Handling:** Codebook automatically adapts to dimension. VERIFIED in unit tests (dim=256).

### 2e. Very Few KV Heads (2 heads in Qwen3.5-35B, Nemotron)
**Issue:** With only 2 KV heads, the cache is already small. TurboQuant savings
are proportionally less in absolute terms.
**Handling:** Still works correctly, just less impactful. At 32K context:
Qwen3.5-35B cache = 2.4 GB -> 0.45 GB (saves 2 GB).

### 2f. QK Normalization (MiniMax M2.5)
**Issue:** MiniMax applies RMSNorm to both Q and K before attention. If we quantize
K after normalization, the norm is already applied. If before, we need to handle it.
**Handling:** TurboQuant operates on the final K/V passed to cache.update_and_fetch().
The model computes K, applies any norms, THEN passes to cache. TurboQuant sees
the final normalized K — this is correct.

### 2g. RoPE Variants
**Issue:** Different models use different RoPE: traditional vs interleaved, different
theta values, YaRN scaling.
**Handling:** RoPE is applied to K BEFORE caching. TurboQuant operates on the
post-RoPE K values. The Hadamard rotation is independent of RoPE — it spreads
coordinate energy uniformly regardless of what distribution the inputs have.

### 2h. Sliding Window Attention (RotatingKVCache)
**Issue:** Some models use RotatingKVCache with max_size. TurboQuant must handle
the circular buffer pattern.
**Handling:** Current implementation uses TurboQuantKVCache (not rotating).
For sliding window models, we'd need a TurboQuantRotatingKVCache. Not yet implemented.
**Impact:** Low priority — sliding window models already have bounded cache.

### 2i. BatchKVCache (continuous batching in vmlx)
**Issue:** vmlx uses BatchKVCache for concurrent request handling.
**Handling:** TurboQuantKVCache doesn't support batched operations yet.
For vmlx integration, need to add extract/merge/filter methods.
**Impact:** Required for production vmlx integration.

### 2j. Attention Sink Tokens
**Issue:** First few tokens (BOS, system prompt start) receive disproportionate
attention from all subsequent tokens. Quantizing these hurts quality.
**Handling:** Config has `sink_tokens: 4` parameter. When compress() is called,
the first N sink tokens should be kept at full precision.
**Status:** Parameter exists but not yet enforced in compress().

## 3. Architecture Support Matrix

| Architecture | Attention | Hybrid | TurboQuant Status |
|-------------|-----------|--------|-------------------|
| Qwen3.5 (all sizes) | GQA + GatedDeltaNet SSM | Yes | WORKING |
| MiniMax M2.5 | GQA + QK norm | No | READY (untested E2E) |
| Nemotron Cascade | GQA + Mamba-2 | Yes | READY (gate dequant issue) |
| Mistral Small 4 | MLA (decompressed) | No | READY (needs mlx-lm patches) |
| Qwen3-MoE | GQA | No | READY (standard arch) |
| Llama / standard | MHA/GQA | No | READY (standard arch) |
| DeepSeek V3 | MLA (absorbed) | No | UNTESTED (different MLA variant) |

## 4. Detection Priority

When TurboQuant encounters a model, detection happens in this order:

1. **JANG gate check:** Is `jang_config.json` present with `turboquant.enabled`?
   → No: exit, no TurboQuant
2. **Layer count:** How many layers? → Sets n_layers for config
3. **MLA detection:** `kv_lora_rank > 0`?
   → Yes: asymmetric key/value dims (k=nope+rope, v=v_head_dim)
   → No: symmetric dims (head_dim for both)
4. **Hybrid detection:** `text_config.layer_types[]` present?
   → Yes: map full_attention → TurboQuant, everything else → ArraysCache
   → No: `hybrid_override_pattern` present?
     → Yes: map E/A → TurboQuant, M → ArraysCache
     → No: all layers are attention → all get TurboQuant
5. **Per-layer bits:** Critical layers (first/last) get more bits
