# Porting JANG to a new framework

This doc tells framework authors (ollama, llama.cpp, candle, a homegrown WASM runtime, etc.) what
they need to implement to load and run JANG and JANGTQ models.

## On-disk layout

A JANG model directory is a superset of standard HuggingFace safetensors:

```
my-model-JANG_4K/
├── config.json                          # HF model config (standard) + quantization key
├── jang_config.json                     # JANG-specific metadata (REQUIRED)
├── tokenizer.json                       # HF fast tokenizer (standard)
├── tokenizer_config.json                # HF tokenizer config (standard)
├── special_tokens_map.json              # HF special tokens
├── chat_template.jinja                  # Optional; inline in tokenizer_config.json also allowed
├── generation_config.json               # HF generation defaults (optional)
├── preprocessor_config.json             # Required for vision/VL models
├── video_preprocessor_config.json       # Required for video VL models
├── model.safetensors.index.json         # Shard index (standard)
├── model-00001-of-00003.safetensors     # Weight shards
├── model-00002-of-00003.safetensors
└── model-00003-of-00003.safetensors
```

### Identifying a JANG model

Check `jang_config.json`:

```json
{
  "format": "jang",
  "format_version": "2.0",
  "quantization": {
    "method": "jang-importance",
    "profile": "JANG_4K",
    "target_bits": 4.0,
    "actual_bits": 4.23,
    "block_size": 64,
    "bit_widths_used": [3, 4, 6, 8],
    "quantization_scheme": "asymmetric",
    "quantization_backend": "mx.quantize"
  },
  "source_model": {
    "name": "Qwen/Qwen3-30B-A3B",
    "dtype": "bfloat16",
    "parameters": "30B"
  },
  "runtime": {
    "total_weight_bytes": 18000000000,
    "total_weight_gb": 18.0
  }
}
```

The sentinel: `format == "jang"` and `format_version` starts with `"2."`.

Legacy v1 models have `format_version == "1.1"` and use `.jang.safetensors` shards with a
`model.jang.index.json` index. The Python loader handles both; see `FORMAT.md` for the v1 layout.

## JANG (standard) per-tensor layout

For each quantized tensor in the safetensors shards, you will find three companion tensors:

| Key suffix | Type | Shape | Purpose |
|---|---|---|---|
| `<name>.weight` | `uint32` | `(out_features, packed_in)` | Packed quantized weights |
| `<name>.scales` | `float16` | `(out_features, n_groups)` | Per-group scale factors |
| `<name>.biases` | `float16` | `(out_features, n_groups)` | Per-group biases (zero-points) |

For MoE stacked expert tensors the shapes have a leading expert dimension:
`(num_experts, out_features, packed_in)` / `(num_experts, out_features, n_groups)`.

`n_groups = ceil(in_features / group_size)` where `group_size` defaults to 64
(see `jang_config.json.quantization.block_size`).

`packed_in = ceil(in_features * bits / 32)`

### Dequantization

For a weight `W[i, j]`, find its containing group `g = j // group_size`. Let:

- `s = scales[i, g]`
- `z = biases[i, g]`
- `q` = the quantized integer unpacked from `.weight` at position `(i, j)` (see below)

Then `W[i, j] = q * s + z`. Symmetric or asymmetric — both stored identically; the sign is
encoded in `z`.

**Bit widths vary per tensor.** A JANG_4K model has some tensors at 3-bit, some at 4-bit, some
at 6-bit, some at 8-bit. The bit width for a given tensor is NOT stored as a metadata field;
it is inferred from the weight tensor shape:

```python
in_dim = scales.shape[-1] * group_size         # recover original in_features
bits = (weight.shape[-1] * 32) // in_dim       # e.g. 4 for 4-bit
```

This is the same formula MLX uses internally; see `FORMAT.md` section "Bit Width Inference".

### Unpacking bits from uint32

JANG packs bits little-endian within each uint32 row. For a `k`-bit tensor, the quantized integer
at column `j` is:

```python
word_index = (j * k) // 32
bit_offset = (j * k) % 32
word = packed_weight[row, word_index]
q = (word >> bit_offset) & ((1 << k) - 1)
```

Rows are packed independently — no cross-row bit spanning. When `bit_offset + k > 32` the
value wraps into the next word; the reference Python packing code at
`jang-tools/jang_tools/pack.py` handles this case explicitly.

### MoE expert tensor naming

JANG converts per-expert weights from HuggingFace's per-expert dict to MLX's stacked
`switch_mlp` convention during conversion:

| HuggingFace source | JANG v2 name |
|---|---|
| `experts.N.gate_proj.weight` (per-expert) | `switch_mlp.gate_proj.weight` `[E, out, packed]` |
| `experts.N.up_proj.weight` (per-expert) | `switch_mlp.up_proj.weight` `[E, out, packed]` |
| `experts.N.down_proj.weight` (per-expert) | `switch_mlp.down_proj.weight` `[E, out, packed]` |
| `experts.gate_up_proj.weight` (fused, HF) | split into `gate_proj` + `up_proj` at load |

The stacking eliminates per-expert tensor iteration at inference time.

## JANGTQ (TurboQuant) extensions

A JANGTQ model has `"method": "jangtq"` in `jang_config.json.quantization` (or check that
`.weight` tensors for expert MLP layers have a companion `.tq_codebook` tensor).

### JANGTQ expert MLP layout

For a stacked expert tensor such as `switch_mlp.gate_proj.weight`:

| Key suffix | Purpose |
|---|---|
| `<name>.weight` (uint32) | Packed codebook indices |
| `<name>.scales` (float16) | Per-block scales |
| `<name>.tq_codebook` (float16) | The codebook itself — shape `[num_codes, block_size]` |

Dequantization is two steps:
1. Unpack `codebook_idx` from `.weight` (same bit-unpacking as standard JANG above)
2. `vec = codebook[codebook_idx] * scales[block]`

If Hadamard rotation was applied during quantization, `<name>.tq_hadamard_seed` will be present.
Apply the inverse Hadamard to the dequantized block before use:
`vec = hadamard_inverse(vec, seed=tq_hadamard_seed)`.

### Reference decoders

- Python: `jang-tools/jang_tools/turboquant/codebook.py` (codebook lookup),
  `jang-tools/jang_tools/turboquant/rotation.py` (Hadamard)
- Metal kernel: `jang-runtime/Sources/JANGCoreMetal/JANGTQMatmul.metal` — fuses
  codebook read + scale + matmul into a single Metal dispatch

## config.json quantization key

v2 models add a `quantization` key to the standard HuggingFace `config.json`:

```json
{
  "model_type": "qwen3_5_moe",
  "quantization": {
    "group_size": 64,
    "bits": 4
  }
}
```

`bits` is the most common bit width in the model (the COMPRESS tier). The MLX loader uses this
to construct `QuantizedLinear` / `QuantizedSwitchLinear` layers, then corrects individual layer
bit widths by inspecting tensor shapes on load.

## Integration checklist

When adding JANG support to your framework:

- [ ] Parse `jang_config.json` and assert `format == "jang"`
- [ ] Load tensor metadata from `model.safetensors.index.json`
- [ ] For each quantized tensor name, locate `.weight` + `.scales` + `.biases`
- [ ] Infer per-tensor bit width from shape: `bits = (weight.shape[-1] * 32) // in_dim`
- [ ] Unpack bits little-endian from uint32 words per the formula above
- [ ] Apply affine dequant: `W = q * scales[group] + biases[group]`
- [ ] For JANGTQ models: load `.tq_codebook` and apply codebook lookup before scaling
- [ ] If `<name>.tq_hadamard_seed` is present: apply inverse Hadamard post-dequant
- [ ] Honor the chat template (inline in `tokenizer_config.json`, or `chat_template.jinja`)
- [ ] Preserve `config.json` fields `tool_call_parser`, `reasoning_parser`, `enable_thinking`
      if your framework supports those semantics
- [ ] For VL models: use `preprocessor_config.json`
- [ ] For video VL models: additionally use `video_preprocessor_config.json`

## Performance notes

- JANG v2 models are mmap-friendly: `mx.load()` maps the file without copying data.
  Load time is near-instant regardless of model size.
- Dequantize on-the-fly during the matmul — do not materialize the full FP16 weight tensor.
  The reference Metal kernel at
  `jang-runtime/Sources/JANGCoreMetal/JangV2QuantMatmul.metal` does this.
- JANGTQ adds a codebook lookup before the matmul. The kernel at
  `jang-runtime/Sources/JANGCoreMetal/JANGTQMatmul.metal` fuses codebook read + scale + matmul.
- For MoE models with 512+ experts, use bfloat16 activations (float16 overflows on high fan-out
  aggregation). See `jang-tools/jang_tools/architectures.py` for the affected model types.

## Questions

File an issue at https://github.com/jjang-ai/jangq/issues with the `porting` label. Attach your
target framework name — the community can help faster if we know what runtime you are integrating.
