# ZAYA1-8B Runtime Prep

Examples and handoff notes for `Zyphra/ZAYA1-8B`.

## Local Source

Default source path:

```sh
/Users/eric/jang/models/Zyphra/ZAYA1-8B
```

The download is pinned to Hugging Face commit:

```text
2b008c91b7f0004636394dbd2d7b4ca2c2e820e7
```

Remote `main` reported the same SHA with `lastModified=2026-05-06T20:02:30Z`
during the local prep pass.

## Files

| File | Purpose |
|---|---|
| `ATTENTION_ARCHITECTURE.md` | Focused CCA attention, cache, batching, paged-KV, prefix-cache, TurboQuant-KV, and quantization policy notes. |
| `00_inspect_source.py` | Reads config, tokenizer/template metadata, safetensor headers, layer schedule, and cache/quantization policy without loading tensors. |
| `01_python_vllm_smoke.py` | OpenAI-compatible smoke client for the official Zyphra vLLM runtime. Exercises reasoning-on/off and concurrent requests. |
| `02_python_runtime_contract.py` | Header-only JSON contract for cache geometry, batching, prefix, paged KV, and TurboQuant KV integration. |
| `ZayaRuntimeContract.swift` | Standalone Swift contract script for vMLX integration owners. Prints the exact cache, batching, prefix-cache, paged-KV, and TurboQuant-KV policy. |

## Architecture Summary

`model_type=zaya` is not a stock MLX or mlx-lm architecture today.

The 80 decoder layers alternate:

```text
even layers: ZayaDecoderATTLayer with CCA attention
odd layers:  ZayaDecoderMLPLayer with top-1 ZAYA MoE
```

Key dimensions from the real config:

```text
hidden_size = 2048
num_hidden_layers = 80
attention layers = 40
moe layers = 40
num_attention_heads = 16
cca_num_q_heads = 8
num_query_groups / kv heads = 2
head_dim = 128
num_experts = 16
moe_router_topk = 1
zaya_mlp_expansion = 256
max_position_embeddings = 131072
rope_theta = 5000000
```

CCA attention has two state families:

```text
standard KV per attention layer: keys/values [B, 2, T, 128]
CCA inner state: conv_state [B, 1280, 2] and prev_hs [B, 2048]
```

The official vLLM path treats ZAYA as a hybrid model and explicitly disables prefix caching. Do not enable prefix caching in JANG/vMLX until the cache key and stored payload include CCA inner state, not only standard KV.

## Compatibility Policy

| Feature | Status | Notes |
|---|---|---|
| Continuous batching | Compatible with per-slot hybrid cache | Each sequence needs independent KV plus CCA conv/prev_hs state. Batch views must preserve per-sequence offsets. |
| Paged KV | Compatible for standard attention KV | Paged blocks must not pretend CCA state was restored unless conv/prev_hs were restored too. |
| Prefix cache | Disabled for first port | Official vLLM asserts prefix caching off. Enable only after exact restore tests pass across prompt reuse. |
| TurboQuant KV | KV-only experimental | Encode standard attention KV only. Keep CCA conv/prev_hs in float32 as requested by the model card/runtime. |
| JANGTQ weights | New arch work required | Use pre-stacked routed experts and preserve router/CCA precision floors. |
| MXFP4 weights | New arch work required | A safe first bundle should keep CCA convs, router path, norms, residual scaling, and `temp` passthrough. |

## Conversion Notes

Converter entry points:

```sh
jang-convert-zaya-mxfp4 /Users/eric/jang/models/Zyphra/ZAYA1-8B /path/to/ZAYA1-8B-MXFP4 --dry-run
jang-convert-zaya-jangtq /Users/eric/jang/models/Zyphra/ZAYA1-8B /path/to/ZAYA1-8B-JANGTQ3 JANGTQ3 --dry-run
```

Full conversion requires the JANG MLX dependencies. The dry-run path only scans
headers and is safe to use before launching a large local conversion.

For JANGTQ, split each expert `linear_fc1.weight` into gate/up halves and map:

```text
linear_fc1[:2048, :] -> gate_proj
linear_fc1[2048:, :] -> up_proj
linear_fc2           -> down_proj
```

Emit routed experts pre-stacked under one switch-MLP namespace per layer/projection. Do not write per-expert `.tq_*` keys for new bundles.
For 3-bit profiles, ZAYA's 2048-wide experts require row-wise padded packing.
The converter writes `jang_config.tq_in_features` so the Swift sidecar builder
uses the exact logical width instead of the padded packed-column width.

Recommended first JANGTQ profile:

```text
routed experts: 2/3/4-bit MXTQ by profile
CCA attention linears and o_proj: 8-bit affine, group_size 32
embed/lm_head: 8-bit affine, group_size 32
router path: fp16/bf16 passthrough for first coherence pass
conv_qk, temp, norms, residual scaling, balancing_biases: passthrough
```

For MXFP4, start with 4-bit affine `group_size=32` for large 2D linears, but keep the same passthrough list until a real coherence run proves lower precision is safe.

Every JANGTQ output must include `jangtq_runtime.safetensors` before upload or
Swift runtime use:

```sh
python3 -m jang_tools.build_jangtq_sidecar /path/to/ZAYA1-8B-JANGTQ3
```
