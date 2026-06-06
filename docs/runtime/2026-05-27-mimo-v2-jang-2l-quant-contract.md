# MiMo-V2.5 JANG_2L Quantization Contract

Date: 2026-05-27

## Source Truth

Local source: `/Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5`

Verified by `jang-tools/tests/mimo_v2_contract_test.py`:

- `model_type=mimo_v2`
- 48 decoder layers, 256 routed experts, top-8
- Full attention layers: 9
- SWA layers: 39
- Full attention KV heads: 4
- SWA KV heads: 8
- Full qkv shape: `(13568, 4096)` = q `12288`, k `768`, v `512`
- SWA qkv shape: `(14848, 4096)` = q `12288`, k `1536`, v `1024`
- `attention_value_scale=0.707`
- partial RoPE factor `0.334`, applied to the first 64 dims of q/k
- full-attention `rope_theta=10000000`
- SWA `swa_rope_theta=10000`
- SWA attention sink bias is present; full-attention sink is not
- `visual.*`, `audio_encoder.*`, and `model.mtp.*` tensors are present
- text `o_proj` weights are ignored by source FP8 quantization and must remain bf16/passthrough

The upstream README's KV-head table is not the source of truth here. Config plus tensor shapes win.

## Current Local Bundle Status

Bundle: `/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L`

Verified on 2026-05-27:

- structural verifier passes: `109184` tensor keys across `106` shards
- safetensor payload: `104.63 GB`
- Finder/du footprint: `98G`
- `model.mtp.*` tensors removed: `mtp_keys=0`
- vision preserved: `visual_keys=364`
- audio preserved: `audio_encoder_keys=75`, `speech_embeddings=20`, `audio_tokenizer/` present
- lazy `mlx_lm` load passes with the in-tree `mimo_v2` registration
- runtime metadata says `bundle_has_mtp=false`, `mtp_mode=absent`
- generation quality is not proven yet; previous report of gibberish means this is not publish-ready

The first runtime ablation switch is:

```sh
JANG_MIMO_DISABLE_SINK=1
```

That switch keeps the 39 SWA sink-bias tensors loadable but bypasses the sink path in attention forward. Use it for a normal-vs-sink-off prompt comparison before deeper layer diffs.

## JANG_2L Quantization Policy

Use an affine JANG_2L bundle first. Do not start with JANGTQ until the MLX model path is coherent.

Required policy:

- routed experts: affine 2-bit for bulk expert weights
- expert floors: do not let router-critical or residual-sensitive pieces collapse below the established floor; gate/up/down floors must be explicit in metadata
- attention qkv: 8-bit affine
- attention o_proj: bf16 passthrough
- embeddings and lm_head: 8-bit affine unless a source-backed incompatibility is found
- router gate and `e_score_correction_bias`: bf16/fp16 passthrough
- norms and biases: passthrough
- visual tower: passthrough for first working bundle
- audio encoder and audio tokenizer assets: preserve/copy for first working bundle; do not silently drop
- MTP tensors: dropped for the current audio+vision bundle. Do not auto-enable speculative decoding; runtime mode is `absent`.

## JANG_2K / K Variant

The converter supports a quality variant:

```sh
python -m jang_tools.mimo_v2.convert_jang \
  --src /Volumes/EricsLLMDrive/jangq-ai/sources/MiMo-V2.5 \
  --dst ~/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2K \
  --profile 2k \
  --drop-mtp
```

Policy:

- routed `gate_proj`: 2-bit affine
- routed `up_proj`: 2-bit affine
- routed `down_proj`: 4-bit affine
- attention/embed/lm_head/layer-0 dense: 8-bit affine
- attention `o_proj`, norms, routers, vision, and audio: passthrough as in `JANG_2L`

This is the coherent `4/2/2` K profile for MiMo. It is not built locally yet.

## JANGTQ Status

Do not build MiMo JANGTQ until the affine MLX decode path is coherent. JANGTQ would add TurboQuant hydration and fused routed-expert kernels on top of the same MiMo attention/RoPE/MoE math, so it is the wrong next debug layer while text output is gibberish.

After the affine path is fixed, the expected first JANGTQ candidate is `JANGTQ2` no-MTP with audio+vision preserved. `JANGTQ_K` should follow only as a quality candidate.

## Runtime Metadata

Both `config.json` and `jang_config.json` must stamp enough data for vMLX to avoid guessing:

- `capabilities.family = "mimo_v2"`
- `capabilities.modalities = ["text", "vision", "audio"]`
- `capabilities.cache_type = "kv"`
- `capabilities.reasoning_parser = "think_xml"`
- `capabilities.tool_parser = "xml_function"` and `supports_tools=true`; the source template emits `<tool_call><function=...><parameter=...>` blocks, which map to vMLX `XMLFunctionParser`
- `capabilities.supports_thinking = true`
- `capabilities.supports_tools = true` only if the tokenizer/template contract is verified
- `runtime.bundle_has_mtp = false`
- `runtime.mtp_mode = "absent"`
- base decode is autoregressive; no native accept/reject speculative decode path is available in this bundle
- include `mxtq_bits` and `routed_expert_bits` style fields even for affine/JANG if downstream code relies on them for bit accounting
- include attention subtype facts: full/SWA layer counts, full/SWA KV heads, qkv split sizes, value scale, SWA window, and sink-bias support
- include cache topology facts: hybrid full/SWA KV, prefix cache supported, L2 disk cache supported, TurboQuant KV only for ordinary full-attention `KVCacheSimple` layers, and native rotating KV for SWA layers

## MLX Model Port Requirements

The MLX port must mirror `modeling_mimo_v2.py`, not MiniMax assumptions:

- fused `qkv_proj` split depends on layer type
- q/k head dim is 192; v head dim is 128
- only q/k receive RoPE
- RoPE applies to 64 dims, then concatenates no-RoPE dims back
- value states are multiplied by `attention_value_scale` before cache update
- SWA mask uses window 128
- SWA sink bias adds an extra softmax column and then drops that probability before multiplying V
- no MTP path in base decode
- vision/audio paths can be load-preserved before full multimodal inference is exposed, but they must not be mislabeled as absent

## vMLX Python Engine Acceptance

Before claiming "ready":

- register/resolve `mimo_v2` in the Python runtime path
- model-config registry returns family `mimo_v2`, cache `kv`, and the intended parser/tool policy
- loader does not auto-enable MTP for this bundle
- cache key includes model config/runtime fingerprint
- prefix cache hit works on a repeated prompt
- paged cache hit works with the asymmetric full/SWA KV dimensions
- L2 disk cache writes and restores
- TQ-native disk path is either proven compatible or explicitly skipped
- live TurboQuant KV auto mode does not replace nonstandard cache slots incorrectly
- a short autoregressive text smoke uses bundle `generation_config.json` defaults, with no hidden sampler clamps or forced thinking text

## Speed Status

Only lazy load has been measured in this pass, not decode throughput. Lazy load takes about `0.5s` because weights remain lazy. That is not a token/s benchmark.

Do not publish speed claims until:

- normal vs `JANG_MIMO_DISABLE_SINK=1` prompt smoke identifies whether sink is the gibberish cause
- a coherent decode path runs with the bundle defaults
- token/s is measured separately for prefill and decode
- vMLX prefix cache, paged/L2 cache, and TurboQuant-KV compatibility are tested against the hybrid full/SWA cache topology

## Current Guard

Run:

```sh
uv run --project jang-tools pytest -q jang-tools/tests/mimo_v2_contract_test.py
```

This verifies the current source contract and the FP8 E4M3 `weight_scale_inv` codec against a real expert tensor.

Current verification commands:

```sh
uv run --project jang-tools python -m jang_tools.mimo_v2.verify_bundle \
  /Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L

uv run --project jang-tools pytest -q \
  jang-tools/jang_tools/mimo_v2/tests/test_fp8_codec.py \
  jang-tools/tests/mimo_v2_contract_test.py

uv run --project jang-tools python -m py_compile jang-tools/jang_tools/mimo_v2/*.py
```
