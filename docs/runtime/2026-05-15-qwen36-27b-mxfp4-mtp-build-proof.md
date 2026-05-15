# Qwen3.6-27B-MXFP4-MTP Build Proof - 2026-05-15

## Scope

Build a native MLX MXFP4 Qwen3.6 27B VLM bundle with Qwen MTP tensors
preserved and runtime metadata marked enabled:

`/Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-MTP`

This is a sibling to `Qwen3.6-27B-JANG_4M-MTP`. It starts from the known-good
local MXFP4 CRACK artifact and appends MTP tensors from the BF16 source.

## Inputs

Source with native MTP and VL tensors:

`/Users/eric/models/Sources/Qwen/Qwen3.6-27B`

Known-good MXFP4 base:

`/Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK`

The base MXFP4 artifact loads and generates through stock `mlx_vlm`, but it has
no `mtp.*` tensor keys. The source has 15 native `mtp.*` tensors and 333 visual
tensors.

## Build Command

```sh
cd /Users/eric/jang/jang-tools
uv run python examples/mtp/patch_qwen36_mxfp4_mtp.py --replace
```

The patch helper:

- copies the known-good MXFP4 base bundle;
- appends an extra `format=mlx` safetensors shard for MTP;
- quantizes 2D MTP matmuls with native MLX `mx.quantize(..., mode="mxfp4")`;
- stores MTP norm/control tensors as fp16;
- fixes stale Qwen EOS id `248044` to `248046`;
- writes `jang_config.json` with `runtime.mtp_mode=preserved_enabled`.

## Output

Build output:

```json
{
  "output": "/Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-MTP",
  "source_mtp_tensors": 15,
  "runtime_mtp_entries": 23,
  "quantized_linears": 8,
  "passthrough_tensors": 7,
  "shards": 4,
  "total_weight_bytes": 15439720452,
  "total_weight_gb": 14.38
}
```

Runtime metadata:

```json
{
  "total_weight_bytes": 15439720452,
  "total_weight_gb": 14.38,
  "bundle_has_mtp": true,
  "mtp_layers": 1,
  "mtp_mode": "preserved_enabled"
}
```

Strict probe:

```sh
cd /Users/eric/jang/jang-tools
uv run python examples/mtp/qwen36_mtp_runtime_probe.py \
  /Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-MTP \
  --strict
```

Result:

- `ok=true`;
- `errors=[]`;
- `mtp_tensor_count=23`;
- `visual_tensor_count=333`;
- `has_preprocessor_config=true`;
- `has_video_preprocessor_config=true`.

## Runtime Proof

Generation through `jang_tools.loader.load_jang_vlm_model`:

- text prompt: `Answer with only the number: 2+2=`
- output: `4`
- red-square image prompt: `What is the dominant color? Answer one word.`
- output: `red`

The loaded runtime uses native MXFP4 modules:

- 497 `QuantizedLinear` modules;
- 1 `QuantizedEmbedding` module;
- mode `mxfp4`;
- bits `4`;
- group size `32`;
- uint8 MXFP scales;
- no affine bias placeholders.

## Runtime Boundary

Current autoregressive decode filters preserved `mtp.*` tensors at load time
because the current `mlx-vlm` Qwen3.6 model class does not expose MTP modules.
That is expected for today's decoder. The artifact itself keeps MTP tensors and
advertises `preserved_enabled` so an MTP-aware runtime can wire the speculative
draft/verify path later without rebuilding the bundle.
