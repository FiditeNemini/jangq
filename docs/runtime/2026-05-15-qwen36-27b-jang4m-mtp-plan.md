# Qwen3.6-27B-JANG_4M-MTP Plan - 2026-05-15

## Scope

Build one working Qwen3.6-only affine JANG artifact with MTP weights preserved:

`Qwen3.6-27B-JANG_4M-MTP`

Qwen3.5 artifacts are historical references only and are not runtime targets.

## Current state

Source download target:

`/Volumes/EricsLLMDrive/Sources/Qwen/Qwen3.6-27B`

Local working mirror:

`/Users/eric/models/Sources/Qwen/Qwen3.6-27B`

Downloaded source config already confirms:

- `model_type=qwen3_5`
- `architectures=["Qwen3_5ForConditionalGeneration"]`
- `text_config.num_hidden_layers=64`
- `text_config.mtp_num_hidden_layers=1`
- `text_config.mtp_use_dedicated_embeddings=false`
- `text_config.hidden_size=5120`
- `text_config.intermediate_size=17408`
- `text_config.vocab_size=248320`
- source index has `15` `mtp.*` tensors and `333` `model.visual.*` tensors

Current local JANG artifact:

`/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-CRACK`

This artifact has the MTP config fields but no `mtp.*` tensor keys, so it is not
activatable as-is.

## Build result

Built artifact:

`/Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-MTP`

Conversion command:

```sh
cd /Users/eric/jang/jang-tools
uv run python -m jang_tools convert \
  /Users/eric/models/Sources/Qwen/Qwen3.6-27B \
  -o /Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-MTP \
  -p JANG_4M \
  --force-dtype bf16
```

Verified output:

- 29 indexed safetensors shards;
- actual indexed shard bytes: `17820460160` (`16.6` GiB);
- `runtime.total_weight_bytes=17820460160`;
- `runtime.total_weight_gb=16.6`;
- `runtime.bundle_has_mtp=true`;
- `runtime.mtp_layers=1`;
- `runtime.mtp_mode=preserved_enabled`;
- converted bundle has `31` MTP runtime tensor entries;
- converted bundle has `333` `vision_tower.*` tensor entries;
- `preprocessor_config.json` and `video_preprocessor_config.json` are present.

Strict probe:

```sh
cd /Users/eric/jang/jang-tools
uv run --quiet python examples/mtp/qwen36_mtp_runtime_probe.py \
  /Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-MTP \
  --strict
```

Result:

- `ok=true`;
- `errors=[]`.

Generation smoke through the JANG VLM loader:

- text-only prompt: `What is 2 + 2? Answer with only the number.`
- output: `4`;
- image prompt over a generated red square:
  `What color is the square? Answer with only one word.`
- output: `red`.

The current runtime proof is plain autoregressive decode with MTP weights
preserved. Speculative MTP accept/reject wiring remains a separate runtime task.

## Artifact contract

The new artifact must contain:

- base 64 Qwen3.6 27B text/hybrid layers as current JANG_4M does;
- `mtp.*` tensor keys from source;
- Qwen `model.visual.*` VL tensors from source;
- `preprocessor_config.json` and `video_preprocessor_config.json`;
- `config.json` preserving `text_config.mtp_num_hidden_layers=1`;
- `jang_config.json` with explicit runtime MTP metadata.

Initial runtime metadata:

```json
{
  "runtime": {
    "bundle_has_mtp": true,
    "mtp_layers": 1,
    "mtp_mode": "preserved_enabled"
  },
  "mtp": {
    "kept": true,
    "enabled": true,
    "num_layers": 1
  }
}
```

`preserved_enabled` means the bundle keeps MTP tensors and advertises them for
MTP-aware runtimes. Runtimes without an accept/reject speculative loop may still
fall back to plain autoregressive decode, but converters must not zero the
nested Qwen MTP fields or drop the weights.

## Converter plan

Use the regular affine JANG converter path, not JANGTQ.

Target file:

`jang-tools/jang_tools/convert.py`

Required behavior:

- include source tensors whose names start with `mtp.`;
- quantize MTP 2D matmuls with the same affine JANG policy used for comparable
  base-layer tensors;
- preserve MTP norms/non-2D control tensors as passthrough fp16 where required;
- preserve Qwen `model.visual.*` tensors as fp16 passthrough so image/video VL
  paths are still loadable after conversion;
- keep `text_config.mtp_num_hidden_layers=1` in `config.json`;
- stamp `runtime.bundle_has_mtp=true`, `runtime.mtp_layers=1`,
  `runtime.mtp_mode=preserved_enabled`;
- do not rely on Qwen3.5 naming without checking the Qwen3.6 source index.

Expected Qwen-family MTP namespace to confirm after download:

- `mtp.fc.*`
- `mtp.layers.0.self_attn.*`
- `mtp.layers.0.mlp.*`

## Inspector/example plan

Target files:

- `jang-tools/examples/mtp/inspect_mtp_bundle.py`
- `jang-tools/examples/mtp/qwen36_mtp_runtime_probe.py`
- `jang-tools/examples/mtp/README.md`

The inspector must recognize Qwen-style nested fields:

- `text_config.mtp_num_hidden_layers`
- `text_config.mtp_use_dedicated_embeddings`

The runtime probe must start as a proof harness, not product wiring:

- load metadata and MTP tensors by name;
- confirm MTP tensor shapes match hidden size and expected Qwen layer count;
- run normal decode with MTP disabled first;
- add one-token greedy draft only behind an explicit flag;
- verify any drafted token with the base model before acceptance;
- discard draft state on reject or cancellation;
- never mix MTP draft cache with the accepted base cache.

## vMLX follow-up

The vMLX production-family audit helper currently keys on
`config.num_nextn_predict_layers`. Qwen3.6 uses
`text_config.mtp_num_hidden_layers`, so the audit must be extended before vMLX
can report Qwen MTP accurately.

Target file later:

`/Users/eric/mlx/vllm-mlx/tests/cross_matrix/run_production_family_audit.py`

Required change later:

- detect `cfg.text_config.mtp_num_hidden_layers`;
- report `weights_present_runtime_unwired` only when `mtp.*` tensors exist;
- report metadata inconsistency when config says MTP but the index has no
  `mtp.*` tensors.

## First commands after source download completes

Static source census:

```sh
python3 jang-tools/examples/mtp/inspect_mtp_bundle.py \
  /Volumes/EricsLLMDrive/Sources/Qwen/Qwen3.6-27B
```

Expected before conversion:

- `configured_mtp_layers=1`
- `artifact_has_mtp_weights=true`
- nonzero `mtp_tensor_count`
- `artifact_has_vision_weights=true`
- nonzero `visual_tensor_count`

If the source has config MTP fields but no `mtp.*` tensors, stop: there is no
native Qwen3.6 27B MTP to preserve from that source.

If the source or output has Qwen conditional-generation metadata but no
`model.visual.*` tensors, stop before upload: the VL path has been stripped or
the wrong source was used.
