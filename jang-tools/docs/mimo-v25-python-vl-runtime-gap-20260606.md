# MiMo V2.5 JANG_2L Python VL/audio runtime gap

Date: 2026-06-06

This note records the current boundary between the MiMo V2.5 JANG_2L artifact and the Python/vMLX runtime.

## Current artifact state

Canonical bundle synced from Max2:

```text
/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

The bundle contains the expected text, vision, and audio sidecars:

- `config.json`
- `tokenizer_config.json`
- `generation_config.json`
- `preprocessor_config.json`
- `model.safetensors.index.json`
- `audio_tokenizer/`
- `visual.*` weights
- `audio_encoder.*` weights
- `speech_embeddings.*` weights

On 2026-06-06 the local bundle was checksum-synced in place from:

```text
erics-m5-max2.local:/Users/eric/.mlxstudio/models/JANGQ-AI/MiMo-V2.5-JANG_2L
```

Rsync replaced the mismatched local `config.json` and deleted stale local `config.json.pre-text-runtime-metadata-20260606`. The large safetensor payload was checksummed and did not require transfer.

## Current Python/vMLX proof

vMLX proof artifacts in `/Users/eric/mlx/vllm-mlx-finite-launch-guard`:

```text
build/current-mimo-v25-jang2l-local-sync-runtime-proof-20260606.json
build/current-mimo-v25-jang2l-local-sync-image-proof-20260606.json
```

Text proof verdict: `PASS`.

Rows passed:

- `/health`
- `/v1/models`
- `/v1/models/{id}/capabilities`
- Chat Completions France prompt
- Chat Completions arithmetic prompt
- Responses basic prompt
- no parser leak flags
- no loop flags

Image proof verdict: `PASS_FAIL_CLOSED`.

The image request returns HTTP `400` with:

```text
/v1/chat/completions received unsupported media modality image because the loaded runtime is text-only. Supported modalities: text.
```

A text request after the rejected image returns HTTP `200` and visible text `recovered`.

## Missing runtime implementation

The current Python adapter in `vmlx_engine.models.mllm` registers MiMo under `mlx_vlm.models.mimo_v2`, but it is a text-only compatibility wrapper over `jang_tools.mimo_v2.mlx_model`.

The missing pieces are explicit in that wrapper:

- `VisionConfig` is an empty stub.
- `AudioConfig` is an empty stub.
- `VisionModel.sanitize()` filters out `visual.*`, `audio_encoder.*`, and `speech_embeddings.*`.
- `Model.load_weights()` skips `visual.*`, `audio_encoder.*`, `speech_embeddings.*`, and `model.mtp.*`.
- `Model.get_input_embeddings(..., pixel_values=...)` raises `UnsupportedMediaModalityError`.
- `Model.__call__(..., pixel_values=...)` raises `UnsupportedMediaModalityError`.
- There is no `jang_tools/mimo_v2/mimo_v2_multimodal.py` in the installed package or local `jang-tools` tree.

Therefore MiMo VL/audio failure in Python/vMLX is a runtime implementation gap, not a missing sidecar upload.

## Required implementation before advertising MiMo media

Build real MiMo multimodal runtime code before flipping capabilities:

- Vision config parser for the 28-block ViT tower.
- Conv3D patch embedding with `[t=2, h=16, w=16]` layout.
- Vision GQA attention with full/window pattern and sink handling.
- Vision merger from 5120 to 4096 hidden size.
- Image/video placeholder expansion matching the ChatML template tokens.
- Audio tokenizer loading and feature extraction path.
- Speech embedding summation over 20 channels.
- Six-layer Qwen2-style audio local transformer.
- Audio projection MLP to 4096 hidden size.
- Mixed media/text `inputs_embeds` construction.
- Position/RoPE handling for media-expanded prompts.
- Media-aware prefix/cache salt and L2 restore proof.
- Streaming/non-streaming parity after media turns.
- Tool-call and JSON/XML parser proof after media turns.
- Post-error text recovery proof.

Until those rows pass, capabilities must keep MiMo runtime modalities as `text` and media sidecars as `preserved_unwired`.
