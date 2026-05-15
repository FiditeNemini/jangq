# MTP Runtime Examples

Small, low-RAM helpers for checking whether a bundle has MTP and whether a
profile is realistic for a target memory size.

## Inspect A Bundle

```sh
python3 jang-tools/examples/mtp/inspect_mtp_bundle.py /path/to/model-or-bundle
```

This reads config and safetensors index metadata only. It does not load model
weights.

For the active Qwen3.6 27B MTP lane:

```sh
python3 jang-tools/examples/mtp/inspect_mtp_bundle.py \
  /Volumes/EricsLLMDrive/Sources/Qwen/Qwen3.6-27B
```

Expected source-side signal before building `Qwen3.6-27B-JANG_4M-MTP`:

- `configured_mtp_layers` is `1`;
- `artifact_has_mtp_weights` is `true`;
- `mtp_tensor_count` is nonzero.
- `artifact_has_vision_weights` is `true`;
- `visual_tensor_count` is nonzero.

If config advertises MTP but `artifact_has_mtp_weights` is `false`, runtime MTP
cannot be activated from that artifact; recover a source with `mtp.*` weights or
stop.

For Qwen conditional-generation sources, also stop if vision metadata exists but
`artifact_has_vision_weights` is `false`; that means the VL tower was stripped or
the wrong source tree is being inspected.

## Qwen3.6 MTP/VL Probe

```sh
python3 jang-tools/examples/mtp/qwen36_mtp_runtime_probe.py \
  /Users/eric/models/Sources/Qwen/Qwen3.6-27B \
  --strict
```

This is still a metadata/header proof, not a generation proof. In strict mode it
fails if Qwen MTP config exists without `mtp.*` weights, if conditional-
generation metadata exists without visual tensors, or if a converted bundle
still advertises the stale `runtime.mtp_mode=preserved_disabled`.

For the built local `Qwen3.6-27B-JANG_4M-MTP` artifact:

```sh
python3 jang-tools/examples/mtp/qwen36_mtp_runtime_probe.py \
  /Users/eric/models/dealign.ai/Qwen3.6-27B-JANG_4M-MTP \
  --strict
```

Expected converted-bundle signal:

- `ok=true`;
- `runtime.mtp_mode=preserved_enabled`;
- `runtime.total_weight_gb=16.6`;
- `mtp_tensor_count=31`;
- `visual_tensor_count=333`;
- `has_preprocessor_config=true`;
- `has_video_preprocessor_config=true`.

## Qwen3.6 MXFP4 MTP Patch

For the local MXFP4 sibling artifact:

```sh
cd /Users/eric/jang/jang-tools
uv run python examples/mtp/patch_qwen36_mxfp4_mtp.py --replace
```

This copies the known-good `Qwen3.6-27B-MXFP4-CRACK` bundle, appends native
MTP tensors from `/Users/eric/models/Sources/Qwen/Qwen3.6-27B`, quantizes 2D
MTP matmuls with MLX `mode="mxfp4"`, and stamps
`runtime.mtp_mode=preserved_enabled`.

Expected local output:

- `/Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-MTP`;
- `runtime.total_weight_gb=14.38`;
- `mtp_tensor_count=23`;
- `visual_tensor_count=333`;
- text smoke output: `4`;
- image smoke output for a red square: `red`.

Current autoregressive runtime filters `mtp.*` tensors while loading because
today's `mlx-vlm` Qwen3.6 class does not expose MTP modules. The tensors remain
in the bundle for an MTP-aware speculative runtime.

## Estimate Hy3 Fit

```sh
python3 jang-tools/examples/mtp/estimate_jangtq_fit.py \
  /Users/eric/models/Tencent/Hy3-preview \
  --profile JANGTQ_K \
  --device-gb 128
```

The estimator is intentionally conservative. It should guide conversion order,
not replace measured runtime memory proofs.
