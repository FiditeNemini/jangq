# Python Runtime Registration And vmlx Audit

Created 2026-05-10.

## Scope

Audit the `../vmlx` Python engine surface without editing it, then apply the
fixes in `jang-tools`, which is the canonical JANGTQ runtime package consumed
by vmlx.

## vmlx Reference Finding

`../vmlx` delegates JANGTQ runtime loading to this package:

| vmlx surface | Runtime dependency |
|---|---|
| `vmlx_engine.loaders.load_jangtq` | re-exports `jang_tools.load_jangtq.load_jangtq_model` |
| `vmlx_engine.loaders.load_jangtq_vlm` | re-exports `jang_tools.load_jangtq_vlm.load_jangtq_vlm_model` |
| `vmlx_engine.utils.jang_loader` | detects `.tq_packed` bundles and calls `jang_tools.load_jangtq` |

Therefore custom model registration must be fixed in `jang-tools`, not by
duplicating loaders inside vmlx.

The MiniMax JANGTQ path was already covered by the generic hydrator:
`load_jangtq.py` maps
`model.layers.N.block_sparse_moe.experts.E.{w1,w2,w3}` into
`switch_mlp.{gate_proj,down_proj,up_proj}` and keeps the routed-bit source of
truth in each tensor's `.tq_bits` metadata.

## Fixes Applied

| Family | Fix |
|---|---|
| Hy3 (`hy_v3`) | `import jang_tools.hy3` now registers `mlx_lm.models.hy_v3` immediately, matching `load_jangtq.py`'s production path. |
| ZAYA text (`zaya`) | `import jang_tools.zaya` now registers `mlx_lm.models.zaya`; `load_jangtq.py` auto-imports it before MLX skeleton construction. |
| ZAYA1-VL (`zaya1_vl`) | `load_jangtq_vlm.py` imports `jang_tools.zaya1_vl` before `mlx_vlm` resolves `model_type=zaya1_vl`. The package keeps a lazy dependency boundary when `mlx_vlm` is absent. |
| ZAYA1-VL converters | Console scripts registered for JANGTQ and MXFP4 conversion. |
| MiniMax (`minimax_m2`) | No code change needed in this pass; the existing hydrator path already recognizes the MiniMax per-expert `w1/w2/w3` namespace. |

## Runtime Boundaries

- Hy3 is text-only, KV-cache, MTP-preserved-disabled.
- ZAYA text uses the local ZAYA CCA/MOD runtime; it is not a stock MLX-LM
  family and must be registered before `mlx_lm.utils.load_model`.
- ZAYA1-VL uses the local VLM adapter plus Qwen2.5-VL vision tower; it must go
  through the VLM loader for image requests.
- `../vmlx` and `../vmlx-swift-lm` were inspected as read-only references in
  this pass. No files were edited there.

## Verification

```sh
uv run --project jang-tools pytest -q \
  jang-tools/tests/test_hy3_capabilities.py \
  jang-tools/tests/test_zaya_capabilities.py \
  jang-tools/tests/test_zaya1_vl_jangtq_loader.py

uv run --project jang-tools python -m py_compile \
  jang-tools/jang_tools/hy3/__init__.py \
  jang-tools/jang_tools/zaya/__init__.py \
  jang-tools/jang_tools/zaya1_vl/__init__.py \
  jang-tools/jang_tools/load_jangtq.py \
  jang-tools/jang_tools/load_jangtq_vlm.py
```
