"""ZAYA1-VL bundle loader.

Loads a ZAYA1-VL BF16/MXFP4/affine bundle. JANGTQ/MXTQ ZAYA1-VL bundles
should go through `jang_tools.load_jangtq.load_jangtq_model` so the
routed expert projections are replaced with TurboQuant modules; this
function is for BF16 / affine-only fallback paths or VL-specific tools
that need direct mlx_vlm wiring.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core
from jang_tools.quant_shape_inference import infer_quant_overrides_for_bundle

logger = logging.getLogger(__name__)


def load_zaya1_vl_model(model_path: str | Path, *, lazy: bool = False):
    """Return ``(model, processor)`` for a ZAYA1-VL bundle.

    Requires `mlx_vlm` — imports lazily so the rest of `jang_tools` stays
    importable on environments without VL deps.
    """
    try:
        from mlx_vlm.utils import load as _vlm_load  # type: ignore
    except ImportError as exc:  # pragma: no cover — runtime dep gate
        raise ImportError(
            "load_zaya1_vl_model requires mlx_vlm. "
            "Install via `pip install mlx_vlm`."
        ) from exc

    # Deferred — pulling .model imports mlx_vlm too, so do it after the
    # availability check above so users get a single clean ImportError.
    from .model import register_mlx_vlm_zaya1_vl

    path = Path(model_path)
    register_mlx_vlm_zaya1_vl()

    cfg = json.loads((path / "config.json").read_text())
    try:
        cfg = infer_quant_overrides_for_bundle(path, cfg)
    except Exception as exc:  # noqa: BLE001 — best-effort-parse fallback
        logger.debug("ZAYA1-VL quant-shape inference skipped: %s", exc)

    model, processor = _vlm_load(str(path), processor_config=cfg, lazy=lazy)
    if not hasattr(model, "config"):
        model.config = cfg
    if not lazy:
        getattr(mlx.core, "eval")(model.parameters())
    logger.info(
        "ZAYA1-VL runtime loaded: layers=%s, vision=%s",
        len(getattr(model, "layers", [])),
        type(getattr(model, "vision_tower", None)).__name__,
    )
    return model, processor
