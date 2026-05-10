"""Hy3-preview bundle loader.

JANGTQ Hy3 bundles MUST be loaded via
`jang_tools.load_jangtq.load_jangtq_model`, which detects
`model_type='hy_v3'` and routes through `register_mlx_lm_hy3` here, then
replaces routed-expert projections with TurboQuant kernels.

For BF16 / affine-only Hy3 bundles, this module's `load_hy3_model`
resolves the standard mlx_lm path.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core
from mlx_lm.utils import load_model, load_tokenizer

from jang_tools.quant_shape_inference import infer_quant_overrides_for_bundle

from .model import register_mlx_lm_hy3

logger = logging.getLogger(__name__)


def load_hy3_model(model_path: str | Path, *, lazy: bool = False):
    """Return ``(model, tokenizer)`` for a Hy3 BF16/affine bundle.

    JANGTQ Hy3 bundles must go through `jang_tools.load_jangtq.load_jangtq_model`
    instead — that function calls `register_mlx_lm_hy3()` then swaps routed
    expert projections with TurboQuantLinear after load.
    """
    path = Path(model_path)
    register_mlx_lm_hy3()

    cfg = json.loads((path / "config.json").read_text())
    try:
        cfg = infer_quant_overrides_for_bundle(path, cfg)
    except Exception as exc:  # noqa: BLE001 — best-effort-parse fallback
        logger.debug("Hy3 quant-shape inference skipped: %s", exc)

    model, loaded_cfg = load_model(
        path,
        model_config=cfg,
        lazy=lazy,
        strict=True,
    )
    tokenizer = load_tokenizer(path, eos_token_ids=loaded_cfg.get("eos_token_id"))
    if not hasattr(model, "config"):
        model.config = loaded_cfg
    if not lazy:
        getattr(mlx.core, "eval")(model.parameters())
    logger.info(
        "Hy3 runtime loaded: layers=%s (MTP layer dropped at sanitize), cache=KV",
        len(getattr(model, "layers", [])),
    )
    return model, tokenizer
