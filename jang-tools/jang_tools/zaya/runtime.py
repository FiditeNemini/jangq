"""ZAYA bundle loader.

Loads a ZAYA BF16/MXFP4/affine bundle. JANGTQ/MXTQ ZAYA bundles should
go through `jang_tools.load_jangtq.load_jangtq_model` so the routed
expert projections are replaced with TurboQuant modules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core
from mlx_lm.utils import load_model, load_tokenizer

from jang_tools.quant_shape_inference import infer_quant_overrides_for_bundle

from .model import register_mlx_lm_zaya

logger = logging.getLogger(__name__)


def load_zaya_model(model_path: str | Path, *, lazy: bool = False):
    """Return ``(model, tokenizer)`` for a ZAYA bundle.

    Parameters
    ----------
    model_path : str | Path
        Directory containing `config.json`, `tokenizer.json`, and the
        sharded safetensors files.
    lazy : bool, default False
        If True, defer parameter materialization to first forward.
    """
    path = Path(model_path)
    register_mlx_lm_zaya()

    cfg = json.loads((path / "config.json").read_text())
    try:
        cfg = infer_quant_overrides_for_bundle(path, cfg)
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        logger.debug("ZAYA quant-shape inference skipped: %s", exc)

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
        # Force MLX lazy graph materialization (mx.eval).
        getattr(mlx.core, "eval")(model.parameters())
    logger.info(
        "ZAYA runtime loaded: layers=%s, cache=CCA(KV+conv_state+prev_hs)",
        len(getattr(model, "layers", [])),
    )
    return model, tokenizer
