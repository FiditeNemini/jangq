"""Zyphra ZAYA1-VL (model_type=zaya1_vl) JANG runtime.

ZAYA1-VL is a vision-LM built on the ZAYA text trunk: 40 layers (vs ZAYA's
80) of CCA attention + top-1 MOD/MoE, plus a Qwen2.5-VL vision tower for
image inputs. Routed experts ship pre-stacked under
`zaya_block.experts.switch_mlp.*`. The trunk's per-block image-token LoRA
paths (rank-8 attn / rank-32 MLP) modulate output ONLY at image-token
positions; text positions decode unmodified.

Public surface mirrors the laguna/mistral3/dsv4/zaya/hy3 packages so
vmlx can re-export this module.

`mlx_vlm` is required at runtime (for `qwen2_5_vl.VisionModel` + base
classes) but the package's symbols are loaded lazily — `from
jang_tools.zaya1_vl import load_zaya1_vl_model` works without VL deps;
the actual call into the model raises a clear ImportError if mlx_vlm
isn't installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .runtime import load_zaya1_vl_model


def register_mlx_vlm_zaya1_vl() -> None:
    """Alias the ZAYA1-VL model module under `mlx_vlm.models.zaya1_vl`.

    No-op when `mlx_vlm` is not installed (the model module itself
    imports mlx_vlm symbols, so eager-registration on a non-VL Python
    environment would crash the parent package's import). When mlx_vlm
    IS available this fires at package-import time so `load_jangtq_model`
    can rely on `from jang_tools.zaya1_vl import …` to register the model.
    """
    try:
        import mlx_vlm  # noqa: F401  type: ignore
    except ImportError:
        return
    from . import model as _model

    _model.register_mlx_vlm_zaya1_vl()


# JANGTQ VLM loading imports this package before mlx_vlm resolves
# `model_type=zaya1_vl`. Safe no-op when mlx_vlm isn't installed.
register_mlx_vlm_zaya1_vl()

__all__ = [
    "Model",
    "ModelConfig",
    "TextConfig",
    "Zaya1VLLanguageModel",
    "LanguageModel",
    "register_mlx_vlm_zaya1_vl",
    "load_zaya1_vl_model",
]


def __getattr__(name):
    """Lazy attribute resolver — defers `from .model import …` (which pulls
    in mlx_vlm) to first access. Lets `from jang_tools.zaya1_vl import
    load_zaya1_vl_model` work in environments without VL deps."""
    if name in {
        "Model",
        "ModelConfig",
        "TextConfig",
        "Zaya1VLLanguageModel",
        "LanguageModel",
    }:
        from . import model as _model

        return getattr(_model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:  # pragma: no cover — type-check time only
    from .model import (
        LanguageModel,
        Model,
        ModelConfig,
        TextConfig,
        Zaya1VLLanguageModel,
    )
