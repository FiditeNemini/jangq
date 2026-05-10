"""Zyphra ZAYA (model_type=zaya) JANG runtime.

ZAYA alternates CCA attention layers with top-1 MOD/MoE layers. Bundles
ship pre-stacked routed experts under `zaya_block.experts.switch_mlp.*`.

Public surface mirrors the laguna/mistral3 packages so vmlx can re-export
this module under `vmlx_engine.loaders.load_zaya`.
"""
from .model import Model, ModelArgs, register_mlx_lm_zaya
from .runtime import load_zaya_model

__all__ = [
    "Model",
    "ModelArgs",
    "register_mlx_lm_zaya",
    "load_zaya_model",
]
