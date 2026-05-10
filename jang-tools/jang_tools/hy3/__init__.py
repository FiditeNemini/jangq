"""Tencent Hy3-preview (model_type=hy_v3) JANG runtime.

Hy3-preview is a 295B/21B-active text-only MoE: 80 dense GQA layers + 1
MTP draft layer, 192 routed experts top-8 + 1 shared expert, sigmoid
router with `expert_bias` (DSV3-style aux-free balancing), 256K context.

The architecture is identical to mlx_lm's `dots1` family with these
naming differences in the bundle:

  Bundle (hy_v3)                              dots1
  ---------------------------------------------------------------
  num_experts                                 n_routed_experts
  num_shared_experts                          n_shared_experts
  route_norm                                  norm_topk_prob
  router_scaling_factor                       routed_scaling_factor
  mlp.router.gate.weight                      mlp.gate.weight
  mlp.expert_bias                             mlp.gate.e_score_correction_bias
  mlp.shared_mlp.{gate,up,down}_proj.*        mlp.shared_experts.{gate,up,down}_proj.*
  mlp.experts.{e}.{gate,up,down}_proj.*       mlp.experts.{gate,up,down}_proj.* (stacked)
  rope_parameters.rope_theta                  rope_theta
  layers[80] (MTP)                            (preserved_disabled - dropped at load)

`Model.sanitize` performs all the renames + drops the MTP layer for the
first runtime pass. Loading goes through standard mlx_lm.utils.load_model
once `register_mlx_lm_hy3()` aliases this module under
`mlx_lm.models.hy_v3`. JANGTQ bundles must additionally route through
`jang_tools.load_jangtq.load_jangtq_model` so routed expert projections
are replaced with TurboQuant kernels.
"""
from .model import Model, ModelArgs, build_args_from_hy3_config, register_mlx_lm_hy3
from .runtime import load_hy3_model

# `load_jangtq_model` imports this package before mlx_lm resolves
# `model_type=hy_v3`; registration must therefore be import-time behavior.
register_mlx_lm_hy3()

__all__ = [
    "Model",
    "ModelArgs",
    "build_args_from_hy3_config",
    "register_mlx_lm_hy3",
    "load_hy3_model",
]
