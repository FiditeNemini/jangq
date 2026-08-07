"""Zero-centered RMSNorm shift for Qwen3.5-family conversion (mlxstudio#130).

Qwen3.5 / Qwen3-Next HF checkpoints store zero-centered RMSNorm weights
(gamma - 1); the HF runtime adds +1.0 at every norm. mlx_lm folds the +1.0
into the weights once, in sanitize(), but gates the shift on "looks
unsanitized" heuristics:

    should_shift = has_mtp_weights or has_unsanitized_conv1d

JANG v2 text bundles strip mtp.* keys before sanitize and store conv1d in
MLX layout, so the gate never fires and standard nn.RMSNorm runs with
gamma ~= 0 -> degenerate output (mlxstudio#130; same "already sanitized"
heuristic class as the #129 muP bug).

Fix: apply the +1.0 shift once, at conversion, on the float source
weights, and stamp config.json with jang_norms_pre_shifted so loaders
know. Wrapped-VL checkpoints are excluded: the VL load path
(_load_jang_v2_vlm minimal sanitize) applies the shift unconditionally at
load, so VL bundles must keep raw norms.

The suffix table matches mlx_lm's qwen3_5/qwen3_next sanitize exactly.
linear_attn.norm (RMSNormGated) is deliberately NOT in the table and must
never be shifted.
"""

from __future__ import annotations

NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)

ZERO_CENTERED_TEXT_FAMILIES = {
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "qwen3_next",
}


def mlx_lm_sanitize_would_shift(model_type, tensor_names, conv1d_last_dims):
    """Mirror mlx_lm's sanitize norm-shift gate exactly (mlxstudio#130).

    qwen3_5*: shift iff mtp keys present OR any conv1d is HF layout
    (last dim = kernel width, not 1). qwen3_next: sanitize early-returns
    unless the per-expert HF key exists — that early-return IS its gate.

    Used at conversion (on SOURCE names/shapes: mlx-format sources were
    already shifted by mlx_lm's own conversion, so the shift must be
    skipped there) and at load (per shard, to skip shards mlx_lm's own
    sanitize will shift). No weight-value heuristics.
    """
    if model_type == "qwen3_next":
        return "model.layers.0.mlp.experts.0.up_proj.weight" in tensor_names
    if any("mtp." in name for name in tensor_names):
        return True
    return any(last_dim != 1 for last_dim in conv1d_last_dims)


class ZeroCenteredNormShift:
    """Per-tensor +1.0 RMSNorm shift for one zero-centered checkpoint."""

    def __init__(self, model_type: str) -> None:
        self.model_type = model_type
        self.shifted: list[str] = []

    @classmethod
    def from_config(cls, config: dict, vl_wrapped: bool) -> "ZeroCenteredNormShift | None":
        """Return a shift instance for zero-centered text checkpoints, else None."""
        if vl_wrapped:
            return None
        text_cfg = config.get("text_config") or {}
        model_type = str(text_cfg.get("model_type") or config.get("model_type") or "")
        if model_type not in ZERO_CENTERED_TEXT_FAMILIES:
            return None
        return cls(model_type)

    def apply(self, tensor_name: str, weights):
        """Return weights with +1.0 folded in for matching 1-D norm tensors.

        Works on both numpy and mx arrays (only uses .ndim and __add__).
        """
        if getattr(weights, "ndim", None) != 1:
            return weights
        if not any(tensor_name.endswith(sfx) for sfx in NORM_SUFFIXES):
            return weights
        self.shifted.append(tensor_name)
        return weights + 1.0
