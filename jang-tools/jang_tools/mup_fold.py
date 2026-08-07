"""Fold muP scaling multipliers into weights at JANG conversion time.

Falcon-H1 (and other muP-parameterized checkpoints) declare scalar/vector
multipliers in config.json that the HF runtime applies to activations at
every forward pass. Downstream MLX loaders (mlx-lm, vmlx-swift) instead
fold those multipliers into the weights once, at load — but they gate the
fold on the conv1d tensor still being in HF layout:

    c1d = weights["model.layers.0.mamba.conv1d.weight"]
    if c1d.shape[-1] <= c1d.shape[1]:
        return weights  # "already sanitized" -> skips the muP fold

JANG conversion writes conv1d in MLX layout, so those loaders skip the
fold and the model runs completely unscaled (mlxstudio#129: fluent but
degenerate output, GSM8K 0%). The fold cannot be repaired at load time
for packed-quantized tensors, so it must happen here, on the float source
weights, before quantization. For affine quantization w = q*scale + bias
this is exact: folding the float tensor by m is identical to scale*=m,
bias*=m on the quantized artifact.

Multiplier semantics follow the HF reference (modeling_falcon_h1.py), not
mlx-lm's sanitize, which has two bugs we must not inherit:
  - key_multiplier is dead code there (gated on "key_proj", Falcon names
    it "k_proj"; upstream fix ml-explore/mlx-lm#1622),
  - v_proj is never folded, though attention_in_multiplier scales the
    whole attention-module input (q, k AND v). Real Falcon-H1 configs
    ship attention_in_multiplier == 1.0, which is why this goes unnoticed.
"""

from __future__ import annotations

import numpy as np


def _cfg_float(config: dict, key: str) -> float:
    value = config.get(key)
    if value is None:
        return 1.0
    return float(value)


class MupFold:
    """Per-tensor muP multiplier folding for one source checkpoint."""

    def __init__(self, config: dict):
        self._attention_in = _cfg_float(config, "attention_in_multiplier")
        self._attention_out = _cfg_float(config, "attention_out_multiplier")
        self._key = _cfg_float(config, "key_multiplier")
        self._embedding = _cfg_float(config, "embedding_multiplier")
        self._lm_head = _cfg_float(config, "lm_head_multiplier")
        self._ssm_in = _cfg_float(config, "ssm_in_multiplier")
        self._ssm_out = _cfg_float(config, "ssm_out_multiplier")
        mlp = config.get("mlp_multipliers") or [1.0, 1.0]
        self._mlp_gate = float(mlp[0])
        self._mlp_down = float(mlp[1])
        self._mup_vector = self._build_mup_vector(config)
        self.folded: dict[str, str] = {}

    @classmethod
    def from_config(cls, config: dict) -> "MupFold | None":
        """Return a fold instance for muP architectures, else None."""
        model_type = str(config.get("model_type", ""))
        if model_type != "falcon_h1":
            return None
        return cls(config)

    @staticmethod
    def _build_mup_vector(config: dict) -> np.ndarray | None:
        """Per-channel multiplier over in_proj's zxBCdt output sections.

        Mirrors HF's get_mup_vector: [z, x, B, C, dt] sections sized
        [d_ssm, d_ssm, G*S, G*S, heads] with ssm_multipliers[0..4].
        """
        multipliers = config.get("ssm_multipliers")
        if not multipliers:
            return None
        if len(multipliers) != 5:
            raise ValueError(
                f"ssm_multipliers must have 5 entries, got {multipliers!r}"
            )
        d_ssm = config.get("mamba_d_ssm")
        if not d_ssm:
            d_ssm = int(config.get("mamba_expand", 2)) * int(config["hidden_size"])
        groups_time_state = int(config["mamba_n_groups"]) * int(
            config["mamba_d_state"]
        )
        num_heads = int(config["mamba_n_heads"])
        sizes = [int(d_ssm), int(d_ssm), groups_time_state, groups_time_state, num_heads]
        return np.concatenate(
            [
                np.full(size, float(mult), dtype=np.float64)
                for size, mult in zip(sizes, multipliers)
            ]
        )

    def _rule_for(self, tensor_name: str) -> tuple[object, str] | None:
        """Return (multiplier, label) for the tensor, or None.

        Input-side scalars fold into weights only; output-side factors
        must also fold into layer biases, so `.bias` rules carry only the
        output-side component.
        """
        name = tensor_name
        is_bias = name.endswith(".bias")
        if not (name.endswith(".weight") or is_bias):
            return None
        stem = name.rsplit(".", 1)[0]

        if stem.endswith("embed_tokens"):
            return (self._embedding, "embedding_multiplier")
        if stem.endswith("lm_head"):
            return (self._lm_head, "lm_head_multiplier")
        if stem.endswith(("q_proj", "v_proj")):
            # attention_in_multiplier scales the attention-module INPUT
            # (input-side): exact on weights, biases untouched.
            if is_bias:
                return None
            return (self._attention_in, "attention_in_multiplier")
        if stem.endswith(("k_proj", "key_proj")):
            if is_bias:
                return (self._key, "key_multiplier")
            return (
                self._attention_in * self._key,
                "attention_in_multiplier*key_multiplier",
            )
        if stem.endswith("o_proj"):
            return (self._attention_out, "attention_out_multiplier")
        if stem.endswith("out_proj"):
            return (self._ssm_out, "ssm_out_multiplier")
        if stem.endswith("gate_proj"):
            return (self._mlp_gate, "mlp_multipliers[0]")
        if stem.endswith("down_proj"):
            return (self._mlp_down, "mlp_multipliers[1]")
        if stem.endswith("in_proj"):
            if self._mup_vector is None:
                raise ValueError(
                    f"muP fold: {tensor_name} needs ssm_multipliers, but the "
                    "source config.json does not declare them"
                )
            if is_bias:
                return (self._mup_vector, "mup_vector")
            return (
                self._ssm_in * self._mup_vector[:, None],
                "ssm_in_multiplier*mup_vector",
            )
        return None

    def apply(self, tensor_name: str, weights: np.ndarray) -> np.ndarray:
        """Return weights with the tensor's muP multiplier folded in."""
        rule = self._rule_for(tensor_name)
        if rule is None:
            return weights
        multiplier, label = rule
        if isinstance(multiplier, np.ndarray):
            expected_rows = multiplier.shape[0]
            if weights.shape[0] != expected_rows:
                raise ValueError(
                    f"muP fold: {tensor_name} has {weights.shape[0]} output "
                    f"channels but the config-derived mup_vector has "
                    f"{expected_rows} — refusing to fold a mismatched vector"
                )
        elif float(multiplier) == 1.0:
            return weights
        self.folded[tensor_name] = label
        return (weights.astype(np.float64) * multiplier).astype(np.float32)
