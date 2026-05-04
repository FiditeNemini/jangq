"""Native MLX implementations of mlp1 (vision projector) + sound_projection.

Both modules are tiny (3 weights each). Validated against PyTorch reference
in tests/test_projectors_parity.py — abs tol 1e-3 on bf16, 1e-5 on fp32.

Tensor naming on disk (matches the source modeling.py):
  mlp1.0.weight                       LayerNorm   (vit_hidden * (1/down)**2 = 5120)
  mlp1.1.weight                       Linear      (5120, projector_hidden=20480)
  mlp1.3.weight                       Linear      (20480, llm_hidden=2688)
  sound_projection.norm.weight         RMSNorm     (sound_hidden=1024)
  sound_projection.linear1.weight      Linear      (1024, projection_hidden=4096)
  sound_projection.linear2.weight      Linear      (4096, llm_hidden=2688)
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class VisionMLPProjector(nn.Module):
    """mlp1: post-pixel-shuffle vision-token → LLM-hidden projector.

    Forward:
        x = LayerNorm(x)
        x = Linear1(x)
        x = GELU(x)
        x = Linear3(x)

    Index 2 (GELU) has no weights, hence the "0, 1, 3" key naming.
    """

    def __init__(self, in_dim: int, projector_dim: int, llm_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)  # mlp1.0
        self.fc1 = nn.Linear(in_dim, projector_dim, bias=False)  # mlp1.1
        # mlp1.2 is GELU — no parameters
        self.fc2 = nn.Linear(projector_dim, llm_dim, bias=False)  # mlp1.3

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.fc2(x)
        return x


class SoundProjector(nn.Module):
    """sound_projection: parakeet output → LLM-hidden projector.

    Forward:
        x = RMSNorm(x)
        x = linear1(x)
        x = SquaredReLU(x)         # x = ReLU(x)**2
        x = linear2(x)
    """

    def __init__(
        self,
        sound_hidden: int = 1024,
        projection_hidden: int = 4096,
        llm_hidden: int = 2688,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = nn.RMSNorm(sound_hidden, eps=eps)
        self.linear1 = nn.Linear(sound_hidden, projection_hidden, bias=bias)
        self.linear2 = nn.Linear(projection_hidden, llm_hidden, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.linear1(x)
        x = nn.relu(x) ** 2  # SquaredReLU
        x = self.linear2(x)
        return x


def map_mlp1_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap on-disk `mlp1.{0,1,3}.*` keys to our nn.Module attribute names."""
    rename = {
        "mlp1.0.weight": "layer_norm.weight",
        "mlp1.1.weight": "fc1.weight",
        "mlp1.3.weight": "fc2.weight",
        # Optional biases (some bundles ship with bias)
        "mlp1.0.bias": "layer_norm.bias",
        "mlp1.1.bias": "fc1.bias",
        "mlp1.3.bias": "fc2.bias",
    }
    return {rename[k]: v for k, v in weights.items() if k in rename}


def map_sound_projection_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap on-disk `sound_projection.*` keys."""
    rename = {
        "sound_projection.norm.weight": "norm.weight",
        "sound_projection.linear1.weight": "linear1.weight",
        "sound_projection.linear2.weight": "linear2.weight",
        "sound_projection.linear1.bias": "linear1.bias",
        "sound_projection.linear2.bias": "linear2.bias",
    }
    return {rename[k]: v for k, v in weights.items() if k in rename}
