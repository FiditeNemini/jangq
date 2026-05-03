"""Smoke test for jang_tools.jangrt.jangtq_hydrate.

Builds a tiny synthetic fixture model + weight dict that mirrors the
Laguna / Mistral3 JANGTQ shape, calls hydrate_jangtq, and asserts:

  1. nn.Linear modules with `.tq_packed` keys are SWAPPED to
     TurboQuantLinear (2D packed).
  2. SwitchLinear modules with `.tq_packed` keys are SWAPPED to
     TurboQuantSwitchLinear (3D packed).
  3. The returned weight dict no longer carries `.tq_packed` /
     `.tq_norms` / `.tq_bits` keys for swapped modules.
  4. Modules that DON'T have a `.tq_packed` companion are left alone.
  5. TQ keys for paths that don't exist in the model are silently
     skipped (matches MTP-only weight handling in the real loaders).

The test runs entirely on CPU-side metadata — no Metal kernel calls,
no real weights — so it's safe to run in CI.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

from jang_tools.jangrt.jangtq_hydrate import hydrate_jangtq
from jang_tools.turboquant.tq_kernel import (
    TurboQuantLinear,
    TurboQuantSwitchLinear,
)


class _MoEBlock(nn.Module):
    """Mirror of Laguna's per-layer MoE: dense gate + per-expert switch
    matrices held under `experts.{gate_up_proj, down_proj}`. The bare
    nn.Module subclass `experts` reproduces what Laguna ships."""

    def __init__(self, hidden: int = 64, intermediate: int = 128, n_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.experts = nn.Module()
        # Pre-stacked expert matmul placeholders (3D bare arrays — these
        # will be REPLACED by the helper). We store them as nn.Linear
        # with the right shape so dotted-attr resolution works.
        self.experts.gate_up_proj = nn.Linear(hidden, 2 * intermediate, bias=False)
        self.experts.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _Layer(nn.Module):
    def __init__(self, hidden: int = 64, n_experts: int = 4):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp = _MoEBlock(hidden=hidden, n_experts=n_experts)


class _ToyModel(nn.Module):
    def __init__(self, n_layers: int = 2, hidden: int = 64, n_experts: int = 4):
        super().__init__()
        self.layers = [_Layer(hidden=hidden, n_experts=n_experts) for _ in range(n_layers)]
        self.lm_head = nn.Linear(hidden, 100, bias=False)


def _make_tq_packed(out: int, in_features: int, bits: int):
    """Build a dummy `.tq_packed` tensor matching the converter's shape.
    `tq_packed` is uint32 with shape (out, in_features // (32 // bits))."""
    vals_per_u32 = 32 // bits
    packed_cols = in_features // vals_per_u32
    return mx.zeros((out, packed_cols), dtype=mx.uint32)


def _make_tq_norms(out: int):
    return mx.zeros((out,), dtype=mx.float16)


def test_hydrate_swaps_linear_to_turboquant():
    """A 2D `.tq_packed` triplet swaps the matching nn.Linear to
    TurboQuantLinear and removes the TQ keys from the returned dict."""
    hidden = 64
    model = _ToyModel(n_layers=2, hidden=hidden)
    bits = 2
    weights = {
        # Linear that has TQ — should swap.
        "layers.0.self_attn.q_proj.tq_packed": _make_tq_packed(hidden, hidden, bits),
        "layers.0.self_attn.q_proj.tq_norms": _make_tq_norms(hidden),
        "layers.0.self_attn.q_proj.tq_bits": mx.array([bits], dtype=mx.uint8),
        # Linear without TQ — should be left alone, key passes through.
        "layers.0.self_attn.k_proj.weight": mx.zeros((hidden, hidden)),
    }
    leftover = hydrate_jangtq(model, weights)

    # 1. Module swapped.
    assert isinstance(model.layers[0].self_attn.q_proj, TurboQuantLinear), (
        f"q_proj was {type(model.layers[0].self_attn.q_proj).__name__}, "
        "expected TurboQuantLinear"
    )
    # 2. Untouched module stays nn.Linear.
    assert isinstance(model.layers[0].self_attn.k_proj, nn.Linear)
    # 3. TQ keys consumed; non-TQ key passed through.
    assert "layers.0.self_attn.q_proj.tq_packed" not in leftover
    assert "layers.0.self_attn.q_proj.tq_norms" not in leftover
    assert "layers.0.self_attn.q_proj.tq_bits" not in leftover
    assert "layers.0.self_attn.k_proj.weight" in leftover


def test_hydrate_swaps_3d_to_turboquant_switch():
    """A 3D `.tq_packed` (n_experts axis) swaps the module to
    TurboQuantSwitchLinear."""
    hidden = 64
    intermediate = 128
    n_experts = 4
    bits = 2
    model = _ToyModel(n_layers=1, hidden=hidden, n_experts=n_experts)

    vals_per_u32 = 32 // bits
    weights = {
        "layers.0.mlp.experts.gate_up_proj.tq_packed":
            mx.zeros((n_experts, 2 * intermediate, hidden // vals_per_u32), dtype=mx.uint32),
        "layers.0.mlp.experts.gate_up_proj.tq_norms":
            mx.zeros((n_experts, 2 * intermediate), dtype=mx.float16),
        "layers.0.mlp.experts.gate_up_proj.tq_bits":
            mx.array([bits], dtype=mx.uint8),
    }
    leftover = hydrate_jangtq(model, weights)

    assert isinstance(model.layers[0].mlp.experts.gate_up_proj, TurboQuantSwitchLinear)
    assert model.layers[0].mlp.experts.gate_up_proj.num_experts == n_experts
    assert leftover == {}


def test_hydrate_skips_unknown_module_paths():
    """TQ keys that point at paths the model doesn't have are silently
    skipped — mirrors the converter writing MTP weights that the
    inference model doesn't instantiate."""
    model = _ToyModel(n_layers=1)
    bits = 2
    weights = {
        # Path doesn't exist on _ToyModel.
        "mtp.0.experts.99.tq_packed": _make_tq_packed(64, 64, bits),
        "mtp.0.experts.99.tq_norms": _make_tq_norms(64),
        "mtp.0.experts.99.tq_bits": mx.array([bits], dtype=mx.uint8),
        # Real path that DOES match — should still swap.
        "layers.0.self_attn.q_proj.tq_packed": _make_tq_packed(64, 64, bits),
        "layers.0.self_attn.q_proj.tq_norms": _make_tq_norms(64),
        "layers.0.self_attn.q_proj.tq_bits": mx.array([bits], dtype=mx.uint8),
    }
    leftover = hydrate_jangtq(model, weights)
    assert isinstance(model.layers[0].self_attn.q_proj, TurboQuantLinear)
    assert leftover == {}


def test_hydrate_handles_incomplete_triplets():
    """If a triplet is missing one of packed/norms/bits the helper
    skips it rather than crashing."""
    model = _ToyModel(n_layers=1)
    bits = 2
    weights = {
        # Only packed — missing norms + bits.
        "layers.0.self_attn.q_proj.tq_packed": _make_tq_packed(64, 64, bits),
    }
    leftover = hydrate_jangtq(model, weights)
    # Module stays nn.Linear (no swap performed).
    assert isinstance(model.layers[0].self_attn.q_proj, nn.Linear)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
