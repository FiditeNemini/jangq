"""Verify the Laguna JANGTQ runtime's combined-`gate_up_proj` split.

The converter `convert_laguna_jangtq.py` writes routed-expert weights as
TWO TQ keys per layer:

    layers.L.mlp.experts.gate_up_proj.{tq_packed,tq_norms,tq_bits}
    layers.L.mlp.experts.down_proj.{tq_packed,tq_norms,tq_bits}

`gate_up_proj` is gate + up CONCATENATED along axis=-2 (rows). The model
itself uses `LagunaMoE.switch_mlp = SwitchGLU(...)` which has THREE
sub-projections under `switch_mlp.{gate_proj, up_proj, down_proj}`. The
runtime's JANGTQ branch must therefore split `gate_up_proj` at the row
midpoint and rename the keys before the generic hydrator runs.

This test exercises ONLY the split + rename logic (CPU-side dict
manipulation), not the full Laguna model build (which needs an actual
config + tokenizer + Metal kernel run). The full E2E forward pass is
gated on a real bundle being available locally.
"""
from __future__ import annotations

import re

import mlx.core as mx
import pytest


# Re-implement the laguna runtime's TQ split in-place here so the test
# is independent of import side effects (the runtime imports SwitchGLU,
# pulls in mlx_lm, etc., which is heavyweight for a CPU dict test).
_GATE_UP_PAT = re.compile(
    r"^(layers\.\d+\.mlp)\.experts\.gate_up_proj\.(tq_packed|tq_norms|tq_bits)$"
)
_DOWN_PAT = re.compile(
    r"^(layers\.\d+\.mlp)\.experts\.down_proj\.(tq_packed|tq_norms|tq_bits)$"
)


def _laguna_tq_split(weights: dict) -> dict:
    """Mirror of jang_tools.laguna.runtime.load's JANGTQ branch."""
    out: dict = {}
    for k, v in weights.items():
        m_gu = _GATE_UP_PAT.match(k)
        if m_gu:
            prefix, suffix = m_gu.group(1), m_gu.group(2)
            if suffix == "tq_packed":
                mid = v.shape[-2] // 2
                out[f"{prefix}.switch_mlp.gate_proj.tq_packed"] = v[..., :mid, :]
                out[f"{prefix}.switch_mlp.up_proj.tq_packed"] = v[..., mid:, :]
            elif suffix == "tq_norms":
                mid = v.shape[-1] // 2
                out[f"{prefix}.switch_mlp.gate_proj.tq_norms"] = v[..., :mid]
                out[f"{prefix}.switch_mlp.up_proj.tq_norms"] = v[..., mid:]
            else:
                out[f"{prefix}.switch_mlp.gate_proj.tq_bits"] = v
                out[f"{prefix}.switch_mlp.up_proj.tq_bits"] = v
            continue
        m_dn = _DOWN_PAT.match(k)
        if m_dn:
            prefix, suffix = m_dn.group(1), m_dn.group(2)
            out[f"{prefix}.switch_mlp.down_proj.{suffix}"] = v
            continue
        out[k] = v
    return out


def test_gate_up_split_axis_minus_2():
    """`gate_up_proj` packed (n_exp, 2*inter, packed_cols) splits into
    gate (rows 0..inter) and up (rows inter..2*inter)."""
    n_exp, inter, packed_cols, bits = 4, 32, 16, 2
    gate = mx.arange(n_exp * inter * packed_cols, dtype=mx.uint32).reshape(n_exp, inter, packed_cols)
    up = mx.arange(n_exp * inter * packed_cols, dtype=mx.uint32).reshape(n_exp, inter, packed_cols) + 9999
    combined_packed = mx.concatenate([gate, up], axis=-2)  # (n_exp, 2*inter, packed_cols)
    gate_norms = mx.zeros((n_exp, inter), dtype=mx.float16)
    up_norms = mx.ones((n_exp, inter), dtype=mx.float16)
    combined_norms = mx.concatenate([gate_norms, up_norms], axis=-1)

    weights = {
        "layers.0.mlp.experts.gate_up_proj.tq_packed": combined_packed,
        "layers.0.mlp.experts.gate_up_proj.tq_norms": combined_norms,
        "layers.0.mlp.experts.gate_up_proj.tq_bits": mx.array([bits], dtype=mx.uint8),
        "layers.0.mlp.experts.down_proj.tq_packed": mx.zeros((n_exp, 64, packed_cols), dtype=mx.uint32),
        "layers.0.mlp.experts.down_proj.tq_norms": mx.zeros((n_exp, 64), dtype=mx.float16),
        "layers.0.mlp.experts.down_proj.tq_bits": mx.array([bits], dtype=mx.uint8),
    }

    out = _laguna_tq_split(weights)

    # The split keys exist with the right naming.
    assert "layers.0.mlp.switch_mlp.gate_proj.tq_packed" in out
    assert "layers.0.mlp.switch_mlp.up_proj.tq_packed" in out
    assert "layers.0.mlp.switch_mlp.down_proj.tq_packed" in out
    # Old combined keys are gone.
    assert "layers.0.mlp.experts.gate_up_proj.tq_packed" not in out
    assert "layers.0.mlp.experts.down_proj.tq_packed" not in out

    # gate half byte-equal to original gate; up half byte-equal to original up.
    out_gate_packed = out["layers.0.mlp.switch_mlp.gate_proj.tq_packed"]
    out_up_packed = out["layers.0.mlp.switch_mlp.up_proj.tq_packed"]
    assert out_gate_packed.shape == (n_exp, inter, packed_cols)
    assert out_up_packed.shape == (n_exp, inter, packed_cols)
    assert mx.array_equal(out_gate_packed, gate)
    assert mx.array_equal(out_up_packed, up)

    # Norms split symmetrically.
    out_gate_norms = out["layers.0.mlp.switch_mlp.gate_proj.tq_norms"]
    out_up_norms = out["layers.0.mlp.switch_mlp.up_proj.tq_norms"]
    assert out_gate_norms.shape == (n_exp, inter)
    assert out_up_norms.shape == (n_exp, inter)
    assert mx.array_equal(out_gate_norms, gate_norms)
    assert mx.array_equal(out_up_norms, up_norms)

    # Both halves carry the same tq_bits scalar.
    assert int(out["layers.0.mlp.switch_mlp.gate_proj.tq_bits"][0].item()) == bits
    assert int(out["layers.0.mlp.switch_mlp.up_proj.tq_bits"][0].item()) == bits


def test_down_proj_pure_rename():
    """`down_proj` is already 3D-stacked; the runtime only renames its keys."""
    n_exp, hidden, packed_cols, bits = 4, 64, 16, 2
    down_packed = mx.arange(n_exp * hidden * packed_cols, dtype=mx.uint32).reshape(
        n_exp, hidden, packed_cols
    )
    weights = {
        "layers.3.mlp.experts.down_proj.tq_packed": down_packed,
        "layers.3.mlp.experts.down_proj.tq_norms": mx.zeros((n_exp, hidden), dtype=mx.float16),
        "layers.3.mlp.experts.down_proj.tq_bits": mx.array([bits], dtype=mx.uint8),
    }
    out = _laguna_tq_split(weights)
    assert "layers.3.mlp.switch_mlp.down_proj.tq_packed" in out
    assert "layers.3.mlp.experts.down_proj.tq_packed" not in out
    assert mx.array_equal(out["layers.3.mlp.switch_mlp.down_proj.tq_packed"], down_packed)


def test_passthrough_keys_unmodified():
    """Non-TQ-experts keys (norms, embed, attention, etc.) flow through
    untouched — only `experts.{gate_up_proj,down_proj}` patterns trigger
    the rename."""
    weights = {
        "layers.0.self_attn.q_proj.tq_packed": mx.zeros((64, 16), dtype=mx.uint32),
        "layers.0.self_attn.q_proj.tq_norms": mx.zeros((64,), dtype=mx.float16),
        "layers.0.self_attn.q_proj.tq_bits": mx.array([2], dtype=mx.uint8),
        "layers.0.input_layernorm.weight": mx.zeros((64,), dtype=mx.float16),
        "embed_tokens.weight": mx.zeros((100, 64), dtype=mx.uint32),
        "embed_tokens.scales": mx.zeros((100, 1), dtype=mx.float16),
    }
    out = _laguna_tq_split(weights)
    # All keys preserved verbatim — no `experts.` keyword in any of these.
    for k in weights:
        assert k in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
