"""Zero-centered RMSNorm shift contract (mlxstudio#130).

Text qwen3_5-family JANG bundles must store PRE-SHIFTED (+1.0) norms
because mlx_lm's sanitize gate (mtp keys / HF-layout conv1d) can never
fire on the JANG v2 text load path. Wrapped-VL bundles must keep RAW
norms (the VL load path shifts unconditionally).
"""

import numpy as np

from jang_tools.loader import _mlx_lm_sanitize_shifts_norms
from jang_tools.zero_centered_norms import (
    ZeroCenteredNormShift,
    mlx_lm_sanitize_would_shift,
)


def test_from_config_family_gating():
    assert ZeroCenteredNormShift.from_config({"model_type": "qwen3_5_moe"}, False) is not None
    assert ZeroCenteredNormShift.from_config({"model_type": "qwen3_5"}, False) is not None
    assert ZeroCenteredNormShift.from_config({"model_type": "qwen3_next"}, False) is not None
    assert ZeroCenteredNormShift.from_config({"model_type": "qwen3_5_moe_text"}, False) is not None
    assert ZeroCenteredNormShift.from_config({"model_type": "falcon_h1"}, False) is None
    assert ZeroCenteredNormShift.from_config({"model_type": "llama"}, False) is None
    assert ZeroCenteredNormShift.from_config({}, False) is None


def test_nested_text_config_model_type_is_detected():
    cfg = {"model_type": "qwen3_5_vl", "text_config": {"model_type": "qwen3_5_moe"}}
    assert ZeroCenteredNormShift.from_config(cfg, False) is not None


def test_vl_wrapped_checkpoints_are_excluded():
    # VL bundles keep raw norms: _load_jang_v2_vlm shifts unconditionally
    # at load, so shifting at conversion would double-apply.
    cfg = {"model_type": "qwen3_5_vl", "text_config": {"model_type": "qwen3_5_moe"}}
    assert ZeroCenteredNormShift.from_config(cfg, True) is None
    assert ZeroCenteredNormShift.from_config({"model_type": "qwen3_5_moe"}, True) is None


def test_all_five_mlx_lm_norm_suffixes_are_shifted():
    shift = ZeroCenteredNormShift.from_config({"model_type": "qwen3_5_moe"}, False)
    prefix = "model.layers.0"
    names = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
        "model.norm.weight",
        f"{prefix}.self_attn.q_norm.weight",
        f"{prefix}.self_attn.k_norm.weight",
    ]
    for name in names:
        w = np.zeros(4, dtype=np.float32)
        out = shift.apply(name, w)
        np.testing.assert_allclose(out, np.ones(4, dtype=np.float32))
    assert shift.shifted == names


def test_wrapped_language_model_prefix_still_matches():
    # After mlx_lm qwen3_5_moe sanitize, text keys are wrapped under
    # language_model.; suffix matching must still hit.
    shift = ZeroCenteredNormShift.from_config({"model_type": "qwen3_5_moe"}, False)
    w = np.zeros(4, dtype=np.float32)
    out = shift.apply("language_model.model.norm.weight", w)
    np.testing.assert_allclose(out, np.ones(4, dtype=np.float32))


def test_gated_linear_attn_norm_is_never_shifted():
    # linear_attn.norm is RMSNormGated — not in mlx_lm's norm_keys.
    shift = ZeroCenteredNormShift.from_config({"model_type": "qwen3_5_moe"}, False)
    w = np.zeros(4, dtype=np.float32)
    assert shift.apply("model.layers.0.linear_attn.norm.weight", w) is w
    assert shift.shifted == []


def test_non_norm_and_non_1d_tensors_untouched():
    shift = ZeroCenteredNormShift.from_config({"model_type": "qwen3_5_moe"}, False)
    w2d = np.zeros((2, 4), dtype=np.float32)
    assert shift.apply("model.norm.weight", w2d) is w2d
    proj = np.zeros(4, dtype=np.float32)
    assert shift.apply("model.layers.0.self_attn.q_proj.weight", proj) is proj
    bias = np.zeros(4, dtype=np.float32)
    assert shift.apply("model.layers.0.input_layernorm.bias", bias) is bias
    assert shift.shifted == []


class _Arr:
    """Shape-only stand-in for mx arrays in gate tests."""

    def __init__(self, shape):
        self.shape = shape


def test_loader_gate_mirrors_qwen3_5_conv1d_rule():
    # MLX-layout conv1d (last dim 1) → mlx_lm sanitize gate does NOT fire
    # → the JANG loader repair must apply.
    mlx_layout = {
        "language_model.model.layers.0.linear_attn.conv1d.weight": _Arr((8, 4, 1)),
        "language_model.model.norm.weight": _Arr((4,)),
    }
    assert not _mlx_lm_sanitize_shifts_norms("qwen3_5_moe", mlx_layout)

    # HF-layout conv1d (last dim = kernel) → mlx_lm shifts this shard
    # itself → the repair must skip it (no double shift).
    hf_layout = {
        "language_model.model.layers.0.linear_attn.conv1d.weight": _Arr((8, 1, 4)),
    }
    assert _mlx_lm_sanitize_shifts_norms("qwen3_5_moe", hf_layout)
    assert _mlx_lm_sanitize_shifts_norms("qwen3_5", hf_layout)


def test_conversion_source_gate_hf_raw_vs_mlx_format():
    # HF-raw qwen3_5 sources carry mtp.* weights and/or HF-layout conv1d
    # → mlx_lm's sanitize would shift → source norms are RAW → we shift.
    assert mlx_lm_sanitize_would_shift(
        "qwen3_5_moe", {"mtp.layers.0.norm.weight", "model.norm.weight"}, []
    )
    assert mlx_lm_sanitize_would_shift(
        "qwen3_5", {"model.layers.0.linear_attn.conv1d.weight"}, [4]
    )
    # mlx-format sources (e.g. the CRACK abliteration pipeline) had mtp
    # stripped + conv1d moved to MLX layout by mlx_lm's own conversion,
    # which already folded the +1.0 → we must NOT shift again.
    assert not mlx_lm_sanitize_would_shift(
        "qwen3_5", {"model.layers.0.linear_attn.conv1d.weight", "model.norm.weight"}, [1]
    )
    # qwen3_next: per-expert HF keys = raw source; stacked mlx-format = shifted.
    assert mlx_lm_sanitize_would_shift(
        "qwen3_next", {"model.layers.0.mlp.experts.0.up_proj.weight"}, []
    )
    assert not mlx_lm_sanitize_would_shift(
        "qwen3_next", {"model.layers.0.mlp.switch_mlp.up_proj.weight"}, [1]
    )


def test_from_config_records_model_type_for_source_gate():
    shift = ZeroCenteredNormShift.from_config({"model_type": "qwen3_next"}, False)
    assert shift.model_type == "qwen3_next"
    cfg = {"model_type": "qwen3_5_vl", "text_config": {"model_type": "qwen3_5_moe"}}
    assert ZeroCenteredNormShift.from_config(cfg, False).model_type == "qwen3_5_moe"


def test_loader_gate_mtp_leg_fires_on_wrapped_midkey_mtp():
    # Converter preserves mtp weights; wrapped "*.mtp.*" keys survive the
    # loader's startswith("mtp.") strip, and mlx_lm's substring gate
    # shifts those shards itself — the repair must skip them.
    shard = {
        "language_model.model.mtp.layers.0.input_layernorm.weight": _Arr((4,)),
        "language_model.model.norm.weight": _Arr((4,)),
    }
    assert _mlx_lm_sanitize_shifts_norms("qwen3_5_moe", shard)


def test_loader_gate_mirrors_qwen3_next_early_return():
    # qwen3_next sanitize early-returns unless per-expert HF keys exist —
    # even with HF-layout conv1d present it never shifts, so the repair
    # must apply.
    jang_shard = {
        "model.layers.0.linear_attn.conv1d.weight": _Arr((8, 1, 4)),
        "model.norm.weight": _Arr((4,)),
    }
    assert not _mlx_lm_sanitize_shifts_norms("qwen3_next", jang_shard)

    hf_raw_shard = dict(jang_shard)
    hf_raw_shard["model.layers.0.mlp.experts.0.up_proj.weight"] = _Arr((16, 4))
    assert _mlx_lm_sanitize_shifts_norms("qwen3_next", hf_raw_shard)
