"""muP fold contract for Falcon-H1 conversion (mlxstudio#129).

The fold table is HF-exact (modeling_falcon_h1.py), deliberately NOT
mlx-lm's sanitize, which has a dead key_proj branch and never folds
v_proj.
"""

import numpy as np
import pytest

from jang_tools.mup_fold import MupFold


FALCON_CFG = {
    "model_type": "falcon_h1",
    "hidden_size": 4,
    "attention_in_multiplier": 2.0,
    "attention_out_multiplier": 3.0,
    "key_multiplier": 5.0,
    "embedding_multiplier": 7.0,
    "lm_head_multiplier": 11.0,
    "ssm_in_multiplier": 13.0,
    "ssm_out_multiplier": 17.0,
    "mlp_multipliers": [19.0, 23.0],
    "ssm_multipliers": [1.5, 2.5, 3.5, 4.5, 5.5],
    "mamba_d_ssm": 2,
    "mamba_n_groups": 1,
    "mamba_d_state": 2,
    "mamba_n_heads": 3,
}


def _fold_factor(fold: MupFold, name: str) -> float:
    w = np.ones((1, 1), dtype=np.float32)
    return float(fold.apply(name, w)[0, 0])


def test_non_falcon_configs_get_no_fold():
    assert MupFold.from_config({"model_type": "qwen3_5"}) is None
    assert MupFold.from_config({}) is None


def test_scalar_fold_table_matches_hf_reference():
    fold = MupFold.from_config(FALCON_CFG)
    assert fold is not None

    prefix = "model.layers.0"
    assert _fold_factor(fold, "model.embed_tokens.weight") == 7.0
    assert _fold_factor(fold, "lm_head.weight") == 11.0
    assert _fold_factor(fold, f"{prefix}.self_attn.q_proj.weight") == 2.0
    # v_proj IS folded by attention_in_multiplier (mlx-lm misses this).
    assert _fold_factor(fold, f"{prefix}.self_attn.v_proj.weight") == 2.0
    # k_proj folds attention_in * key (mlx-lm's key branch is dead code).
    assert _fold_factor(fold, f"{prefix}.self_attn.k_proj.weight") == 2.0 * 5.0
    assert _fold_factor(fold, f"{prefix}.self_attn.o_proj.weight") == 3.0
    assert _fold_factor(fold, f"{prefix}.mamba.out_proj.weight") == 17.0
    assert _fold_factor(fold, f"{prefix}.feed_forward.gate_proj.weight") == 19.0
    assert _fold_factor(fold, f"{prefix}.feed_forward.down_proj.weight") == 23.0


def test_bias_rules_carry_only_output_side_components():
    fold = MupFold.from_config(FALCON_CFG)
    prefix = "model.layers.0"

    def bias_factor(name: str) -> float:
        b = np.ones(1, dtype=np.float32)
        return float(fold.apply(name, b)[0])

    # Input-side scalars never touch biases.
    q_bias = np.ones(1, dtype=np.float32)
    assert fold.apply(f"{prefix}.self_attn.q_proj.bias", q_bias) is q_bias
    v_bias = np.ones(1, dtype=np.float32)
    assert fold.apply(f"{prefix}.self_attn.v_proj.bias", v_bias) is v_bias
    # k_proj bias carries only key_multiplier (output-side).
    assert bias_factor(f"{prefix}.self_attn.k_proj.bias") == 5.0
    # Output-side scalars fold biases fully.
    assert bias_factor(f"{prefix}.self_attn.o_proj.bias") == 3.0
    assert bias_factor(f"{prefix}.mamba.out_proj.bias") == 17.0
    assert bias_factor(f"{prefix}.feed_forward.down_proj.bias") == 23.0


def test_in_proj_folds_ssm_in_times_mup_vector_per_row():
    fold = MupFold.from_config(FALCON_CFG)
    # Sections: [d_ssm=2, d_ssm=2, G*S=2, G*S=2, heads=3] → 11 rows.
    expected_vector = np.concatenate(
        [
            np.full(2, 1.5),
            np.full(2, 2.5),
            np.full(2, 3.5),
            np.full(2, 4.5),
            np.full(3, 5.5),
        ]
    )
    weights = np.ones((11, 4), dtype=np.float32)
    out = fold.apply("model.layers.0.mamba.in_proj.weight", weights)
    np.testing.assert_allclose(
        out, np.broadcast_to(13.0 * expected_vector[:, None], (11, 4)), rtol=1e-6
    )

    bias = np.ones(11, dtype=np.float32)
    out_bias = fold.apply("model.layers.0.mamba.in_proj.bias", bias)
    np.testing.assert_allclose(out_bias, expected_vector, rtol=1e-6)


def test_in_proj_shape_mismatch_fails_loud():
    fold = MupFold.from_config(FALCON_CFG)
    with pytest.raises(ValueError, match="refusing to fold"):
        fold.apply("model.layers.0.mamba.in_proj.weight", np.ones((7, 4), dtype=np.float32))


def test_in_proj_without_ssm_multipliers_fails_loud():
    cfg = dict(FALCON_CFG)
    cfg.pop("ssm_multipliers")
    fold = MupFold.from_config(cfg)
    with pytest.raises(ValueError, match="ssm_multipliers"):
        fold.apply("model.layers.0.mamba.in_proj.weight", np.ones((11, 4), dtype=np.float32))


def test_unit_multipliers_are_skipped_and_not_recorded():
    cfg = {"model_type": "falcon_h1", "hidden_size": 4}
    fold = MupFold.from_config(cfg)
    w = np.ones((2, 2), dtype=np.float32)
    out = fold.apply("model.layers.0.self_attn.q_proj.weight", w)
    assert out is w
    assert fold.folded == {}


def test_d_ssm_falls_back_to_expand_times_hidden():
    cfg = dict(FALCON_CFG)
    cfg.pop("mamba_d_ssm")
    cfg["mamba_expand"] = 2
    fold = MupFold.from_config(cfg)
    # d_ssm = 2*4 = 8 → rows = 8+8+2+2+3 = 23
    weights = np.ones((23, 4), dtype=np.float32)
    out = fold.apply("model.layers.0.mamba.in_proj.weight", weights)
    assert out.shape == (23, 4)
    np.testing.assert_allclose(out[0], 13.0 * 1.5, rtol=1e-6)


def test_non_projection_tensors_untouched():
    fold = MupFold.from_config(FALCON_CFG)
    norm = np.ones(4, dtype=np.float32)
    assert fold.apply("model.layers.0.input_layernorm.weight", norm) is norm
    conv = np.ones((8, 4, 1), dtype=np.float32)
    assert fold.apply("model.layers.0.mamba.conv1d.weight", conv) is conv


def test_folded_ledger_records_labels():
    fold = MupFold.from_config(FALCON_CFG)
    fold.apply("model.layers.0.self_attn.k_proj.weight", np.ones((1, 1), dtype=np.float32))
    assert fold.folded == {
        "model.layers.0.self_attn.k_proj.weight": "attention_in_multiplier*key_multiplier"
    }
