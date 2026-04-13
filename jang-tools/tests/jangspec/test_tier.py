"""Unit tests for jang_tools.jangspec.tier."""

import pytest

from jang_tools.jangspec.tier import TierSplit, classify_tensors, is_dense_model


def test_splits_moe_tensors():
    names = [
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
        "model.norm.weight",
        "lm_head.weight",
        "lm_head.scales",
        "lm_head.biases",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj.scales",
        "layers.0.self_attn.q_proj.biases",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate.weight",   # router
        "layers.0.mlp.gate.scales",
        "layers.0.mlp.gate.biases",
        "layers.0.switch_mlp.gate_proj.weight",
        "layers.0.switch_mlp.gate_proj.scales",
        "layers.0.switch_mlp.gate_proj.biases",
        "layers.0.switch_mlp.up_proj.weight",
        "layers.0.switch_mlp.up_proj.scales",
        "layers.0.switch_mlp.up_proj.biases",
        "layers.0.switch_mlp.down_proj.weight",
        "layers.0.switch_mlp.down_proj.scales",
        "layers.0.switch_mlp.down_proj.biases",
        "layers.0.shared_expert.gate_proj.weight",
        "layers.0.shared_expert.gate_proj.scales",
        "layers.0.shared_expert.gate_proj.biases",
    ]

    split = classify_tensors(names)
    assert isinstance(split, TierSplit)

    # Hot core: embeddings, norms, lm_head, attention, router, shared expert
    assert "model.embed_tokens.weight" in split.hot_core
    assert "lm_head.weight" in split.hot_core
    assert "layers.0.self_attn.q_proj.weight" in split.hot_core
    assert "layers.0.mlp.gate.weight" in split.hot_core
    assert "layers.0.shared_expert.gate_proj.weight" in split.hot_core
    assert "model.norm.weight" in split.hot_core

    # Expert streamed: just the switch_mlp base names, once each
    assert split.expert_base_names == [
        "layers.0.switch_mlp.down_proj",
        "layers.0.switch_mlp.gate_proj",
        "layers.0.switch_mlp.up_proj",
    ]


def test_is_dense_model_true_when_no_switch_mlp():
    names = [
        "model.embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ]
    assert is_dense_model(names) is True


def test_is_dense_model_false_when_switch_mlp_present():
    names = [
        "model.embed_tokens.weight",
        "layers.0.switch_mlp.gate_proj.weight",
    ]
    assert is_dense_model(names) is False
