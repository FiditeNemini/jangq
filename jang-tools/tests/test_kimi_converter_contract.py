"""Kimi K2.6 JANGTQ converter contract tests."""

from pathlib import Path

from jang_tools.kimi_prune import convert_kimi_jangtq as mod


def test_kimi_jangtq_k_profile_uses_mixed_routed_projection_bits():
    assert mod.normalize_profile("JANGTQ_K") == "K"
    assert mod.normalize_profile("2K") == "K"

    gate = "language_model.model.layers.3.mlp.experts.0.gate_proj.weight"
    up = "language_model.model.layers.3.mlp.experts.0.up_proj.weight"
    down = "language_model.model.layers.3.mlp.experts.0.down_proj.weight"
    shared_down = "language_model.model.layers.3.mlp.shared_experts.down_proj.weight"
    router = "language_model.model.layers.3.mlp.gate.weight"

    assert mod.get_bits_and_method(gate, "JANGTQ_K") == (2, "mxtq")
    assert mod.get_bits_and_method(up, "K") == (2, "mxtq")
    assert mod.get_bits_and_method(down, "2K") == (4, "mxtq")
    assert mod.get_bits_and_method(shared_down, "K") == (8, "affine")
    assert mod.get_bits_and_method(router, "K") == (16, "passthrough")


def test_kimi_jangtq_k_metadata_records_per_projection_routed_bits():
    cfg = mod.build_jang_config(
        src=Path("/models/Kimi-K2.6"),
        profile="JANGTQ_K",
        n_experts=384,
        first_dense=1,
        n_layers=61,
    )

    assert cfg["profile"] == "JANGTQ_K"
    assert cfg["mxtq_bits"]["routed_expert"] == {
        "gate_proj": 2,
        "up_proj": 2,
        "down_proj": 4,
    }
    assert cfg["mxtq_bits"]["attention"] == 8
    assert cfg["mxtq_bits"]["shared_expert"] == 8


def test_kimi_aux_copy_list_includes_vl_processor_contract_files():
    names = mod.auxiliary_files_to_copy()

    assert "preprocessor_config.json" in names
    assert "kimi_k25_processor.py" in names
    assert "kimi_k25_vision_processing.py" in names
    assert "media_utils.py" in names
