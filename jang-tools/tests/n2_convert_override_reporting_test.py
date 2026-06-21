from jang_tools.convert import _is_n2_expert_down


def test_n2_expert_down_override_matches_source_tensor_name():
    assert _is_n2_expert_down("model.language_model.layers.0.mlp.experts.down_proj")
    assert not _is_n2_expert_down("model.language_model.layers.0.mlp.experts.gate_up_proj")


def test_n2_down_override_recompute_marker_is_present():
    source = "jang-tools/jang_tools/convert.py"
    text = open(source, encoding="utf-8").read()
    assert "alloc_summary = summarize_allocation_compact(_tensor_bits, tensor_info, num_experts)" in text
    assert "actual_bits = alloc_summary[\"average_bits\"]" in text
