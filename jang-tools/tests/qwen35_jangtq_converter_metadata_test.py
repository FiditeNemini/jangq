from pathlib import Path


CONVERTER = Path(__file__).resolve().parents[1] / "jang_tools" / "convert_qwen35_jangtq.py"


def _converter_source() -> str:
    return CONVERTER.read_text(encoding="utf-8")


def test_jangtq_config_default_quantization_describes_affine_skeleton():
    source = _converter_source()

    assert '"bits": affine_default_bits' in source
    assert "affine_default_bits = 8" in source
    assert '"mode": "affine"' in source
    assert 'config["mxtq_bits"] = routed_bits_meta' in source
    assert '"routed_expert_bits": routed_bits_meta' in source
    assert 'config["quantization"] = {"group_size": 64, "bits": default_expert_bits}' not in source
    assert '"bits_default": default_expert_bits' not in source


def test_jangtq_records_affine_overrides_but_keeps_mxtq_separate():
    source = _converter_source()

    assert "affine_quant_overrides = {}" in source
    assert "def record_affine_quant(out_name, bits):" in source
    assert "record_affine_quant(out_name, bits)" in source
    assert '"mxtq_tensor_group_count": total_mxtq' in source
    assert 'config["quantization"] = quantization' in source


def test_jangtq_converter_carries_qwen35_eos_fix_into_config_and_generation_config():
    source = _converter_source()

    assert '"qwen3_5_moe": {248044: 248046}' in source
    assert '"qwen3_5_vl": {248044: 248046}' in source
    assert 'text_cfg["eos_token_id"] = new_eos' in source
    assert 'config["eos_token_id"] = text_cfg["eos_token_id"]' in source
    assert 'gen_cfg_src = SRC / "generation_config.json"' in source
    assert "eos_fixed_generation_config = True" in source
    assert 'if f == "generation_config.json" and eos_fixed_generation_config:' in source
