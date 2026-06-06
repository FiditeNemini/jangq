import json


def test_mimo_v2_generated_config_keeps_runtime_modalities_text_only(tmp_path):
    from jang_tools.mimo_v2.convert_jang import QuantProfile, _write_config_json

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mimo_v2",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.334,
                "sliding_window": 128,
                "vision_config": {"model_type": "mimo_v2_vision"},
                "audio_config": {"model_type": "mimo_v2_audio"},
            }
        )
    )

    _write_config_json(src, dst, QuantProfile.parse("2"), 64, {}, include_mtp=True)

    cfg = json.loads((dst / "config.json").read_text())
    assert cfg["capabilities"]["modalities"] == ["text"]
    assert cfg["capabilities"]["preserved_modalities"] == ["vision", "audio"]
    assert cfg["capabilities"]["unwired_modalities"] == ["vision", "audio"]
    assert cfg["capabilities"]["multimodal_status"] == "weights_preserved_text_runtime"
    assert cfg["runtime"]["multimodal_mode"] == "weights_preserved_text_runtime"
    assert cfg["vision_config"]["model_type"] == "mimo_v2_vision"
    assert cfg["audio_config"]["model_type"] == "mimo_v2_audio"
