import json

import torch
from safetensors.torch import save_file


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


def test_mimo_v2_verify_bundle_accepts_prestacked_jangtq_layout(tmp_path):
    from jang_tools.mimo_v2.verify_bundle import verify_bundle

    bundle = tmp_path / "MiMo-V2.5-JANGTQ_2"
    bundle.mkdir()
    cfg = {
        "model_type": "mimo_v2",
        "jang_profile": "JANGTQ_2",
        "mxtq_bits": {"routed_expert": {"gate_proj": 2, "up_proj": 2, "down_proj": 2}},
        "routed_expert_bits": {"gate_proj": 2, "up_proj": 2, "down_proj": 2},
        "rope_parameters": {},
        "quantization": {
            "bits": 8,
            "group_size": 64,
            "quant_method": "affine",
            "mode": "affine",
            "routed_experts": "tq_prestacked_switch_mlp",
        },
        "capabilities": {
            "modalities": ["text"],
            "preserved_modalities": ["vision", "audio"],
            "unwired_modalities": ["vision", "audio"],
            "multimodal_status": "weights_preserved_text_runtime",
            "cache_type": "kv",
            "tools": {"parser": "xml_function"},
            "reasoning": {"parser": "think_xml"},
        },
        "runtime": {
            "cache_topology": {
                "family": "hybrid_full_swa_kv",
                "prefix_cache": True,
                "l2_disk_cache": True,
                "turboquant_kv": "full_attention_layers_only",
                "swa_layers": "rotating_kv_native",
            },
            "bundle_has_mtp": False,
            "mtp_mode": "absent",
            "multimodal_mode": "weights_preserved_text_runtime",
            "tq_layout": "prestacked_switch_mlp",
        },
    }
    (bundle / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    tokenizer_config = {"chat_template": ""}
    (bundle / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config),
        encoding="utf-8",
    )
    for aux in (
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "preprocessor_config.json",
        "configuration_mimo_v2.py",
        "modeling_mimo_v2.py",
        "chat_template.jinja",
    ):
        (bundle / aux).write_text("{}" if aux.endswith(".json") else "", encoding="utf-8")
    (bundle / "audio_tokenizer").mkdir()
    save_file({"dummy": torch.zeros((1,), dtype=torch.float32)}, bundle / "audio_tokenizer/model.safetensors")

    tensors = {
        "model.layers.0.self_attn.qkv_proj.weight": torch.zeros((1,), dtype=torch.int32),
        "model.layers.0.self_attn.qkv_proj.scales": torch.zeros((1,), dtype=torch.float16),
        "model.layers.0.self_attn.qkv_proj.biases": torch.zeros((1,), dtype=torch.float16),
        "model.embed_tokens.weight": torch.zeros((1,), dtype=torch.int32),
        "model.embed_tokens.scales": torch.zeros((1,), dtype=torch.float16),
        "model.embed_tokens.biases": torch.zeros((1,), dtype=torch.float16),
        "lm_head.weight": torch.zeros((1,), dtype=torch.int32),
        "lm_head.scales": torch.zeros((1,), dtype=torch.float16),
        "lm_head.biases": torch.zeros((1,), dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.weight": torch.zeros((1,), dtype=torch.int32),
        "model.layers.0.mlp.gate_proj.scales": torch.zeros((1,), dtype=torch.float16),
        "model.layers.0.mlp.gate_proj.biases": torch.zeros((1,), dtype=torch.float16),
        "model.layers.0.self_attn.o_proj.weight": torch.zeros((1,), dtype=torch.int32),
        "model.layers.0.self_attn.o_proj.scales": torch.zeros((1,), dtype=torch.float16),
        "model.layers.0.self_attn.o_proj.biases": torch.zeros((1,), dtype=torch.float16),
        "model.layers.0.input_layernorm.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "model.layers.1.input_layernorm.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "model.norm.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "model.layers.1.mlp.gate.weight": torch.zeros((1,), dtype=torch.float32),
        "model.layers.1.mlp.gate.e_score_correction_bias": torch.zeros((1,), dtype=torch.float32),
        "model.layers.1.self_attn.attention_sink_bias": torch.zeros((1,), dtype=torch.bfloat16),
        "visual.blocks.0.attn.qkv.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "audio_encoder.input_local_transformer.layers.0.input_layernorm.weight": torch.zeros((1,), dtype=torch.bfloat16),
        "speech_embeddings.0.weight": torch.zeros((1,), dtype=torch.bfloat16),
    }
    for layer in range(1, 48):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for suffix in ("tq_packed", "tq_norms", "tq_bits"):
                tensors[
                    f"model.layers.{layer}.mlp.switch_mlp.{proj}.{suffix}"
                ] = torch.zeros((1,), dtype=torch.int32)
    save_file(tensors, bundle / "model-00001-of-00001.safetensors")
    weight_map = {key: "model-00001-of-00001.safetensors" for key in tensors}
    (bundle / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "total_size": (bundle / "model-00001-of-00001.safetensors").stat().st_size
                },
                "weight_map": weight_map,
            }
        ),
        encoding="utf-8",
    )

    assert verify_bundle(bundle) == 0
