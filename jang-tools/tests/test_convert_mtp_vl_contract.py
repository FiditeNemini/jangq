from jang_tools.convert import (
    _build_mtp_runtime_metadata,
    _configured_mtp_layers_from_config,
    _is_mtp_tensor_name,
    _is_vision_tensor_name,
    _prepare_vision_passthrough_tensor,
    _sanitize_output_tensor_name,
)


def test_qwen_nested_mtp_config_is_detected():
    cfg = {
        "model_type": "qwen3_5",
        "text_config": {
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        },
    }

    assert _configured_mtp_layers_from_config(cfg) == 1


def test_mtp_runtime_metadata_marks_preserved_enabled_when_weights_exist():
    cfg = {
        "text_config": {
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": False,
        }
    }
    tensor_names = {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.norm.weight",
    }

    meta = _build_mtp_runtime_metadata(cfg, tensor_names)

    assert meta["runtime"] == {
        "bundle_has_mtp": True,
        "mtp_layers": 1,
        "mtp_mode": "preserved_enabled",
    }
    assert meta["mtp"] == {
        "kept": True,
        "enabled": True,
        "num_layers": 1,
        "tensor_count": 3,
    }
    assert meta["bundle_has_mtp"] is True
    assert meta["mtp_layers"] == 1


def test_mtp_runtime_metadata_flags_missing_weights_when_config_claims_mtp():
    cfg = {"text_config": {"mtp_num_hidden_layers": 1}}

    meta = _build_mtp_runtime_metadata(cfg, {"model.layers.0.mlp.up_proj.weight"})

    assert meta["runtime"]["bundle_has_mtp"] is False
    assert meta["runtime"]["mtp_layers"] == 1
    assert meta["runtime"]["mtp_mode"] == "metadata_only_missing_weights"
    assert meta["mtp"]["kept"] is False
    assert meta["mtp"]["enabled"] is False


def test_qwen_visual_namespace_is_classified_as_vision_tensor():
    assert _is_vision_tensor_name("model.visual.patch_embed.proj.weight")
    assert _is_vision_tensor_name("model.visual.blocks.0.attn.qkv.weight")
    assert _is_vision_tensor_name("vision_tower.blocks.0.attn.qkv.weight")
    assert not _is_vision_tensor_name("model.language_model.layers.0.mlp.up_proj.weight")


def test_qwen_mtp_namespace_is_classified_as_mtp_tensor():
    assert _is_mtp_tensor_name("mtp.fc.weight")
    assert _is_mtp_tensor_name("language_model.mtp.layers.0.self_attn.q_proj.weight")
    assert not _is_mtp_tensor_name("model.visual.blocks.0.attn.qkv.weight")


def test_qwen_conditional_generation_names_are_sanitized_to_runtime_keys():
    assert (
        _sanitize_output_tensor_name("model.language_model.layers.0.mlp.up_proj.weight")
        == "language_model.model.layers.0.mlp.up_proj.weight"
    )
    assert (
        _sanitize_output_tensor_name("model.visual.patch_embed.proj.weight")
        == "vision_tower.patch_embed.proj.weight"
    )
    assert _sanitize_output_tensor_name("lm_head.weight") == "language_model.lm_head.weight"
    assert _sanitize_output_tensor_name("mtp.fc.weight") == "mtp.fc.weight"


def test_qwen_5d_patch_embed_transposes_to_mlx_layout():
    import numpy as np

    tensor = np.zeros((1152, 3, 2, 16, 16), dtype=np.float32)
    out = _prepare_vision_passthrough_tensor(
        "vision_tower.patch_embed.proj.weight",
        tensor,
    )

    assert out.shape == (1152, 2, 16, 16, 3)
    assert out.dtype == np.float16
