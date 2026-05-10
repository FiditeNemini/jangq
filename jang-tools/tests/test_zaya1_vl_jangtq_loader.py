import json
from types import SimpleNamespace

import mlx.core as mx


def test_zaya1_vl_jangtq_vlm_sanitize_maps_regular_text_weights():
    """ZAYA1-VL JANGTQ regular weights must land on the local VLM adapter.

    The TurboQuant groups are replaced by dotted-path lookup before regular
    weights load, but attention/router/norm/LoRA tensors still pass through
    ``_vlm_minimal_sanitize`` and ``model.load_weights(strict=False)``.
    Raw bundle keys use ``model.layers.*``; the local adapter exposes those
    parameters under ``language_model.model.layers.*``.  Without this rewrite,
    MLX silently ignores the regular weights and the model decodes from mostly
    initialized attention/router state.
    """

    from jang_tools.load_jangtq import _vlm_minimal_sanitize

    fake_model = SimpleNamespace(
        model_type="zaya1_vl",
        config=SimpleNamespace(
            model_type="zaya1_vl",
            text_config=SimpleNamespace(tie_word_embeddings=True),
        ),
    )
    raw_conv = mx.array([[[1.0, 2.0, 3.0]]])

    fixed = _vlm_minimal_sanitize(
        fake_model,
        {
            "model.layers.0.attn.self_attn.qkv.linear_q.weight": mx.zeros((4, 4)),
            "model.layers.0.attn.self_attn.qkv.conv_qk.0.weight": raw_conv,
            "model.layers.0.zaya_block.router.down_proj.weight": mx.zeros((2, 4)),
            "vision_tower.blocks.0.attn.qkv.weight": mx.zeros((4, 4)),
            "lm_head.weight": mx.zeros((4, 4)),
        },
    )

    assert (
        "language_model.model.layers.0.attn.self_attn.qkv.linear_q.weight"
        in fixed
    )
    conv_key = "language_model.model.layers.0.attn.self_attn.qkv.conv_qk.0.weight"
    assert conv_key in fixed
    assert tuple(fixed[conv_key].shape) == (1, 3, 1)
    assert (
        "language_model.model.layers.0.mlp.zaya_block.router.down_proj.weight"
        in fixed
    )
    assert "vision_tower.blocks.0.attn.qkv.weight" in fixed
    assert "lm_head.weight" not in fixed
    assert "language_model.lm_head.weight" not in fixed
    assert not any(key.startswith("model.layers.") for key in fixed)


def test_zaya1_vl_jangtq_vlm_sanitize_transposes_vision_patch_embed_conv3d():
    """ZAYA1-VL JANGTQ vision patch weights must use MLX Conv3d layout.

    The Qwen2.5-VL image processor returns flattened patches.  mlx-vlm's
    PatchEmbed reshapes those to `(N, T, H, W, C)` before calling MLX Conv3d,
    so the conv weight must be `(out, T, H, W, C)`.  The bundle stores the
    PyTorch layout `(out, C, T, H, W)`.
    """

    from jang_tools.load_jangtq import _vlm_minimal_sanitize

    fake_model = SimpleNamespace(
        model_type="zaya1_vl",
        config=SimpleNamespace(
            model_type="zaya1_vl",
            text_config=SimpleNamespace(tie_word_embeddings=True),
        ),
    )
    raw_patch = mx.zeros((1280, 3, 1, 14, 14))

    fixed = _vlm_minimal_sanitize(
        fake_model,
        {"vision_tower.patch_embed.proj.weight": raw_patch},
    )

    assert fixed["vision_tower.patch_embed.proj.weight"].shape == (1280, 1, 14, 14, 3)


def test_jangtq_vlm_affine_quant_mode_normalizes_container_mode():
    from jang_tools.load_jangtq_vlm import (
        _affine_quantize_mode,
        _vlm_quant_weight_key_candidates,
    )

    assert _affine_quantize_mode({"mode": "affine+mxtq"}) == "affine"
    assert _affine_quantize_mode({"mode": "affine"}) == "affine"
    assert _affine_quantize_mode({}) == "affine"
    assert (
        "model.layers.0.attn.self_attn.qkv.linear_q.scales"
        in _vlm_quant_weight_key_candidates(
            "language_model.model.layers.0.attn.self_attn.qkv.linear_q",
            "zaya1_vl",
        )
    )
    assert (
        "model.embed_tokens.scales"
        in _vlm_quant_weight_key_candidates(
            "language_model.model.embed_tokens",
            "zaya1_vl",
        )
    )


def test_jangtq_vlm_builds_zaya1_vl_inference_processor():
    from pathlib import Path

    from transformers.image_processing_utils import BaseImageProcessor

    from jang_tools.load_jangtq_vlm import _build_zaya1_vl_processor

    model_dir = Path("/Users/eric/models/JANGQ/ZAYA1-VL-8B-JANGTQ2")
    if not (model_dir / "chat_template.json").exists():
        import pytest

        pytest.skip("local ZAYA1-VL JANGTQ2 bundle is not present")

    processor = _build_zaya1_vl_processor(model_dir)

    text_prompt = processor.apply_chat_template(
        [{"role": "user", "content": "What is 17 + 28?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    image_prompt = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    assert processor.chat_template == json.loads(
        (model_dir / "chat_template.json").read_text()
    )["chat_template"]
    assert text_prompt == (
        "<|im_start|>user\n"
        "What is 17 + 28?<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert image_prompt == (
        "<|vision_start|><image><|vision_end|>\n"
        "<|im_start|>user\n"
        "Describe this image.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert hasattr(processor, "tokenizer")
    assert hasattr(processor, "image_processor")
    assert hasattr(processor.tokenizer, "stopping_criteria")
    assert hasattr(processor, "detokenizer")
    assert not isinstance(processor.image_processor, BaseImageProcessor)


def test_jangtq_hydrate_uses_logical_input_dims_for_quantized_switch_layers():
    """Pre-quantized MLX switch placeholders expose packed storage shapes.

    ZAYA1-VL JANGTQ starts from a placeholder QuantizedSwitchLinear. Its
    ``weight.shape[-1]`` is packed affine storage, not the logical input width
    used by TurboQuant codebook/sign caches. Hydration must prefer the module
    contract field (`input_dims`) before falling back to raw weight shape.
    """

    from jang_tools.load_jangtq import _infer_tq_input_features

    existing = SimpleNamespace(
        input_dims=2048,
        bits=8,
        weight=mx.zeros((16, 2048, 512), dtype=mx.uint32),
    )
    assert _infer_tq_input_features(existing) == 2048

    packed_only = SimpleNamespace(
        bits=8,
        weight=mx.zeros((16, 2048, 512), dtype=mx.uint32),
    )
    assert _infer_tq_input_features(packed_only) == 2048
