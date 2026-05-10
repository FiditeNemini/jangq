from jang_tools.convert import _get_mlx_compatible_group_size


def test_mlx_group_size_shrinks_for_dsv4_tiny_projection():
    assert _get_mlx_compatible_group_size(32, 128) == 32


def test_mlx_group_size_prefers_largest_supported_divisor_not_exceeding_request():
    assert _get_mlx_compatible_group_size(192, 128) == 64
    assert _get_mlx_compatible_group_size(128, 64) == 64


def test_mlx_group_size_returns_none_when_unquantizable_by_mlx():
    assert _get_mlx_compatible_group_size(24, 128) is None


def test_jang_loader_quant_metadata_prefers_block_size_then_group_size():
    from jang_tools.loader import _jang_default_bits, _jang_quant_block_size

    assert _jang_quant_block_size({"quantization": {"group_size": 32}}) == 32
    assert (
        _jang_quant_block_size(
            {"quantization": {"block_size": 64, "group_size": 32}}
        )
        == 64
    )
    assert _jang_default_bits({"quantization": {"bits": 4, "bit_widths_used": [2]}}) == 4
    assert _jang_default_bits({"quantization": {"bit_widths_used": [2, 4]}}) == 2


def test_jang_loader_skips_modules_not_divisible_by_group_size():
    import numpy as np
    from types import SimpleNamespace

    from jang_tools.loader import (
        _module_can_quantize_with_group_size,
        _module_path_is_raw_adapter,
    )

    assert _module_can_quantize_with_group_size(
        SimpleNamespace(weight=np.zeros((1024, 2048), dtype=np.float32)),
        32,
    )
    assert not _module_can_quantize_with_group_size(
        SimpleNamespace(weight=np.zeros((1024, 8), dtype=np.float32)),
        32,
    )
    assert not _module_can_quantize_with_group_size(
        SimpleNamespace(input_dims=8),
        32,
    )
    assert _module_path_is_raw_adapter(
        "model.layers.0.attn.self_attn.qkv.lora_linear_q.0"
    )
    assert _module_path_is_raw_adapter(
        "model.layers.0.mlp.zaya_block.experts.local_experts.0.lora_fc1.1"
    )
    assert not _module_path_is_raw_adapter(
        "model.layers.0.attn.self_attn.qkv.linear_q"
    )
