import mlx.core as mx
import mlx.nn as nn

from jang_tools.loader import _fix_quantized_bits, _strip_runtime_ignored_weights


def test_fix_quantized_bits_promotes_uint8_scales_to_mxfp4_and_drops_affine_bias_placeholder():
    module = nn.QuantizedLinear(64, 4, bias=False, group_size=32, bits=4)
    module.weight = mx.zeros((4, 8), dtype=mx.uint32)
    module.scales = mx.zeros((4, 2), dtype=mx.uint8)
    assert module.mode == "affine"
    assert hasattr(module, "biases")

    class Wrapper(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

    wrapper = Wrapper(module)

    _fix_quantized_bits(wrapper, {})

    assert wrapper.child.mode == "mxfp4"
    assert wrapper.child.bits == 4
    assert wrapper.child.group_size == 32
    assert not hasattr(wrapper.child, "biases") or wrapper.child.biases is None


def test_fix_quantized_bits_uses_embedding_logical_dims_for_ambiguous_affine_shape():
    """Qwen3.6 JANG_4M embeds are 4-bit/gs64 despite an 8-bit/gs32 alias."""
    module = nn.QuantizedEmbedding(248320, 5120, group_size=32, bits=8)
    module.weight = mx.zeros((248320, 640), dtype=mx.uint32)
    module.scales = mx.zeros((248320, 80), dtype=mx.float16)
    module.biases = mx.zeros((248320, 80), dtype=mx.float16)

    class Wrapper(nn.Module):
        def __init__(self, child):
            super().__init__()
            self.child = child

    wrapper = Wrapper(module)

    _fix_quantized_bits(wrapper, {})

    assert wrapper.child.bits == 4
    assert wrapper.child.group_size == 64


def test_strip_runtime_ignored_weights_drops_mtp_and_importance_only():
    weights = {
        "language_model.model.layers.0.self_attn.q_proj.weight": object(),
        "language_model.model.layers.0.self_attn.q_proj.importance": object(),
        "mtp.fc.weight": object(),
        "mtp.layers.0.input_layernorm.weight": object(),
    }

    filtered = _strip_runtime_ignored_weights(weights)

    assert list(filtered) == ["language_model.model.layers.0.self_attn.q_proj.weight"]
