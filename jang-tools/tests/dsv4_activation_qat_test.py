import inspect
from types import SimpleNamespace

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _run_qat_contract_on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


def _np(value):
    return np.asarray(value)


def test_dsv4_e4m3_block_qat_matches_reference_value_lattice():
    from jang_tools.dsv4.mlx_model import act_quant_sim

    # A maximum of 448 makes the block's UE8M0 scale exactly 1.  These values
    # cover subnormal, normal, tie-to-even, saturation, and signed rounding.
    values = np.array(
        [
            0.0,
            0.001,
            0.001953125,
            0.0029296875,
            0.0146484375,
            0.96875,
            1.0625,
            1.1875,
            432.0,
            448.0,
            -0.0029296875,
            -1.1875,
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            0.0,
            0.001953125,
            0.001953125,
            0.00390625,
            0.015625,
            1.0,
            1.0,
            1.25,
            448.0,
            448.0,
            -0.00390625,
            -1.25,
        ],
        dtype=np.float32,
    )
    block = np.pad(values, (0, 64 - values.size), constant_values=448.0)

    actual = act_quant_sim(mx.array(block), block_size=64)
    mx.eval(actual)

    np.testing.assert_array_equal(_np(actual)[: values.size], expected)


def test_dsv4_fp4_block_qat_matches_e2m1_ties_and_scale():
    from jang_tools.dsv4.mlx_model import fp4_act_quant_sim

    # A maximum of 6 makes the power-of-two block scale exactly 1.
    values = np.array(
        [
            0.0,
            0.25,
            0.2501,
            0.75,
            1.25,
            1.2501,
            1.75,
            2.5,
            2.5001,
            3.5,
            5.0,
            5.0001,
            6.0,
            -0.75,
            -5.0,
        ],
        dtype=np.float32,
    )
    expected = np.array(
        [
            0.0,
            0.0,
            0.5,
            1.0,
            1.0,
            1.5,
            2.0,
            2.0,
            3.0,
            4.0,
            4.0,
            6.0,
            6.0,
            -1.0,
            -4.0,
        ],
        dtype=np.float32,
    )
    block = np.pad(values, (0, 32 - values.size), constant_values=6.0)

    actual = fp4_act_quant_sim(mx.array(block), block_size=32)
    mx.eval(actual)

    np.testing.assert_array_equal(_np(actual)[: values.size], expected)


def test_dsv4_indexer_hadamard_is_normalized_sylvester_transform():
    from jang_tools.dsv4.mlx_model import hadamard_rotate_activation

    actual = hadamard_rotate_activation(mx.array([[1.0, 2.0, 3.0, 4.0]]))
    mx.eval(actual)

    np.testing.assert_array_equal(_np(actual), [[5.0, -1.0, -2.0, 0.0]])


def test_dsv4_qat_is_wired_after_rope_and_before_cache_or_scoring():
    from jang_tools.dsv4.mlx_model import Compressor, DeepseekV4Attention, Indexer

    attention_source = inspect.getsource(DeepseekV4Attention.__call__)
    kv_rope = attention_source.index("kv = _apply_partial_rope")
    kv_qat = attention_source.index("kv = _fp8_qat_non_rope")
    cache_update = attention_source.index("local_cache.update_and_fetch")
    assert kv_rope < kv_qat < cache_update

    compressor_source = inspect.getsource(Compressor.__call__)
    pooled_rope = compressor_source.index("new_pooled = _apply_partial_rope")
    indexer_hadamard = compressor_source.index("hadamard_rotate_activation")
    indexer_fp4 = compressor_source.index("fp4_act_quant_sim")
    main_fp8 = compressor_source.index("_fp8_qat_non_rope")
    cache_pool_update = compressor_source.index("cache.update_pool_view")
    assert "fp4_act_quant_sim(\n                    hadamard_rotate_activation(new_pooled)" in compressor_source
    assert pooled_rope < indexer_fp4 < indexer_hadamard < cache_pool_update
    assert pooled_rope < main_fp8 < cache_pool_update

    indexer_source = inspect.getsource(Indexer.select)
    q_rope = indexer_source.index("q = _apply_partial_rope")
    q_hadamard = indexer_source.index("hadamard_rotate_activation")
    q_fp4 = indexer_source.index("fp4_act_quant_sim")
    scoring = indexer_source.index("scores = q.astype")
    assert "fp4_act_quant_sim(hadamard_rotate_activation(q), 32)" in indexer_source
    assert q_rope < q_fp4 < q_hadamard < scoring

    cfg = SimpleNamespace(
        hidden_size=16,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
        index_n_heads=1,
        index_head_dim=128,
        index_topk=4,
        q_lora_rank=64,
    )
    indexer = Indexer(cfg, compress_ratio=4)
    assert indexer.compressor.rotate is True


def test_dsv4_compressor_stages_q8_projection_and_pooling_in_fp32():
    from jang_tools.dsv4.mlx_model import Compressor

    events = []

    class RecordingProjection:
        def __init__(self, inner, name):
            self.inner = inner
            self.name = name

        def __call__(self, value):
            events.append((self.name, "input", value.dtype))
            output = self.inner(value)
            events.append((self.name, "output", output.dtype))
            return output

    class RecordingNorm:
        def __call__(self, value):
            events.append(("norm", "input", value.dtype))
            return value

    class IdentityRope:
        dims = 64

        def __call__(self, value, offset=0, inverse=False, positions=None):
            return value

    cfg = SimpleNamespace(
        hidden_size=64,
        qk_rope_head_dim=64,
        rms_norm_eps=1e-6,
    )
    compressor = Compressor(cfg, compress_ratio=4, head_dim=128)
    for name in ("wkv", "wgate"):
        quantized = getattr(compressor, name).to_quantized(
            group_size=64,
            bits=8,
        )
        # Match the exact 0731 JANG artifact's affine sidecar dtype.
        quantized.scales = quantized.scales.astype(mx.float16)
        quantized.biases = quantized.biases.astype(mx.float16)
        setattr(compressor, name, RecordingProjection(quantized, name))
    compressor.norm = RecordingNorm()

    output = compressor(
        mx.ones((1, 4, 64), dtype=mx.float16),
        IdentityRope(),
        cache=None,
        start_pos=0,
    )
    mx.eval(output)

    assert ("wkv", "input", mx.float32) in events
    assert ("wkv", "output", mx.float32) in events
    assert ("wgate", "input", mx.float32) in events
    assert ("wgate", "output", mx.float32) in events
    assert ("norm", "input", mx.float16) in events
    assert output.dtype == mx.float16
