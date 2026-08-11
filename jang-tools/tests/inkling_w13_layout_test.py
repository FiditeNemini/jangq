import mlx.core as mx
import numpy as np

from jang_tools.inkling.model import Model


def test_native_w13_rows_are_deinterleaved_even_odd():
    gate = mx.array(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    up = gate + 100
    native = mx.stack((gate, up), axis=-2).reshape(2, 6, 4)

    got_gate, got_up = Model._deinterleave_w13(native, 3)

    assert mx.array_equal(got_gate, gate).item()
    assert mx.array_equal(got_up, up).item()


def test_affine_packed_w13_deinterleave_preserves_qdq_rows():
    rng = np.random.default_rng(7)
    gate = rng.normal(size=(2, 4, 32)).astype(np.float32)
    up = rng.normal(size=(2, 4, 32)).astype(np.float32)
    native = np.empty((2, 8, 32), dtype=np.float32)
    native[:, 0::2, :] = gate
    native[:, 1::2, :] = up

    packed, scales, biases = mx.quantize(
        mx.array(native), group_size=32, bits=3
    )
    ref = mx.dequantize(
        packed, scales, biases, group_size=32, bits=3
    )

    gate_packed, up_packed = Model._deinterleave_w13(packed, 4)
    gate_scales, up_scales = Model._deinterleave_w13(scales, 4)
    gate_biases, up_biases = Model._deinterleave_w13(biases, 4)
    got_gate = mx.dequantize(
        gate_packed, gate_scales, gate_biases, group_size=32, bits=3
    )
    got_up = mx.dequantize(
        up_packed, up_scales, up_biases, group_size=32, bits=3
    )

    assert mx.array_equal(got_gate, ref[:, 0::2, :]).item()
    assert mx.array_equal(got_up, ref[:, 1::2, :]).item()
