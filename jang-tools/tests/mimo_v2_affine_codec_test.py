import numpy as np
import pytest
import torch


mlx = pytest.importorskip("mlx.core")


def _torch_minmax_reference(weight: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    rows, cols = weight.shape
    groups = cols // group_size
    x = weight.float().reshape(rows, groups, group_size)
    minv = x.amin(dim=2, keepdim=True)
    maxv = x.amax(dim=2, keepdim=True)
    levels = (1 << bits) - 1
    scale = ((maxv - minv) / float(levels)).clamp_min(1e-7)
    q = torch.round((x - minv) / scale).clamp_(0, levels)
    mlx_scale = (-scale).to(torch.bfloat16).to(torch.float32)
    mlx_bias = maxv.to(torch.bfloat16).to(torch.float32)
    return ((levels - q) * mlx_scale + mlx_bias).reshape_as(weight)


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
def test_cpu_minmax_affine_pack_dequantizes_with_mlx(bits):
    from jang_tools.mimo_v2.affine_codec import quantize_minmax_affine

    weight = torch.linspace(-1.75, 2.25, 64, dtype=torch.float32).reshape(2, 32)
    qweight, scales, biases = quantize_minmax_affine(weight, bits=bits, group_size=32)

    actual = mlx.dequantize(
        mlx.array(qweight.numpy()),
        mlx.array(scales.float().numpy()),
        mlx.array(biases.float().numpy()),
        group_size=32,
        bits=bits,
        mode="affine",
        dtype=mlx.float32,
    )
    actual_t = torch.from_numpy(np.array(actual))
    expected = _torch_minmax_reference(weight, bits=bits, group_size=32)

    torch.testing.assert_close(actual_t, expected, rtol=0, atol=0)
    assert qweight.dtype == torch.uint32
    assert scales.shape == (2, 1)
    assert biases.shape == (2, 1)
