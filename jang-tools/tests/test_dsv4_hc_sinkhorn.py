import torch

from jang_tools.dsv4.ops import hc_split_sinkhorn


def test_dsv4_hc_split_sinkhorn_matches_source_pre_post_contract():
    """Pin the DeepSeek-V4 source kernel semantics for mHC split weights."""
    hc = 2
    mixes = torch.tensor(
        [[[0.0, 1.0, -0.5, 0.25, 0.1, -0.2, 0.3, -0.4]]],
        dtype=torch.float32,
    )
    scale = torch.tensor([1.0, 2.0, 0.5], dtype=torch.float32)
    base = torch.tensor(
        [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15],
        dtype=torch.float32,
    )
    eps = 1e-6

    pre, post, comb = hc_split_sinkhorn(
        mixes, scale, base, hc_mult=hc, iters=3, eps=eps
    )

    expected_pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    expected_post = 2 * torch.sigmoid(
        mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]
    )
    raw_comb = (
        mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    ).view(*mixes.shape[:-1], hc, hc)
    expected_comb = torch.softmax(raw_comb, dim=-1) + eps
    expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + eps)
    for _ in range(2):
        expected_comb = expected_comb / (expected_comb.sum(-1, keepdim=True) + eps)
        expected_comb = expected_comb / (expected_comb.sum(-2, keepdim=True) + eps)

    assert torch.allclose(pre, expected_pre)
    assert torch.allclose(post, expected_post)
    assert torch.allclose(comb, expected_comb)
    assert not torch.allclose(pre.sum(-1), torch.ones_like(pre.sum(-1)))


def test_mlx_dsv4_hc_split_sinkhorn_pure_fallback_accepts_public_keywords(monkeypatch):
    import numpy as np
    import mlx.core as mx

    import jang_tools.dsv4.mlx_model as dsv4

    monkeypatch.setattr(dsv4, "_hc_split_sinkhorn_kernel", None)
    mixes = mx.zeros((2, (2 + 4) * 4), dtype=mx.float32)
    scale = mx.ones((3,), dtype=mx.float32)
    base = mx.zeros(((2 + 4) * 4,), dtype=mx.float32)

    pre, post, comb = dsv4.hc_split_sinkhorn(
        mixes, scale, base, hc_mult=4, iters=3, eps=1e-6
    )
    mx.eval(pre, post, comb)

    assert pre.shape == (2, 4)
    assert post.shape == (2, 4)
    assert comb.shape == (2, 4, 4)
    np.testing.assert_allclose(np.asarray(comb).sum(axis=-1), 1.0, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(comb).sum(axis=-2), 1.0, rtol=1e-5, atol=1e-5)


def test_mlx_dsv4_hc_post_contracts_official_source_axis():
    """A non-symmetric mix must use comb.T @ residual, never comb @ residual."""
    import numpy as np
    import mlx.core as mx

    from jang_tools.dsv4.mlx_model import _dsv4_hc_post

    x = mx.array([[[10.0, 20.0]]], dtype=mx.float32)
    post = mx.array([[[0.25, 0.75]]], dtype=mx.float32)
    residual = mx.array(
        [[[[1.0, 2.0], [3.0, 5.0]]]],
        dtype=mx.float32,
    )
    comb = mx.array(
        [[[[0.1, 0.9], [0.4, 0.6]]]],
        dtype=mx.float32,
    )

    actual = _dsv4_hc_post(x, residual, post, comb)
    mx.eval(actual)

    x_np = np.asarray(x)
    post_np = np.asarray(post)
    residual_np = np.asarray(residual)
    comb_np = np.asarray(comb)
    expected = (
        post_np[..., None] * x_np[..., None, :]
        + np.sum(comb_np[..., None] * residual_np[..., None, :], axis=2)
    )
    old_wrong_orientation = (
        post_np[..., None] * x_np[..., None, :]
        + np.matmul(comb_np, residual_np)
    )

    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=0)
    assert not np.allclose(np.asarray(actual), old_wrong_orientation)
