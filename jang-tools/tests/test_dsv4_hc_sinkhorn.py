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
