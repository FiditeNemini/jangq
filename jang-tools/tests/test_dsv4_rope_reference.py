import torch

from jang_tools.dsv4.ops import precompute_freqs_cis


def test_dsv4_yarn_ramp_matches_source_direction():
    freqs = precompute_freqs_cis(
        dim=64,
        seqlen=2,
        original_seq_len=65536,
        base=10000.0,
        factor=16.0,
        beta_fast=32,
        beta_slow=1,
    )
    phase = torch.angle(freqs[1])

    standard = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))

    assert torch.allclose(phase[:8], standard[:8], atol=1e-6)
    assert phase[-1] < standard[-1]
