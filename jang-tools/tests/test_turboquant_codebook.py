"""Tests for TurboQuant optimal scalar codebook."""
import numpy as np
import pytest
import mlx.core as mx

from jang_tools.turboquant.codebook import (
    compute_codebook,
    quantize_scalar,
    dequantize_scalar,
)


class TestCodebook:
    """Test Lloyd-Max optimal codebook on Beta/Gaussian distribution."""

    def test_codebook_size(self):
        """Codebook should have 2^b entries."""
        for b in [1, 2, 3, 4]:
            cb = compute_codebook(dim=128, bits=b)
            assert len(cb) == 2**b

    def test_codebook_sorted(self):
        """Codebook entries must be sorted ascending."""
        cb = compute_codebook(dim=128, bits=3)
        for i in range(len(cb) - 1):
            assert cb[i] < cb[i + 1]

    def test_codebook_symmetric(self):
        """Codebook should be symmetric around 0 (distribution is symmetric)."""
        cb = compute_codebook(dim=128, bits=2)
        cb_arr = np.array(cb)
        np.testing.assert_allclose(cb_arr, -cb_arr[::-1], atol=1e-6)

    def test_quantize_roundtrip_low_error(self):
        """Quantize->dequantize should have MSE matching paper bounds."""
        rng = np.random.default_rng(42)
        dim = 128
        # Simulate post-rotation distribution: N(0, 1/d)
        x = rng.standard_normal(10000).astype(np.float32) / np.sqrt(dim)
        x_mx = mx.array(x)
        cb = compute_codebook(dim=dim, bits=3)
        cb_mx = mx.array(np.array(cb, dtype=np.float32))
        indices = quantize_scalar(x_mx, cb_mx)
        x_hat = dequantize_scalar(indices, cb_mx)
        mse = float(mx.mean((x_mx - x_hat) ** 2))
        # Paper bound: D_mse(3) ~ 0.03 for unit sphere
        # For N(0, 1/d): scale by 1/d, so bound ~ 0.03/d
        # Be generous: allow 5x paper bound
        assert mse < 0.15 / dim, f"MSE {mse} exceeds 5x paper bound {0.15/dim}"

    def test_quantize_indices_valid(self):
        """All indices must be in [0, 2^b - 1]."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal(1000).astype(np.float32) / np.sqrt(128))
        cb = compute_codebook(dim=128, bits=3)
        cb_mx = mx.array(np.array(cb, dtype=np.float32))
        indices = quantize_scalar(x, cb_mx)
        idx_np = np.array(indices)
        assert idx_np.min() >= 0
        assert idx_np.max() <= 7  # 2^3 - 1

    @pytest.mark.parametrize("dim", [64, 128, 192, 256])
    def test_codebook_different_dims(self, dim):
        """Codebook computation should work for all common head dimensions."""
        cb = compute_codebook(dim=dim, bits=3)
        assert len(cb) == 8
        # Centroids should scale roughly as 1/sqrt(dim)
        max_c = max(abs(c) for c in cb)
        expected_scale = 3.0 / np.sqrt(dim)  # generous bound
        assert max_c < expected_scale, f"Codebook max {max_c} too large for dim={dim}"
