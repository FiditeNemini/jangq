"""Tests for TurboQuant QJL (Quantized Johnson-Lindenstrauss) transform."""
import numpy as np
import pytest
import mlx.core as mx

from jang_tools.turboquant.qjl import (
    generate_qjl_projection,
    qjl_encode,
    qjl_decode,
    qjl_inner_product,
)


class TestQJL:
    """Test QJL 1-bit quantization for unbiased inner product estimation."""

    def test_encode_output_is_signs(self):
        """QJL encoding should produce +/-1 values."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal(128).astype(np.float32))
        S = generate_qjl_projection(128, seed=0)
        signs, norm = qjl_encode(x, S)
        signs_np = np.array(signs)
        assert set(np.unique(signs_np)).issubset({-1.0, 1.0})

    def test_encode_returns_correct_norm(self):
        """QJL should return the L2 norm of the input."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal(128).astype(np.float32))
        S = generate_qjl_projection(128, seed=0)
        signs, norm = qjl_encode(x, S)
        expected_norm = float(mx.sqrt(mx.sum(x * x)))
        assert abs(float(norm) - expected_norm) < 1e-4

    def test_unbiased_inner_product(self):
        """E_S[QJL estimate] should equal true inner product for fixed (a, b)."""
        rng = np.random.default_rng(42)
        dim = 128
        n_trials = 2000

        # Fixed vectors
        a = rng.standard_normal(dim).astype(np.float32)
        b = rng.standard_normal(dim).astype(np.float32)
        true_ip = float(np.dot(a, b))
        a_mx = mx.array(a)
        b_mx = mx.array(b)

        # Average QJL estimate over many random S matrices
        estimates = []
        for trial in range(n_trials):
            S = generate_qjl_projection(dim, seed=trial)
            signs, norm = qjl_encode(b_mx, S)
            est = float(qjl_inner_product(a_mx, signs, norm, S))
            estimates.append(est)

        est_mean = np.mean(estimates)
        # Allow 15% relative error (statistical noise)
        assert abs(est_mean - true_ip) / (abs(true_ip) + 1e-8) < 0.15, \
            f"Bias detected: true_ip={true_ip:.4f}, est_mean={est_mean:.4f}"

    def test_projection_shape(self):
        """Projection matrix should be (dim, dim)."""
        S = generate_qjl_projection(128, seed=0)
        assert S.shape == (128, 128)

    def test_different_seeds_different_projections(self):
        """Different seeds should produce different projections."""
        S0 = generate_qjl_projection(128, seed=0)
        S1 = generate_qjl_projection(128, seed=1)
        assert float(mx.sum(mx.abs(S0 - S1))) > 0
