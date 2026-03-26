"""Tests for TurboQuant Hadamard rotation."""
import numpy as np
import pytest

import mlx.core as mx

from jang_tools.turboquant.rotation import (
    hadamard_rotate,
    hadamard_inverse,
    generate_random_signs,
)


class TestHadamardRotation:
    """Test Randomized Hadamard Transform."""

    def test_roundtrip_identity(self):
        """rotate then inverse = original vector."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, 4, 8, 128)).astype(np.float32))
        signs = generate_random_signs(128, seed=0)
        y = hadamard_rotate(x, signs)
        x_back = hadamard_inverse(y, signs)
        np.testing.assert_allclose(
            np.array(x), np.array(x_back), atol=1e-5, rtol=1e-5
        )

    def test_preserves_norm(self):
        """Rotation must preserve L2 norm (orthogonal transform)."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32))
        signs = generate_random_signs(128, seed=0)
        y = hadamard_rotate(x, signs)
        norm_x = float(mx.sqrt(mx.sum(x * x)))
        norm_y = float(mx.sqrt(mx.sum(y * y)))
        assert abs(norm_x - norm_y) / norm_x < 1e-5

    def test_preserves_inner_product(self):
        """<Pi*a, Pi*b> = <a, b> for orthogonal Pi."""
        rng = np.random.default_rng(42)
        a = mx.array(rng.standard_normal((128,)).astype(np.float32))
        b = mx.array(rng.standard_normal((128,)).astype(np.float32))
        signs = generate_random_signs(128, seed=0)
        a_rot = hadamard_rotate(a.reshape(1, 1, 1, 128), signs).reshape(128)
        b_rot = hadamard_rotate(b.reshape(1, 1, 1, 128), signs).reshape(128)
        ip_orig = float(mx.sum(a * b))
        ip_rot = float(mx.sum(a_rot * b_rot))
        assert abs(ip_orig - ip_rot) / (abs(ip_orig) + 1e-8) < 1e-4

    def test_output_shape_matches_input(self):
        """Output shape must equal input shape."""
        x = mx.zeros((2, 8, 16, 128))
        signs = generate_random_signs(128, seed=0)
        y = hadamard_rotate(x, signs)
        assert y.shape == x.shape

    @pytest.mark.parametrize("dim", [64, 128, 192, 256])
    def test_different_dimensions(self, dim):
        """Must work for common head dimensions including non-power-of-2."""
        rng = np.random.default_rng(42)
        x = mx.array(rng.standard_normal((1, 1, 1, dim)).astype(np.float32))
        signs = generate_random_signs(dim, seed=0)
        y = hadamard_rotate(x, signs)
        x_back = hadamard_inverse(y, signs)
        np.testing.assert_allclose(
            np.array(x), np.array(x_back), atol=1e-4, rtol=1e-4
        )

    def test_spreads_outliers(self):
        """After rotation, coordinates should be more uniform (no large outliers)."""
        # Create vector with one extreme outlier
        x_np = np.zeros((1, 1, 1, 128), dtype=np.float32)
        x_np[0, 0, 0, 0] = 10.0  # all energy in one coordinate
        x = mx.array(x_np)
        signs = generate_random_signs(128, seed=0)
        y = hadamard_rotate(x, signs)
        y_np = np.array(y).flatten()
        # After rotation, energy should be spread: max should be much smaller
        assert np.max(np.abs(y_np)) < 3.0  # 10/sqrt(128) ~ 0.88, allow margin
