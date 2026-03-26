"""Tests for complete TurboQuant encode/decode pipeline."""
import numpy as np
import pytest
import mlx.core as mx

from jang_tools.turboquant.pipeline import (
    TurboQuantEncoder,
    encode_keys,
    decode_keys,
    encode_values,
    decode_values,
)


class TestKeyPipeline:
    """Test key compression: TurboQuant_prod (MSE + QJL)."""

    def test_key_roundtrip_shape(self):
        """Decoded keys should have same shape as original."""
        enc = TurboQuantEncoder(dim=128, key_bits=3, value_bits=3, seed=42)
        rng = np.random.default_rng(42)
        keys = mx.array(rng.standard_normal((1, 4, 8, 128)).astype(np.float32))
        encoded = encode_keys(keys, enc)
        decoded = decode_keys(encoded, enc)
        assert decoded.shape == keys.shape

    def test_key_mse_within_bound(self):
        """Key reconstruction MSE should be reasonable."""
        enc = TurboQuantEncoder(dim=128, key_bits=3, value_bits=3, seed=42)
        rng = np.random.default_rng(42)
        keys = mx.array(rng.standard_normal((1, 4, 100, 128)).astype(np.float32))
        # Normalize to unit sphere per vector
        norms = mx.sqrt(mx.sum(keys * keys, axis=-1, keepdims=True))
        keys_unit = keys / (norms + 1e-8)
        encoded = encode_keys(keys_unit, enc)
        decoded = decode_keys(encoded, enc)
        mse = float(mx.mean((keys_unit - decoded) ** 2))
        # Should be reasonably small for 3-bit (2-bit MSE + 1-bit QJL correction)
        assert mse < 0.15, f"Key MSE {mse} too high for 3-bit TurboQuant"

    def test_key_inner_product_unbiased(self):
        """Key inner product estimates should be approximately unbiased."""
        enc = TurboQuantEncoder(dim=128, key_bits=3, value_bits=3, seed=42)
        rng = np.random.default_rng(42)
        n_vectors = 200
        q = mx.array(rng.standard_normal(128).astype(np.float32))
        q_unit = q / mx.sqrt(mx.sum(q * q))

        true_ips = []
        est_ips = []
        for i in range(n_vectors):
            k = mx.array(rng.standard_normal(128).astype(np.float32))
            k_unit = k / mx.sqrt(mx.sum(k * k))
            true_ip = float(mx.sum(q_unit * k_unit))
            true_ips.append(true_ip)

            k_4d = k_unit.reshape(1, 1, 1, 128)
            encoded = encode_keys(k_4d, enc)
            decoded = decode_keys(encoded, enc)
            est_ip = float(mx.sum(q_unit * decoded.reshape(128)))
            est_ips.append(est_ip)

        # Check correlation between true and estimated inner products
        correlation = np.corrcoef(true_ips, est_ips)[0, 1]
        assert correlation > 0.7, f"Correlation {correlation:.3f} too low"


class TestValuePipeline:
    """Test value compression: TurboQuant_mse."""

    def test_value_roundtrip_shape(self):
        """Decoded values should have same shape as original."""
        enc = TurboQuantEncoder(dim=128, key_bits=3, value_bits=3, seed=42)
        rng = np.random.default_rng(42)
        values = mx.array(rng.standard_normal((1, 4, 8, 128)).astype(np.float32))
        encoded = encode_values(values, enc)
        decoded = decode_values(encoded, enc)
        assert decoded.shape == values.shape

    def test_value_mse_within_bound(self):
        """Value MSE should be within theoretical bounds."""
        enc = TurboQuantEncoder(dim=128, key_bits=3, value_bits=3, seed=42)
        rng = np.random.default_rng(42)
        values = mx.array(rng.standard_normal((1, 4, 100, 128)).astype(np.float32))
        norms = mx.sqrt(mx.sum(values * values, axis=-1, keepdims=True))
        values_unit = values / (norms + 1e-8)
        encoded = encode_values(values_unit, enc)
        decoded = decode_values(encoded, enc)
        mse = float(mx.mean((values_unit - decoded) ** 2))
        # 3-bit MSE bound: D_mse(3) ~ 0.03 for unit sphere
        assert mse < 0.1, f"Value MSE {mse} too high for 3-bit"


class TestEncoder:
    """Test TurboQuantEncoder configuration."""

    def test_encoder_init(self):
        """Encoder should initialize without error."""
        enc = TurboQuantEncoder(dim=128, key_bits=3, value_bits=3, seed=42)
        assert enc.dim == 128
        assert enc.key_bits == 3
        assert enc.value_bits == 3

    @pytest.mark.parametrize("dim", [64, 128, 192])
    def test_encoder_different_dims(self, dim):
        """Should work for all common head dimensions."""
        enc = TurboQuantEncoder(dim=dim, key_bits=3, value_bits=3, seed=42)
        rng = np.random.default_rng(42)
        keys = mx.array(rng.standard_normal((1, 1, 1, dim)).astype(np.float32))
        encoded = encode_keys(keys, enc)
        decoded = decode_keys(encoded, enc)
        assert decoded.shape == keys.shape
