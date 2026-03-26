"""Tests for TurboQuantKVCache — all lifecycle phases."""
import numpy as np
import pytest
import time
import mlx.core as mx

from jang_tools.turboquant.cache import TurboQuantKVCache


class TestBasicOperations:
    """Test pre-compress (Phase 1) cache behavior."""

    def make_cache(self, k_dim=128, v_dim=128, bits=3, **kwargs):
        return TurboQuantKVCache(
            key_dim=k_dim, value_dim=v_dim,
            key_bits=bits, value_bits=bits, seed=42, **kwargs,
        )

    def test_empty_on_init(self):
        cache = self.make_cache()
        assert cache.empty()
        assert cache.offset == 0

    def test_update_single_token(self):
        cache = self.make_cache()
        keys = mx.ones((1, 4, 1, 128))
        values = mx.ones((1, 4, 1, 128))
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert cache.offset == 1
        assert not cache.empty()
        assert k_out.shape == (1, 4, 1, 128)
        assert v_out.shape == (1, 4, 1, 128)

    def test_update_multiple_tokens(self):
        cache = self.make_cache()
        k1 = mx.ones((1, 4, 8, 128)) * 0.5
        v1 = mx.ones((1, 4, 8, 128)) * 0.3
        cache.update_and_fetch(k1, v1)
        assert cache.offset == 8

        k2 = mx.ones((1, 4, 1, 128)) * 0.7
        v2 = mx.ones((1, 4, 1, 128)) * 0.4
        k_out, v_out = cache.update_and_fetch(k2, v2)
        assert cache.offset == 9
        assert k_out.shape == (1, 4, 9, 128)

    def test_mistral4_dimensions(self):
        cache = self.make_cache(k_dim=192, v_dim=128, bits=3)
        keys = mx.ones((1, 128, 1, 192))
        values = mx.ones((1, 128, 1, 128))
        k_out, v_out = cache.update_and_fetch(keys, values)
        assert k_out.shape == (1, 128, 1, 192)
        assert v_out.shape == (1, 128, 1, 128)

    def test_reconstruction_quality(self):
        cache = self.make_cache(bits=4)
        rng = np.random.default_rng(42)
        keys = mx.array(rng.standard_normal((1, 4, 16, 128)).astype(np.float32))
        values = mx.array(rng.standard_normal((1, 4, 16, 128)).astype(np.float32))
        k_out, v_out = cache.update_and_fetch(keys, values)
        k_mse = float(mx.mean((keys - k_out) ** 2))
        k_var = float(mx.var(keys))
        assert k_mse / k_var < 0.5

    def test_has_offset_property(self):
        cache = self.make_cache()
        assert hasattr(cache, 'offset')

    def test_is_trimmable(self):
        assert self.make_cache().is_trimmable()

    def test_trim(self):
        cache = self.make_cache()
        cache.update_and_fetch(mx.ones((1, 4, 10, 128)), mx.ones((1, 4, 10, 128)))
        assert cache.trim(3) == 3
        assert cache.offset == 7


class TestPostCompressGeneration:
    """Test that generation AFTER compress() is correct and fast."""

    def test_shapes_after_compress(self):
        """Insert, compress, insert more — shapes must grow correctly."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=3, value_bits=3)
        rng = np.random.default_rng(42)

        k1 = mx.array(rng.standard_normal((1, 4, 64, 128)).astype(np.float32))
        v1 = mx.array(rng.standard_normal((1, 4, 64, 128)).astype(np.float32))
        cache.update_and_fetch(k1, v1)

        cache.compress(50)
        assert cache.is_compressed
        assert cache._compressed_tokens == 50
        assert cache._decoded_k_buffer is not None
        assert cache._decoded_k_buffer.shape[2] == 50

        for i in range(20):
            k = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32))
            v = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32))
            k_out, v_out = cache.update_and_fetch(k, v)
            expected = 64 + i + 1
            assert k_out.shape == (1, 4, expected, 128), \
                f"Step {i}: expected seq={expected}, got {k_out.shape[2]}"

    def test_no_nan_after_compress(self):
        """Post-compress output must not contain NaN."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=4, value_bits=4)
        rng = np.random.default_rng(42)

        cache.update_and_fetch(
            mx.array(rng.standard_normal((1, 4, 100, 128)).astype(np.float32)),
            mx.array(rng.standard_normal((1, 4, 100, 128)).astype(np.float32)),
        )
        cache.compress(80)

        for _ in range(10):
            k_out, _ = cache.update_and_fetch(
                mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32)),
                mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32)),
            )
            assert not np.any(np.isnan(np.array(k_out)))

    def test_no_re_decode_after_compress(self):
        """Decoded buffer must be set after compress (no per-step decode)."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=3, value_bits=3)
        rng = np.random.default_rng(42)

        cache.update_and_fetch(
            mx.array(rng.standard_normal((1, 4, 200, 128)).astype(np.float32)),
            mx.array(rng.standard_normal((1, 4, 200, 128)).astype(np.float32)),
        )
        cache.compress(150)

        # Decoded buffer exists and has correct shape
        assert cache._decoded_k_buffer is not None
        assert cache._decoded_k_buffer.shape[2] == 150
        assert cache._decoded_v_buffer.shape[2] == 150

        # Adding token should NOT change the decoded buffer (it's read-only)
        buf_id_before = id(cache._decoded_k_buffer)
        cache.update_and_fetch(
            mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32)),
            mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32)),
        )
        assert id(cache._decoded_k_buffer) == buf_id_before  # same object, no re-decode

    def test_auto_compress_threshold(self):
        """compress_after triggers automatic compression."""
        cache = TurboQuantKVCache(
            key_dim=128, value_dim=128, key_bits=3, value_bits=3,
            compress_after=50,
        )
        rng = np.random.default_rng(42)
        k_out, _ = cache.update_and_fetch(
            mx.array(rng.standard_normal((1, 4, 60, 128)).astype(np.float32)),
            mx.array(rng.standard_normal((1, 4, 60, 128)).astype(np.float32)),
        )
        assert cache.is_compressed
        assert cache._compressed_tokens == 50
        assert k_out.shape == (1, 4, 60, 128)


class TestSinkTokens:
    """Test sink token preservation during compression."""

    def test_sink_preserved_exact(self):
        """First N sink tokens must be exact (not quantized) after compress."""
        cache = TurboQuantKVCache(
            key_dim=128, value_dim=128, key_bits=3, value_bits=3, sink_tokens=4,
        )
        rng = np.random.default_rng(42)
        k = mx.array(rng.standard_normal((1, 4, 64, 128)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, 4, 64, 128)).astype(np.float32))
        cache.update_and_fetch(k, v)

        sink_k_before = np.array(k[..., :4, :])
        sink_v_before = np.array(v[..., :4, :])

        cache.compress()
        full_k, full_v = cache._get_full_cache()

        np.testing.assert_allclose(sink_k_before, np.array(full_k[..., :4, :]), atol=1e-6)
        np.testing.assert_allclose(sink_v_before, np.array(full_v[..., :4, :]), atol=1e-6)

    def test_no_compress_if_all_sink(self):
        """If all tokens are sink tokens, compress is a no-op."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, sink_tokens=10)
        rng = np.random.default_rng(42)
        cache.update_and_fetch(
            mx.array(rng.standard_normal((1, 4, 8, 128)).astype(np.float32)),
            mx.array(rng.standard_normal((1, 4, 8, 128)).astype(np.float32)),
        )
        cache.compress()
        assert not cache.is_compressed

    def test_sink_from_config(self):
        """sink_tokens should be configurable."""
        from jang_tools.turboquant.config import TurboQuantConfig
        cfg = TurboQuantConfig(sink_tokens=8)
        assert cfg.sink_tokens == 8
