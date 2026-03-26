"""Tests for TurboQuant generate utilities and compression."""
import numpy as np
import pytest
import mlx.core as mx

from jang_tools.turboquant.cache import TurboQuantKVCache
from jang_tools.turboquant.generate import compress_cache, cache_memory_report


class TestCompressCache:
    """Test compress_cache utility."""

    def test_compress_produces_packed_storage(self):
        """Compressing cache should produce packed storage much smaller than float."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=3, value_bits=3)
        rng = np.random.default_rng(42)
        k = mx.array(rng.standard_normal((1, 4, 64, 128)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, 4, 64, 128)).astype(np.float32))
        cache.update_and_fetch(k, v)

        float_bytes = cache.nbytes
        compress_cache([cache])
        packed_bytes = cache.compressed_nbytes

        assert packed_bytes < float_bytes, f"Packed {packed_bytes} >= float {float_bytes}"
        ratio = float_bytes / packed_bytes
        assert ratio > 2.0, f"Packed ratio {ratio:.1f}x too low"

    def test_compress_skips_empty(self):
        """Compressing empty cache should not crash."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128)
        n = compress_cache([cache])
        assert n == 0

    def test_compress_count(self):
        """Should return number of compressed layers."""
        caches = [
            TurboQuantKVCache(key_dim=128, value_dim=128),
            TurboQuantKVCache(key_dim=128, value_dim=128),
        ]
        rng = np.random.default_rng(42)
        for c in caches:
            k = mx.array(rng.standard_normal((1, 4, 16, 128)).astype(np.float32))
            v = mx.array(rng.standard_normal((1, 4, 16, 128)).astype(np.float32))
            c.update_and_fetch(k, v)

        n = compress_cache(caches)
        assert n == 2


class TestMemoryReport:
    """Test cache_memory_report utility."""

    def test_report_structure(self):
        """Report should contain expected keys."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128)
        report = cache_memory_report([cache])
        assert "total_bytes" in report
        assert "tq_bytes" in report
        assert "tq_layers" in report
        assert "compressed_tokens" in report
        assert report["tq_layers"] == 1

    def test_report_after_compress(self):
        """Report should show compressed_tokens after compress."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128)
        rng = np.random.default_rng(42)
        k = mx.array(rng.standard_normal((1, 4, 32, 128)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, 4, 32, 128)).astype(np.float32))
        cache.update_and_fetch(k, v)
        compress_cache([cache])

        report = cache_memory_report([cache])
        assert report["compressed_tokens"] == 32


class TestCompressionQuality:
    """Test that compressed cache still produces reasonable decoded output."""

    def test_decode_after_compress(self):
        """Decoding compressed cache should produce finite values."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=4, value_bits=4)
        rng = np.random.default_rng(42)
        k = mx.array(rng.standard_normal((1, 4, 32, 128)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, 4, 32, 128)).astype(np.float32))
        cache.update_and_fetch(k, v)

        # Compress
        cache.compress()

        # Get full cache (should decode compressed tokens)
        full_k, full_v = cache._get_full_cache()
        assert full_k is not None
        assert full_v is not None
        assert full_k.shape == (1, 4, 32, 128)

        k_np = np.array(full_k)
        assert not np.any(np.isnan(k_np)), "Decoded keys contain NaN"
        assert not np.any(np.isinf(k_np)), "Decoded keys contain Inf"

    def test_compression_ratio_at_3bit(self):
        """Packed compressed storage should be >4x smaller than float."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=3, value_bits=3)
        rng = np.random.default_rng(42)
        k = mx.array(rng.standard_normal((1, 4, 256, 128)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, 4, 256, 128)).astype(np.float32))
        cache.update_and_fetch(k, v)

        float_bytes = cache.nbytes
        cache.compress()
        packed_bytes = cache.compressed_nbytes  # just the packed storage

        ratio = float_bytes / packed_bytes
        assert ratio > 4.0, f"Packed ratio only {ratio:.1f}x, expected >4x"
