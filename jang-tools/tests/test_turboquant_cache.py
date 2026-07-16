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

    @pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
    def test_compress_preserves_attention_dtype(self, dtype):
        """TQ decode must not promote a model's KV/SDPA path to float32."""
        cache = TurboQuantKVCache(
            key_dim=64,
            value_dim=64,
            key_bits=8,
            value_bits=8,
        )
        keys = mx.random.normal(shape=(1, 2, 16, 64)).astype(dtype)
        values = mx.random.normal(shape=(1, 2, 16, 64)).astype(dtype)
        cache.update_and_fetch(keys, values)

        cache.compress()
        restored_keys, restored_values = cache.state
        mx.eval(restored_keys, restored_values)

        assert restored_keys.dtype == dtype
        assert restored_values.dtype == dtype
        assert cache._decoded_k_buffer.dtype == dtype
        assert cache._decoded_v_buffer.dtype == dtype
        assert cache._joined_k.dtype == dtype
        assert cache._joined_v.dtype == dtype


class TestBatchApi:
    """BatchGenerator compatibility: real filter/extract/extend semantics."""

    def _cache_with_tokens(self, batch: int, tokens: int, offset: float):
        cache = TurboQuantKVCache(key_dim=16, value_dim=16, key_bits=4, value_bits=4)
        base = mx.arange(batch * 2 * tokens * 16, dtype=mx.float32).reshape(
            batch, 2, tokens, 16
        )
        cache.update_and_fetch(base + offset, base + offset + 1000.0)
        return cache

    def test_extract_returns_single_sequence_turboquant_cache(self):
        cache = self._cache_with_tokens(batch=2, tokens=5, offset=0.0)

        extracted = cache.extract(1)

        assert isinstance(extracted, TurboQuantKVCache)
        assert extracted.offset == 5
        k, v = extracted.state
        assert k.shape == (1, 2, 5, 16)
        assert v.shape == (1, 2, 5, 16)
        original_k, original_v = cache.state
        np.testing.assert_allclose(np.array(k), np.array(original_k[1:2]), atol=1e-6)
        np.testing.assert_allclose(np.array(v), np.array(original_v[1:2]), atol=1e-6)

    def test_filter_keeps_requested_batch_rows_and_allows_decode(self):
        cache = self._cache_with_tokens(batch=2, tokens=5, offset=0.0)

        cache.filter([1])
        k_out, v_out = cache.update_and_fetch(
            mx.ones((1, 2, 1, 16)), mx.ones((1, 2, 1, 16)) * 2
        )

        assert cache.offset == 6
        assert k_out.shape == (1, 2, 6, 16)
        assert v_out.shape == (1, 2, 6, 16)

    def test_filter_accepts_mlx_array_indices_like_batch_kv_cache(self):
        cache = self._cache_with_tokens(batch=3, tokens=5, offset=0.0)

        cache.filter(mx.array([0, 2], dtype=mx.int32))

        assert tuple(int(x) for x in np.array(cache.offset).tolist()) == (5, 5)
        k, v = cache.state
        assert k.shape == (2, 2, 5, 16)
        assert v.shape == (2, 2, 5, 16)

    def test_extend_merges_two_single_sequence_caches(self):
        left = self._cache_with_tokens(batch=1, tokens=5, offset=0.0)
        right = self._cache_with_tokens(batch=1, tokens=3, offset=10_000.0)

        left.extend(right)
        k_out, v_out = left.update_and_fetch(
            mx.ones((2, 2, 1, 16)), mx.ones((2, 2, 1, 16)) * 2
        )

        assert tuple(int(x) for x in np.array(left.offset).tolist()) == (6, 4)
        assert k_out.shape == (2, 2, 6, 16)
        assert v_out.shape == (2, 2, 6, 16)
        extracted = left.extract(1)
        assert extracted.offset == 4

    def test_cache_declares_vmlx_batch_api_contract(self):
        assert TurboQuantKVCache._vmlx_batch_api == "turboquant_kv_v1"

    def test_compressed_cache_filter_extract_roundtrip(self):
        cache = self._cache_with_tokens(batch=2, tokens=10, offset=0.0)
        cache.compress(8)
        assert cache.is_compressed

        extracted = cache.extract(0)
        cache.filter([1])

        assert extracted.offset == 10
        assert cache.offset == 10
        assert extracted.state[0].shape == (1, 2, 10, 16)
        assert cache.state[0].shape == (1, 2, 10, 16)

    def test_batched_cache_make_mask_uses_shared_right_edge_and_left_padding(self):
        left = self._cache_with_tokens(batch=1, tokens=5, offset=0.0)
        right = self._cache_with_tokens(batch=1, tokens=3, offset=10_000.0)
        left.extend(right)

        mask = left.make_mask(2)

        assert mask is not None
        assert mask.shape == (2, 1, 2, 7)


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
