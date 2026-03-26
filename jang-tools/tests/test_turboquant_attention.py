"""Tests for TurboQuant attention (rotate-query SDPA)."""
import numpy as np
import pytest
import mlx.core as mx

from jang_tools.turboquant.attention import turboquant_sdpa
from jang_tools.turboquant.cache import TurboQuantKVCache


class TestTurboQuantSDPA:
    """Test that TurboQuant attention produces reasonable results."""

    def test_output_shape(self):
        """SDPA output shape should be (B, n_q_heads, seq_q, head_dim)."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=3, value_bits=3)
        rng = np.random.default_rng(42)
        k = mx.array(rng.standard_normal((1, 4, 8, 128)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, 4, 8, 128)).astype(np.float32))
        cache.update_and_fetch(k, v)

        q = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32))
        out = turboquant_sdpa(q, cache, scale=128**-0.5)
        assert out.shape == (1, 4, 1, 128)

    def test_attention_not_nan(self):
        """Output should not contain NaN or Inf."""
        cache = TurboQuantKVCache(key_dim=128, value_dim=128, key_bits=3, value_bits=3)
        rng = np.random.default_rng(42)
        k = mx.array(rng.standard_normal((1, 4, 16, 128)).astype(np.float32))
        v = mx.array(rng.standard_normal((1, 4, 16, 128)).astype(np.float32))
        cache.update_and_fetch(k, v)

        q = mx.array(rng.standard_normal((1, 4, 1, 128)).astype(np.float32))
        out = turboquant_sdpa(q, cache, scale=128**-0.5)
        out_np = np.array(out)
        assert not np.any(np.isnan(out_np)), "Output contains NaN"
        assert not np.any(np.isinf(out_np)), "Output contains Inf"

    def test_close_to_exact_attention(self):
        """TurboQuant attention should approximate exact attention."""
        rng = np.random.default_rng(42)
        B, H, S, D = 1, 4, 16, 128
        scale = D**-0.5

        q = mx.array(rng.standard_normal((B, H, 1, D)).astype(np.float32))
        k = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))
        v = mx.array(rng.standard_normal((B, H, S, D)).astype(np.float32))

        # Exact attention
        scores = (q @ mx.transpose(k, (0, 1, 3, 2))) * scale
        weights = mx.softmax(scores, axis=-1)
        exact_out = weights @ v

        # TurboQuant attention (4-bit for better approximation in test)
        cache = TurboQuantKVCache(key_dim=D, value_dim=D, key_bits=4, value_bits=4)
        cache.update_and_fetch(k, v)
        tq_out = turboquant_sdpa(q, cache, scale=scale)

        # Should be correlated (cosine similarity > 0.7)
        exact_flat = np.array(exact_out).flatten()
        tq_flat = np.array(tq_out).flatten()
        cos_sim = np.dot(exact_flat, tq_flat) / (
            np.linalg.norm(exact_flat) * np.linalg.norm(tq_flat) + 1e-8
        )
        assert cos_sim > 0.7, f"Cosine similarity {cos_sim:.3f} too low"
