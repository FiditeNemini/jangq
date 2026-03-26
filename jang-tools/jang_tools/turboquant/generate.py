"""
TurboQuant-aware generation utilities.

Provides a generate wrapper that auto-compresses KV cache after prefill
or at a configurable token threshold. Drop-in replacement for mlx_lm.generate.
"""

from typing import Optional

import mlx.core as mx

from .cache import TurboQuantKVCache


def compress_cache(cache: list, n_tokens: Optional[int] = None):
    """Compress all TurboQuantKVCache layers in a cache list.

    Args:
        cache: List of cache objects (from model.make_cache()).
        n_tokens: How many tokens to compress. None = all.

    Returns:
        Number of TurboQuant layers compressed.
    """
    compressed = 0
    for c in cache:
        if isinstance(c, TurboQuantKVCache) and not c.empty():
            c.compress(n_tokens)
            compressed += 1
    return compressed


def cache_memory_report(cache: list) -> dict:
    """Report memory usage of cache layers.

    Returns dict with:
        tq_layers: number of TurboQuant cache layers
        compressed_tokens: tokens stored in compressed form
        float_bytes: size of float buffers (for fast attention)
        packed_bytes: size of packed compressed storage (for serialization)
        total_bytes: float_bytes + packed_bytes + other cache bytes
    """
    float_bytes = 0
    packed_bytes = 0
    other_bytes = 0
    tq_layers = 0
    compressed_tokens = 0

    for c in cache:
        if isinstance(c, TurboQuantKVCache):
            tq_layers += 1
            float_bytes += c.nbytes  # joined buffer or float buffer
            packed_bytes += c.compressed_nbytes  # packed only
            compressed_tokens = max(compressed_tokens, c._compressed_tokens)
        elif hasattr(c, 'nbytes'):
            other_bytes += c.nbytes

    return {
        "tq_layers": tq_layers,
        "compressed_tokens": compressed_tokens,
        "float_bytes": float_bytes,
        "packed_bytes": packed_bytes,
        "total_bytes": float_bytes + other_bytes,
        # Legacy compat
        "tq_bytes": float_bytes,
    }
