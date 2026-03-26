"""
TurboQuant Scaled Dot-Product Attention.

With the optimized cache (float buffer), this simply reads the decoded
K/V from the cache and runs standard attention. The cache handles all
compression/decompression transparently.
"""

from typing import Optional

import mlx.core as mx

from .cache import TurboQuantKVCache


def turboquant_sdpa(
    queries: mx.array,
    cache: TurboQuantKVCache,
    scale: float,
    mask: Optional[mx.array] = None,
) -> mx.array:
    """Scaled dot-product attention using TurboQuant cache.

    The cache stores decoded K/V in a float buffer, so this is just
    standard attention — same speed as mx.fast.scaled_dot_product_attention.
    """
    if cache.keys is None or cache.offset == 0:
        return mx.zeros_like(queries)

    keys = cache.keys[..., :cache.offset, :]
    values = cache.values[..., :cache.offset, :]

    # Handle GQA
    n_q_heads = queries.shape[1]
    n_kv_heads = keys.shape[1]
    if n_q_heads != n_kv_heads:
        n_repeats = n_q_heads // n_kv_heads
        keys = mx.repeat(keys, n_repeats, axis=1)
        values = mx.repeat(values, n_repeats, axis=1)

    scores = (queries @ mx.transpose(keys, (0, 1, 3, 2))) * scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ values
