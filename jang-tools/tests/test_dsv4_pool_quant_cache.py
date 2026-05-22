import mlx.core as mx


def test_pool_quant_cache_appends_new_rows_without_requantizing_old_pool(monkeypatch):
    """Pool quant must not quantize the accumulated DSV4 pool every decode step."""
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    quant_shapes = []
    original_quant = pq._quant_pool

    def recording_quant(pool, *args, **kwargs):
        quant_shapes.append(tuple(pool.shape))
        return original_quant(pool, *args, **kwargs)

    monkeypatch.setattr(pq, "_quant_pool", recording_quant)

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    first = mx.ones((1, 3, 16), dtype=mx.bfloat16)
    second = mx.ones((1, 1, 16), dtype=mx.bfloat16) * 2

    pool_a = cache.update_pool(first, "compressor_state")
    pool_b = cache.update_pool(second, "compressor_state")
    mx.eval(pool_a, pool_b)

    assert tuple(pool_b.shape) == (1, 4, 16)
    assert quant_shapes == [
        (1, 3, 16),
        (1, 1, 16),
    ]
