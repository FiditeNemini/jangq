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


def test_pool_quant_cache_reuses_materialized_pool_between_appends(monkeypatch):
    """Pool reads must not dequantize historical segments on every decode read."""
    import jang_tools.dsv4.pool_quant_cache as pq
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    dequant_count = 0
    original_dequant = pq._dequant_pool

    def recording_dequant(qpool):
        nonlocal dequant_count
        dequant_count += 1
        return original_dequant(qpool)

    monkeypatch.setattr(pq, "_dequant_pool", recording_dequant)

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    first = mx.ones((1, 3, 16), dtype=mx.bfloat16)
    second = mx.ones((1, 1, 16), dtype=mx.bfloat16) * 2

    pool_a = cache.update_pool(first, "compressor_state")
    mx.eval(pool_a)
    assert dequant_count == 0

    first_read = cache.compressor_state["pooled"]
    second_read = cache.compressor_state["pooled"]
    mx.eval(first_read, second_read)
    assert dequant_count == 0

    pool_b = cache.update_pool(second, "compressor_state")
    mx.eval(pool_b)
    assert dequant_count == 0
    assert tuple(pool_b.shape) == (1, 4, 16)


def test_pool_quant_cache_nbytes_reports_materialized_live_pool():
    """Live nbytes must include the cached dequantized pool view."""
    from jang_tools.dsv4.pool_quant_cache import PoolQuantizedV4Cache

    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    pooled = cache.update_pool(mx.ones((1, 3, 16), dtype=mx.bfloat16), "compressor_state")
    mx.eval(pooled)

    assert cache.nbytes >= pooled.nbytes
