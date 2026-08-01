"""Direct residency contracts for the DSV4 pool-quantized cache."""
from __future__ import annotations

import math
from unittest import mock

import mlx.core as mx
import numpy as np

import jang_tools.dsv4.pool_quant_cache as pool_quant_cache
from jang_tools.dsv4.pool_quant_cache import (
    _POOL_BF16_MAX_BYTES,
    PoolQuantizedV4Cache,
    _POOL_SEGMENT_ROWS,
    _StateProxy,
)


def _cos(a: mx.array, b: mx.array) -> float:
    af = a.astype(mx.float32).reshape(-1)
    bf = b.astype(mx.float32).reshape(-1)
    num = float((af * bf).sum())
    den = math.sqrt(float((af * af).sum())) * math.sqrt(float((bf * bf).sum()))
    return num / max(den, 1e-9)


def test_materialized_pool_is_not_retained_and_segments_are_bounded():
    """Reads return full BF16 for attention without retaining it in state."""
    np.random.seed(1)
    rows = _POOL_BF16_MAX_BYTES // (512 * 2) + 1
    raw = mx.array(np.random.randn(1, rows, 512).astype(np.float32)).astype(mx.bfloat16)
    state = _StateProxy()

    # Cross the boundary with one decode-sized append. This exercises the
    # promotion transition without spending the unit test in 2K tiny kernels.
    state.append_pooled(raw[:, :-1])
    state.append_pooled(raw[:, -1:])

    assert len(state._pooled_q_segments) == math.ceil(rows / _POOL_SEGMENT_ROWS)
    assert {segment[5] for segment in state._pooled_q_segments} == {8}
    assert state._pooled_bf16 is None
    retained_before = state.quant_nbytes()

    materialized = state["pooled"]
    mx.eval(materialized)
    retained_after = state.quant_nbytes()
    assert materialized.shape == raw.shape
    assert retained_after == retained_before
    assert "_pooled_materialized" not in vars(state)
    assert retained_after < materialized.nbytes * 0.60

    cos = _cos(raw, materialized)
    print(
        "  append/compact round-trip "
        f"cos={cos:.4f} retained={retained_after}B bf16={materialized.nbytes}B"
    )
    assert cos >= 0.999


def test_quantized_trim_does_not_materialize_full_pool():
    """Pool trim slices quantized rows and preserves the kept-prefix quality."""
    rows = _POOL_BF16_MAX_BYTES // (512 * 2) + 44
    raw = mx.random.normal((1, rows, 512), dtype=mx.bfloat16)
    state = _StateProxy({"pooled": raw})
    assert state._pooled_bf16 is None

    def _unexpected_dequant(_qpool):
        raise AssertionError("trim must not materialize the BF16 pool")

    with mock.patch.object(pool_quant_cache, "_dequant_pool", _unexpected_dequant):
        state.trim_pooled(17)

    kept = state["pooled"]
    assert kept.shape == (1, rows - 17, 512)
    assert _cos(raw[:, :rows - 17], kept) >= 0.99


def test_short_pool_stays_bf16_without_quantize_or_dequantize():
    """A 361-token ratio-4 pool stays below the 2 MiB hot-tier cap."""
    raw = mx.random.normal((1, 90, 512), dtype=mx.bfloat16)

    def _unexpected_conversion(_value):
        raise AssertionError("short pool must not quantize or dequantize")

    with (
        mock.patch.object(pool_quant_cache, "_quant_pool", _unexpected_conversion),
        mock.patch.object(pool_quant_cache, "_dequant_pool", _unexpected_conversion),
    ):
        cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
        cache.update_pool(raw[:, :89], "compressor_state")
        materialized = cache.update_pool(raw[:, 89:], "compressor_state")
        state = cache.compressor_state

    assert materialized.shape == raw.shape
    assert state._pooled_bf16 is not None
    assert state._pooled_q_segments == []
    assert state.quant_nbytes() == raw.nbytes
    assert raw.nbytes < _POOL_BF16_MAX_BYTES
    assert _cos(raw, materialized) >= 0.9999


def test_short_pool_attention_path_returns_hot_bf16_array_until_promotion(
    monkeypatch,
):
    """The q8 cache must not select tiled attention before q8 exists."""
    monkeypatch.setattr(pool_quant_cache, "_POOL_BF16_MAX_BYTES", 1024)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)

    hot = cache.update_pool_view(
        mx.ones((1, 15, 32), dtype=mx.bfloat16),
        "compressor_state",
    )
    assert isinstance(hot, mx.array)
    assert not getattr(hot, "is_dsv4_quantized_pool_view", False)
    assert cache.compressor_state._pooled_bf16 is hot
    assert cache.compressor_state._pooled_q_segments == []

    promoted = cache.update_pool_view(
        mx.ones((1, 2, 32), dtype=mx.bfloat16),
        "compressor_state",
    )
    assert getattr(promoted, "is_dsv4_quantized_pool_view", False)
    assert cache.compressor_state._pooled_bf16 is None
    assert cache.compressor_state._pooled_q_segments


def test_adaptive_threshold_is_bytes_not_rows():
    """Narrow index pools remain hot longer than wide compressor pools."""
    narrow = mx.random.normal((1, 4096, 128), dtype=mx.bfloat16)
    wide_rows = _POOL_BF16_MAX_BYTES // (512 * 2) + 1
    wide = mx.random.normal((1, wide_rows, 512), dtype=mx.bfloat16)
    narrow_state = _StateProxy({"pooled": narrow})
    wide_state = _StateProxy({"pooled": wide})

    assert narrow.nbytes <= _POOL_BF16_MAX_BYTES
    assert narrow_state._pooled_bf16 is not None
    assert not narrow_state._pooled_q_segments
    assert wide.nbytes > _POOL_BF16_MAX_BYTES
    assert wide_state._pooled_bf16 is None
    assert wide_state._pooled_q_segments


def test_ratio4_ordinary_prompt_stays_hot_and_12k_promotes():
    """4K-token ratio-4 pools stay BF16 while 12K-token pools promote."""
    prompt_4k_pool = mx.zeros((1, 4096 // 4, 512), dtype=mx.bfloat16)
    prompt_12k_pool = mx.zeros((1, 12000 // 4, 512), dtype=mx.bfloat16)
    short_state = _StateProxy({"pooled": prompt_4k_pool})
    long_state = _StateProxy({"pooled": prompt_12k_pool})

    assert prompt_4k_pool.nbytes == 1024 * 1024
    assert short_state._pooled_bf16 is not None
    assert not short_state._pooled_q_segments
    assert prompt_12k_pool.nbytes > _POOL_BF16_MAX_BYTES
    assert long_state._pooled_bf16 is None
    assert long_state._pooled_q_segments


def test_cache_trim_uses_active_storage_tier_without_conversion():
    """The cache's public trim keeps base semantics without pool conversion."""
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    keys = mx.random.normal((1, 4, 16, 64), dtype=mx.bfloat16)
    values = mx.random.normal((1, 4, 16, 64), dtype=mx.bfloat16)
    cache.update_and_fetch(keys, values)
    pooled = mx.random.normal((1, 10, 512), dtype=mx.bfloat16)
    buffer_kv = mx.random.normal((1, 2, 512), dtype=mx.bfloat16)
    buffer_gate = mx.random.normal((1, 2, 512), dtype=mx.bfloat16)
    cache.compressor_state = {
        "buffer_kv": buffer_kv,
        "buffer_gate": buffer_gate,
        "pooled": pooled,
    }
    cache.indexer_state = {
        "buffer_kv": buffer_kv,
        "buffer_gate": buffer_gate,
        "pooled": pooled,
    }

    def _unexpected_dequant(_qpool):
        raise AssertionError("cache.trim must not materialize the BF16 pool")

    with mock.patch.object(pool_quant_cache, "_dequant_pool", _unexpected_dequant):
        cache.trim(4)

    assert cache.compressor_state["buffer_kv"] is None
    assert cache.compressor_state["buffer_gate"] is None
    assert cache.indexer_state["buffer_kv"] is None
    assert cache.indexer_state["buffer_gate"] is None
    assert cache.compressor_state["pooled"].shape == (1, 9, 512)
    assert cache.indexer_state["pooled"].shape == (1, 9, 512)


def test_state_save_load_preserves_quality_and_empty_state_semantics():
    """The standard cache state tuple remains shape- and quality-compatible."""
    cache = PoolQuantizedV4Cache(sliding_window=128)
    keys = mx.random.normal((1, 4, 8, 64), dtype=mx.bfloat16)
    values = mx.random.normal((1, 4, 8, 64), dtype=mx.bfloat16)
    cache.update_and_fetch(keys, values)
    pooled = mx.random.normal((1, 32, 512), dtype=mx.bfloat16)
    cache.compressor_state = {
        "buffer_kv": None,
        "buffer_gate": None,
        "pooled": pooled,
    }

    restored = PoolQuantizedV4Cache(sliding_window=128)
    restored.state = cache.state
    assert _cos(
        cache.compressor_state["pooled"],
        restored.compressor_state["pooled"],
    ) >= 0.999

    empty = mx.zeros((1, 0, 512), dtype=mx.bfloat16)
    restored.compressor_state["pooled"] = empty
    empty_back = restored.compressor_state["pooled"]
    assert empty_back is not None
    assert empty_back.shape == empty.shape
    restored.compressor_state["pooled"] = None
    assert restored.compressor_state["pooled"] is None


def test_storage_state_preserves_q8_codes_without_requantization(monkeypatch):
    """Prompt/L2 snapshots retain encoded segments byte-for-byte."""
    monkeypatch.setattr(pool_quant_cache, "_POOL_BF16_MAX_BYTES", 1)
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    raw = mx.random.normal((1, 129, 64), dtype=mx.bfloat16)
    cache.update_pool(raw, "compressor_state")
    before = cache.compressor_state._pooled_q_segments
    assert before

    storage_state = cache.storage_state

    def _unexpected_conversion(_value):
        raise AssertionError("lossless storage restore must not quantize or dequantize")

    restored = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    with (
        mock.patch.object(pool_quant_cache, "_quant_pool", _unexpected_conversion),
        mock.patch.object(pool_quant_cache, "_dequant_pool", _unexpected_conversion),
    ):
        restored.storage_state = storage_state

    after = restored.compressor_state._pooled_q_segments
    assert len(after) == len(before)
    for source, target in zip(before, after):
        for source_leaf, target_leaf in zip(source[:3], target[:3]):
            assert mx.array_equal(source_leaf, target_leaf).item()
        assert source[3:] == target[3:]


def test_storage_state_preserves_empty_pool_dtype_and_shape():
    cache = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    empty = mx.zeros((1, 0, 512), dtype=mx.bfloat16)
    cache.compressor_state["pooled"] = empty

    restored = PoolQuantizedV4Cache(sliding_window=128, compress_ratio=4)
    restored.storage_state = cache.storage_state
    restored_empty = restored.compressor_state["pooled"]

    assert restored_empty.shape == empty.shape
    assert restored_empty.dtype == empty.dtype


def test_quantized_pool_restores_original_float16_dtype():
    """Mixing float16 SWA with BF16 pools promotes DSV4 attention to float32."""
    raw = mx.random.normal((1, 65, 64), dtype=mx.float16)
    state = _StateProxy()
    state._replace_quantized(raw)

    restored = state["pooled"]

    assert restored.dtype == mx.float16
    assert all(segment[6] == str(mx.float16) for segment in state._pooled_q_segments)
