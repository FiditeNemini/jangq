"""Typed block-delta contract for DeepSeek-V4 native prefix caches.

The DSV4 cache is not a plain positional KV tensor.  Each layer owns a
bounded local rotating window plus cumulative compressor/indexer pools.  A
durable prefix cache therefore stores immutable pool-row deltas for every
token block and periodic exact anchors for the path-dependent local/buffer
state.  Consumers restore only through the latest validated anchor and
re-prefill the matched tail after that checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DSV4_BLOCK_DELTA_SCHEMA = "deepseek_v4_block_delta_v1"
DSV4_POOL_DELTA_SCHEMA = "deepseek_v4_pool_delta_v1"
DSV4_BLOCK_SIZE = 256
DSV4_ANCHOR_INTERVAL_BLOCKS = 8


@dataclass(frozen=True)
class DSV4AnchorRestore:
    """Result of restoring the latest safe DSV4 checkpoint."""

    cache: Any
    checkpoint_tokens: int
    replayed_tokens: int


def canonical_rotating_window(cache: Any, *, expected_offset: int | None = None):
    """Detach one RotatingKVCache in temporal, bounded SWA order.

    ``mlx_lm.RotatingKVCache._update_concat`` may retain
    ``max_size + chunk - 1`` physical rows after a large prefill chunk. The
    physical buffer and ring pointer are therefore not a durable checkpoint.
    This helper keeps the configured prefix ``keep`` rows plus the newest
    window tail, materializes independent arrays, and stamps the insertion
    pointer at the canonical temporal end.
    """
    import mlx.core as mx

    keys = getattr(cache, "keys", None)
    values = getattr(cache, "values", None)
    if keys is None or values is None:
        raise ValueError("DSV4 rotating anchor is missing local K/V")
    if len(keys.shape) < 3 or tuple(keys.shape) != tuple(values.shape):
        raise ValueError("DSV4 rotating anchor has incompatible K/V geometry")
    max_size = int(getattr(cache, "max_size", 0) or 0)
    keep = int(getattr(cache, "keep", 0) or 0)
    offset = int(getattr(cache, "offset", 0) or 0)
    idx = int(getattr(cache, "_idx", 0) or 0)
    if expected_offset is not None and offset != int(expected_offset):
        raise ValueError(
            "DSV4 rotating anchor offset does not match token boundary: "
            f"offset={offset} expected={expected_offset}"
        )
    if max_size <= 0 or keep < 0 or keep > max_size:
        raise ValueError(
            f"invalid DSV4 rotating geometry max_size={max_size} keep={keep}"
        )
    physical_rows = int(keys.shape[-2])
    if int(values.shape[-2]) != physical_rows or idx < 0 or idx > physical_rows:
        raise ValueError("invalid DSV4 rotating physical state")

    def _slice(value, start=None, stop=None):
        parts = [slice(None)] * len(value.shape)
        parts[-2] = slice(start, stop)
        return value[tuple(parts)]

    def _temporal(value):
        if idx == physical_rows:
            return value
        if idx < offset:
            return mx.concatenate(
                [
                    _slice(value, 0, keep),
                    _slice(value, idx, None),
                    _slice(value, keep, idx),
                ],
                axis=-2,
            )
        return _slice(value, 0, idx)

    temporal_keys = _temporal(keys)
    temporal_values = _temporal(values)
    target_rows = min(offset, max_size)
    if int(temporal_keys.shape[-2]) < target_rows:
        raise ValueError(
            "DSV4 rotating anchor is shorter than its semantic window: "
            f"temporal={temporal_keys.shape[-2]} required={target_rows}"
        )
    if int(temporal_keys.shape[-2]) > target_rows:
        tail_rows = target_rows - min(keep, target_rows)

        def _bounded(value):
            if keep and target_rows > keep:
                return mx.concatenate(
                    [_slice(value, 0, keep), _slice(value, -tail_rows, None)],
                    axis=-2,
                )
            return _slice(value, -target_rows, None) if target_rows else _slice(value, 0, 0)

        temporal_keys = _bounded(temporal_keys)
        temporal_values = _bounded(temporal_values)
    detached_keys = temporal_keys + mx.zeros_like(temporal_keys)
    detached_values = temporal_values + mx.zeros_like(temporal_values)
    mx.eval(detached_keys, detached_values)
    return (
        detached_keys,
        detached_values,
        max_size,
        keep,
        offset,
        target_rows,
    )
