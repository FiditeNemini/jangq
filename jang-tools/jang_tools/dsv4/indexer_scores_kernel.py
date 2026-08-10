"""Fused DSV4 indexer score reduction (Metal, env-gated, default OFF).

The stock reduction computes, for one pool tile::

    out[t, c] = sum_h relu(scale * dot(q[h, t, :], pool[c, :])) * w[t, h]

by materializing a ``[H, L, C]`` fp32 score tensor and collapsing it. That
intermediate is ``H`` (64) times the size of the result, and measuring against a
pure-GEMM floor showed the traffic — not the math — is the cost: ~3.0-3.5 ms per
tile against a 0.91 ms fp32 GEMM of the same shape.

This kernel keeps the per-head scores in registers and writes only ``[L, C]``.

Geometry: one threadgroup per (32-pool-row tile, 8-token tile); 128 threads = 4
simdgroups, each owning 8 of the 32 rows. The pool tile is staged once as half
and reused by every head; per head the query tile is staged, an 8x8 simdgroup
matmul over ``D/8`` steps produces that simdgroup's dot block, and each thread
folds ``relu(scale * s) * w[t, h]`` for its two cells into an fp32 register.
Accumulation stays fp32 so this is far closer to the stock values than a
reduced-precision reduction (which changed top-k selection and was rejected).

Causal pool visibility is intentionally NOT applied here: the caller already
masks and top-ks the ``[L, C]`` scores, and keeping that in Python preserves
exact parity with the stock path.

Env gate ``VMLX_DSV4_INDEXER_KERNEL`` (default OFF). First enabled use runs a
randomized self-test against fp32 reference math; any mismatch permanently
disables the kernel for the process and the caller falls back to stock.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import mlx.core as mx

_TM = 8            # query tokens per tile
_TN = 32           # pool rows per tile
_TS = 8            # simdgroup matrix edge
_HEAD_DIM = 128
_TG_THREADS = 128  # 4 simdgroups
_SELF_TEST_MAX_REL = 2.5e-2
# Gate on total tile work (tokens x pool rows), not sequence length: the live
# engine calls this with 512-token chunks and a pool that grows with context.
# Measured at L=512: C=256 (131k) is 0.80x, C=640 (327k) is 1.31x, and it keeps
# improving to 1.39x at C=4608. Below the threshold the stock op chain is
# already cheap enough that the kernel's fixed cost dominates. Decode (L=1)
# never reaches it, so this stays a prefill-only path.
_MIN_TILE_WORK = 262144

_HEADER = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
"""

_SOURCE = """
    uint rt = threadgroup_position_in_grid.x;      // pool-row tile
    uint tt = threadgroup_position_in_grid.y;      // query-token tile
    uint tid = thread_position_in_threadgroup.x;   // 0..127
    uint sg = simdgroup_index_in_threadgroup;      // 0..3
    uint lane = thread_index_in_simdgroup;         // 0..31

    const int L = params[0];
    const int C = params[1];
    const int H = params[2];
    const float scale = fscale[0];

    const uint TM = 8u;
    const uint TN = 32u;
    const uint TS = 8u;
    const uint D  = 128u;

    const uint row_base = rt * TN;
    const uint tok_base = tt * TM;

    threadgroup half qtg[8 * 128];
    threadgroup half ktg[32 * 128];
    threadgroup float dotb[8 * 32];

    // Stage the pool tile once; every head reuses it.
    for (uint i = tid; i < TN * D; i += 128u) {
        uint rc = i / D;
        uint d = i - rc * D;
        uint row = row_base + rc;
        half v = half(0.0f);
        if (row < uint(C)) {
            v = half(float(pool[(size_t)row * (size_t)D + (size_t)d]));
        }
        ktg[i] = v;
    }

    // Each thread owns two output cells of this simdgroup's 8x8 block.
    uint cell0 = lane;
    uint cell1 = lane + 32u;
    uint trow0 = cell0 >> 3;
    uint sub0 = cell0 & 7u;
    uint trow1 = cell1 >> 3;
    uint sub1 = cell1 & 7u;
    uint col0 = sg * TS + sub0;
    uint col1 = sg * TS + sub1;
    uint tok0 = tok_base + trow0;
    uint tok1 = tok_base + trow1;
    uint row0 = row_base + col0;
    uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint h = 0; h < uint(H); ++h) {
        for (uint i = tid; i < TM * D; i += 128u) {
            uint tr = i / D;
            uint d = i - tr * D;
            uint tok = tok_base + tr;
            half v = half(0.0f);
            if (tok < uint(L)) {
                v = half(float(q[((size_t)h * (size_t)L + (size_t)tok)
                                 * (size_t)D + (size_t)d]));
            }
            qtg[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D / TS; ++db) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db * TS, D, 0, false);
            simdgroup_load(mk, ktg + (size_t)(sg * TS) * (size_t)D + db * TS,
                           D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }
        simdgroup_store(mdot, dotb + sg * TS, TN, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tok0 < uint(L) && row0 < uint(C)) {
            float s = dotb[trow0 * TN + col0];
            acc0 += max(s * scale, 0.0f)
                    * w[(size_t)tok0 * (size_t)H + (size_t)h];
        }
        if (tok1 < uint(L) && row1 < uint(C)) {
            float s = dotb[trow1 * TN + col1];
            acc1 += max(s * scale, 0.0f)
                    * w[(size_t)tok1 * (size_t)H + (size_t)h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tok0 < uint(L) && row0 < uint(C)) {
        out[(size_t)tok0 * (size_t)C + (size_t)row0] = acc0;
    }
    if (tok1 < uint(L) && row1 < uint(C)) {
        out[(size_t)tok1 * (size_t)C + (size_t)row1] = acc1;
    }
"""

_KERNEL = None
_DISABLED = False
_SELF_TESTED = False
_LOGGER = logging.getLogger(__name__)
_LAST_STATUS: dict[str, Any] = {"self_test": None, "reason": None, "calls": 0}


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_INDEXER_KERNEL", "0").strip().lower()
    return value in {"1", "on", "true", "yes"}


def _get_kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="vmlx_dsv4_indexer_scores",
            input_names=["q", "pool", "w", "fscale", "params"],
            output_names=["out"],
            header=_HEADER,
            source=_SOURCE,
        )
    return _KERNEL


def _run_kernel(q3: mx.array, pool2: mx.array, w2: mx.array, scale: float) -> mx.array:
    heads, seq_len, _ = map(int, q3.shape)
    rows = int(pool2.shape[0])
    params = mx.array([seq_len, rows, heads], dtype=mx.int32)
    fscale = mx.array([float(scale)], dtype=mx.float32)
    row_tiles = (rows + _TN - 1) // _TN
    tok_tiles = (seq_len + _TM - 1) // _TM
    (out,) = _get_kernel()(
        inputs=[q3, pool2, w2, fscale, params],
        grid=(_TG_THREADS * row_tiles, tok_tiles, 1),
        threadgroup=(_TG_THREADS, 1, 1),
        output_shapes=[(seq_len, rows)],
        output_dtypes=[mx.float32],
    )
    return out


def _reference(q3: mx.array, pool2: mx.array, w2: mx.array, scale: float) -> mx.array:
    """fp32 replica of the stock reduction."""
    scores = q3.astype(mx.float32) @ pool2.astype(mx.float32).T  # [H, L, C]
    scores = mx.maximum(scores, 0) * float(scale)
    weights = w2.astype(mx.float32).T[..., None]                 # [H, L, 1]
    return (scores * weights).sum(axis=0)                        # [L, C]


def _self_test() -> Optional[str]:
    mx.random.seed(13)
    for seq_len, rows, heads in ((8, 32, 4), (13, 40, 8), (64, 96, 16)):
        q3 = (mx.random.normal((heads, seq_len, _HEAD_DIM)) * 0.35)
        pool2 = (mx.random.normal((rows, _HEAD_DIM)) * 0.35)
        w2 = (mx.random.normal((seq_len, heads)) * 0.35).astype(mx.float32)
        scale = float(_HEAD_DIM) ** -0.5
        try:
            got = _run_kernel(q3.astype(mx.bfloat16), pool2.astype(mx.bfloat16),
                              w2, scale)
            ref = _reference(q3.astype(mx.bfloat16), pool2.astype(mx.bfloat16),
                             w2, scale)
            mx.eval(got, ref)
        except Exception as err:
            return f"self-test execution failed ({seq_len}x{rows}x{heads}): {err}"
        denom = max(float(mx.abs(ref).max()), 1e-6)
        rel = float(mx.abs(got - ref).max()) / denom
        if rel > _SELF_TEST_MAX_REL:
            return (f"self-test rel diff {rel:.3e} > {_SELF_TEST_MAX_REL:.1e} "
                    f"({seq_len}x{rows}x{heads})")
        _LAST_STATUS[f"rel_{seq_len}x{rows}x{heads}"] = rel
    return None


def fused_indexer_scores(
    q: mx.array,
    pool_tile: mx.array,
    head_weights: mx.array,
    scale: float,
) -> Optional[mx.array]:
    """Head-reduced scores ``[1, L, C]``; ``None`` means caller uses stock.

    ``q`` is ``[1, H, L, 128]``, ``pool_tile`` ``[1, C, 128]``,
    ``head_weights`` ``[1, L, H]``.
    """
    global _DISABLED, _SELF_TESTED
    if _DISABLED or not _enabled():
        return None

    def _decline(reason: str):
        if not _LAST_STATUS.get("declined"):
            _LAST_STATUS["declined"] = reason
            _LOGGER.info(
                "DSV4 fused indexer kernel declined (%s); q=%s pool=%s w=%s",
                reason,
                tuple(q.shape) if hasattr(q, "shape") else None,
                tuple(pool_tile.shape) if hasattr(pool_tile, "shape") else None,
                tuple(head_weights.shape) if hasattr(head_weights, "shape") else None,
            )
        return None

    if q.ndim != 4 or int(q.shape[0]) != 1 or int(q.shape[3]) != _HEAD_DIM:
        return _decline("q layout")
    if pool_tile.ndim != 3 or int(pool_tile.shape[0]) != 1:
        return _decline("pool layout")
    if int(pool_tile.shape[2]) != _HEAD_DIM:
        return _decline("pool head_dim")
    if head_weights.ndim != 3 or int(head_weights.shape[0]) != 1:
        return _decline("weights layout")
    heads, seq_len = int(q.shape[1]), int(q.shape[2])
    rows = int(pool_tile.shape[1])
    if rows <= 0 or heads <= 0:
        return _decline("empty tile")
    if seq_len * rows < _MIN_TILE_WORK:
        return _decline("tile work below kernel threshold")
    if int(head_weights.shape[1]) != seq_len or int(head_weights.shape[2]) != heads:
        return _decline("weights shape mismatch")

    if not _SELF_TESTED:
        _SELF_TESTED = True
        err = _self_test()
        if err is not None:
            _DISABLED = True
            _LAST_STATUS["self_test"] = "failed"
            _LAST_STATUS["reason"] = err
            _LOGGER.warning(
                "DSV4 fused indexer kernel self-test FAILED (%s); using the "
                "stock reduction for this process",
                err,
            )
            return None
        _LAST_STATUS["self_test"] = "passed"
        _LOGGER.info("DSV4 fused indexer Metal kernel ACTIVE (self-test passed)")

    out = _run_kernel(
        q[0],
        pool_tile[0],
        head_weights[0].astype(mx.float32),
        float(scale),
    )
    _LAST_STATUS["calls"] += 1
    return out[None]


def fused_indexer_status() -> dict[str, Any]:
    return dict(_LAST_STATUS)
