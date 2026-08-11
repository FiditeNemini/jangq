"""DSV4 fused indexed-decode attention kernel (Metal, env-gated, default OFF).

Replaces the CSA decode branch (``L == 1`` with an active indexer top-k
selection) in ``DeepseekV4Attention.__call__``. The stock branch broadcasts
the pool to ``(B, 1, L, P, D)``, gathers the ~2048 selected rows with
``take_along_axis`` into a materialized KV tensor, concatenates it with the
local sliding-window rows and runs a full SDPA — several kernel launches
plus ~2 MB of gathered KV traffic per layer per token.

This module fuses that into a split-K indexed-attention pass that reads K/V
directly through the selected-index list (no materialized gather):

* pass 1 (``vmlx_dsv4_indexed_decode_partial``): one 256-thread threadgroup
  per (16-head group, split-K chunk). The unified row space is
  ``n in [0, R + K)`` — local window rows for ``n < R``, pool rows addressed
  via ``topk[n - R]`` otherwise. Each row is staged once into threadgroup
  memory and consumed by all 16 heads (valid because DSV4 is MQA — one
  shared KV latent, K == V). 8 simdgroups each own heads ``h0 = g*16 + sg``
  and ``h1 = h0 + 8``, matching the heads16 prefill kernel conventions.
  Each chunk keeps an fp32 online-softmax partial ``(M, S, O[512])`` with
  ``M`` initialized to ``-inf`` (the attention sink is folded at merge).
* pass 2 (``vmlx_dsv4_indexed_decode_merge``): 128 threads per head combine
  the per-chunk partials with a numerically stable log-sum-exp merge —
  ``m = max(sink, max_c M_c)``, ``denom = exp(sink - m) + sum_c S_c
  exp(M_c - m)``, ``out = sum_c O_c exp(M_c - m) / denom`` — and write the
  bf16/f16 output row.

Dynamic dims (R, K, P, chunk rows / chunk count) arrive via int32 params
buffers so shape changes never recompile: one compile per pass total.
H = 64 and D = 512 are baked, exactly like the heads16 prefill kernel.

Env gate: ``VMLX_DSV4_INDEXED_DECODE`` (default OFF; set 1 to enable for
A/B). Read per call so a single process can A/B. First enabled use runs a
live self-test against fp32 reference math; on any failure the module
permanently disables itself and the caller falls back to the stock branch.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import mlx.core as mx

_HEAD_DIM = 512
_N_HEADS = 64
_HEADS_PER_TG = 16
_TG_THREADS = 256
_MERGE_THREADS = 128
_SPLIT_K_ROWS = 256
_SELF_TEST_MAX_REL = 2.5e-2

_PARTIAL_SOURCE = """
    uint g = threadgroup_position_in_grid.y;   // 16-head group in [0, H/16)
    uint c = threadgroup_position_in_grid.z;   // split-K chunk
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;  // 0..7
    uint lane = thread_index_in_simdgroup;     // 0..31

    const int R = params[0];
    const int K = params[1];
    const int P = params[2];
    const int CHUNK = params[3];
    const float scale = fscale[0];

    const int N = R + K;
    int n_lo = int(c) * CHUNK;
    int n_hi = n_lo + CHUNK;
    if (n_hi > N) n_hi = N;

    const int h0 = int(g) * 16 + int(sg);
    const int h1 = h0 + 8;

    // Per-lane query fragments: unit u covers elements [4*lane + 128*u, +4).
    const device T* q0p = q + (size_t)h0 * 512;
    const device T* q1p = q + (size_t)h1 * 512;
    float4 q0[4];
    float4 q1[4];
    for (int u = 0; u < 4; ++u) {
        int e = 4 * int(lane) + 128 * u;
        q0[u] = float4(float(q0p[e]), float(q0p[e + 1]),
                       float(q0p[e + 2]), float(q0p[e + 3]));
        q1[u] = float4(float(q1p[e]), float(q1p[e + 1]),
                       float(q1p[e + 2]), float(q1p[e + 3]));
    }

    // Partial online-softmax state for this chunk. The attention sink is
    // folded at merge time, so partials start empty (M = -inf, S = 0).
    float M0 = -INFINITY;
    float M1 = -INFINITY;
    float S0 = 0.0f;
    float S1 = 0.0f;
    float4 O0[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    float4 O1[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

    threadgroup float4 kv_shared[128];

    for (int n = n_lo; n < n_hi; ++n) {
        const device T* src;
        if (n < R) {
            src = kv + (size_t)n * 512;
        } else {
            int sel = topk[n - R];
            // Threadgroup-uniform skip (sel depends only on n): defensive
            // bound check; the stock gather assumes valid indices.
            if (sel < 0 || sel >= P) continue;
            src = pool + (size_t)sel * 512;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128) {
            const device T* s = src + (size_t)tid * 4;
            kv_shared[tid] = float4(float(s[0]), float(s[1]),
                                    float(s[2]), float(s[3]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float4 k0 = kv_shared[lane];
        float4 k1 = kv_shared[lane + 32];
        float4 k2 = kv_shared[lane + 64];
        float4 k3 = kv_shared[lane + 96];
        float s0 = dot(q0[0], k0) + dot(q0[1], k1)
                 + dot(q0[2], k2) + dot(q0[3], k3);
        float s1 = dot(q1[0], k0) + dot(q1[1], k1)
                 + dot(q1[2], k2) + dot(q1[3], k3);
        s0 = simd_sum(s0) * scale;
        s1 = simd_sum(s1) * scale;
        {
            float mn = max(M0, s0);
            float cs = exp(M0 - mn);
            float p = exp(s0 - mn);
            S0 = S0 * cs + p;
            O0[0] = O0[0] * cs + p * k0;
            O0[1] = O0[1] * cs + p * k1;
            O0[2] = O0[2] * cs + p * k2;
            O0[3] = O0[3] * cs + p * k3;
            M0 = mn;
        }
        {
            float mn = max(M1, s1);
            float cs = exp(M1 - mn);
            float p = exp(s1 - mn);
            S1 = S1 * cs + p;
            O1[0] = O1[0] * cs + p * k0;
            O1[1] = O1[1] * cs + p * k1;
            O1[2] = O1[2] * cs + p * k2;
            O1[3] = O1[3] * cs + p * k3;
            M1 = mn;
        }
    }

    // ---- write per-chunk partials (fp32, unnormalized) -----------------
    size_t b0 = (size_t)c * 64 + (size_t)h0;
    size_t b1 = (size_t)c * 64 + (size_t)h1;
    if (lane == 0) {
        pms[b0 * 2] = M0;
        pms[b0 * 2 + 1] = S0;
        pms[b1 * 2] = M1;
        pms[b1 * 2 + 1] = S1;
    }
    device float* o0p = po + b0 * 512;
    device float* o1p = po + b1 * 512;
    for (int u = 0; u < 4; ++u) {
        int e = 4 * int(lane) + 128 * u;
        o0p[e] = O0[u].x;
        o0p[e + 1] = O0[u].y;
        o0p[e + 2] = O0[u].z;
        o0p[e + 3] = O0[u].w;
        o1p[e] = O1[u].x;
        o1p[e + 1] = O1[u].y;
        o1p[e + 2] = O1[u].z;
        o1p[e + 3] = O1[u].w;
    }
"""

_MERGE_SOURCE = """
    uint h = threadgroup_position_in_grid.y;      // head in [0, 64)
    uint tid = thread_position_in_threadgroup.x;  // 0..127, 4 elements each

    const int C = params[0];

    float sink = sinks[h];
    float m = sink;
    for (int c = 0; c < C; ++c) {
        m = max(m, pms[((size_t)c * 64 + h) * 2]);
    }
    // Sink contributes exp(sink - m) to the denominator with a zero value.
    float denom = exp(sink - m);
    for (int c = 0; c < C; ++c) {
        size_t b = ((size_t)c * 64 + h) * 2;
        denom += pms[b + 1] * exp(pms[b] - m);
    }
    float4 acc = float4(0.0f);
    for (int c = 0; c < C; ++c) {
        size_t b = (size_t)c * 64 + h;
        float w = exp(pms[b * 2] - m);
        const device float* op = po + b * 512 + (size_t)tid * 4;
        acc += float4(op[0], op[1], op[2], op[3]) * w;
    }
    acc = acc * (1.0f / denom);
    device T* outp = out + (size_t)h * 512 + (size_t)tid * 4;
    outp[0] = T(acc.x);
    outp[1] = T(acc.y);
    outp[2] = T(acc.z);
    outp[3] = T(acc.w);
"""

_PARTIAL_KERNEL: Optional[Any] = None
_MERGE_KERNEL: Optional[Any] = None
_LAST_STATUS: dict[str, Any] = {
    "enabled_env": None,
    "self_test": "not-run",
    "reason": None,
    "calls": 0,
}
_LOGGER = logging.getLogger(__name__)

_DISABLED = False
_SELF_TESTED = False


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_INDEXED_DECODE", "0").strip().lower()
    on = value in {"1", "on", "true", "yes"}
    _LAST_STATUS["enabled_env"] = value
    return on


def _get_partial_kernel() -> Any:
    global _PARTIAL_KERNEL
    if _PARTIAL_KERNEL is None:
        _PARTIAL_KERNEL = mx.fast.metal_kernel(
            name="vmlx_dsv4_indexed_decode_partial",
            input_names=["q", "kv", "pool", "topk", "fscale", "params"],
            output_names=["pms", "po"],
            source=_PARTIAL_SOURCE,
        )
    return _PARTIAL_KERNEL


def _get_merge_kernel() -> Any:
    global _MERGE_KERNEL
    if _MERGE_KERNEL is None:
        _MERGE_KERNEL = mx.fast.metal_kernel(
            name="vmlx_dsv4_indexed_decode_merge",
            input_names=["pms", "po", "sinks", "params"],
            output_names=["out"],
            source=_MERGE_SOURCE,
        )
    return _MERGE_KERNEL


def _run_kernel(
    q: mx.array,
    kv2d: mx.array,
    pool2d: mx.array,
    topk1d: mx.array,
    sinks32: mx.array,
    *,
    scale: float,
) -> mx.array:
    heads, head_dim = int(q.shape[1]), int(q.shape[3])
    rows = int(kv2d.shape[0])
    pool_rows = int(pool2d.shape[0])
    k = int(topk1d.shape[0])
    if rows == 0:
        # MLX passes size-0 arrays in the constant address space, which
        # breaks the kernel's device-pointer staging. Params still carry
        # R=0, so the placeholder row is never read.
        kv2d = mx.zeros((1, head_dim), dtype=kv2d.dtype)
    total = rows + k
    chunks = max(1, -(-total // _SPLIT_K_ROWS))
    params = mx.array([rows, k, pool_rows, _SPLIT_K_ROWS], dtype=mx.int32)
    fscale = mx.array([float(scale)], dtype=mx.float32)
    partial = _get_partial_kernel()
    pms, po = partial(
        inputs=[q, kv2d, pool2d, topk1d, fscale, params],
        template=[("T", q.dtype)],
        grid=(_TG_THREADS, heads // _HEADS_PER_TG, chunks),
        threadgroup=(_TG_THREADS, 1, 1),
        output_shapes=[(chunks, heads, 2), (chunks, heads, head_dim)],
        output_dtypes=[mx.float32, mx.float32],
    )
    merge = _get_merge_kernel()
    merge_params = mx.array([chunks], dtype=mx.int32)
    (out,) = merge(
        inputs=[pms, po, sinks32, merge_params],
        template=[("T", q.dtype)],
        grid=(_MERGE_THREADS, heads, 1),
        threadgroup=(_MERGE_THREADS, 1, 1),
        output_shapes=[(1, heads, 1, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out


def _reference(
    q: mx.array,
    kv2d: mx.array,
    pool2d: mx.array,
    topk1d: mx.array,
    sinks32: mx.array,
    *,
    scale: float,
) -> mx.array:
    """fp32 replica of the stock decode branch (gather + SDPA + sink)."""
    heads = int(q.shape[1])
    q32 = q.astype(mx.float32)
    gathered = pool2d.astype(mx.float32)[topk1d]
    keys = mx.concatenate([kv2d.astype(mx.float32), gathered], axis=0)
    scores = (q32 @ keys.T[None, None]) * scale  # (1, H, 1, R+K)
    sink = sinks32.reshape(1, heads, 1, 1)
    m = mx.maximum(scores.max(axis=-1, keepdims=True), sink)
    p = mx.exp(scores - m)
    denom = p.sum(axis=-1, keepdims=True) + mx.exp(sink - m)
    return ((p @ keys[None, None]) / denom).astype(q.dtype)


def _self_test() -> Optional[str]:
    """Randomized small-shape kernel vs fp32 reference check.

    Covers the split-K merge (N > _SPLIT_K_ROWS => multiple chunks) with a
    selection length that is not a multiple of the split-K tile.
    """
    mx.random.seed(7)
    rows, pool_rows, k = 16, 600, 500  # N = 516 -> 3 chunks (2 partial)
    scale = float(_HEAD_DIM) ** -0.5
    for dtype in (mx.bfloat16, mx.float16):
        q = (mx.random.normal((1, _N_HEADS, 1, _HEAD_DIM)) * 0.3).astype(dtype)
        kv2d = (mx.random.normal((rows, _HEAD_DIM)) * 0.3).astype(dtype)
        pool2d = (mx.random.normal((pool_rows, _HEAD_DIM)) * 0.3).astype(dtype)
        perm = mx.argsort(mx.random.uniform(shape=(pool_rows,)))
        topk1d = perm[:k].astype(mx.int32)
        sinks32 = (mx.random.normal((_N_HEADS,)) * 0.5).astype(mx.float32)
        try:
            got = _run_kernel(
                q, kv2d, pool2d, topk1d, sinks32, scale=scale
            ).astype(mx.float32)
            ref = _reference(
                q, kv2d, pool2d, topk1d, sinks32, scale=scale
            ).astype(mx.float32)
            mx.eval(got, ref)
        except Exception as err:  # compile/dispatch failure
            return f"self-test execution failed ({dtype}): {err}"
        denom = max(float(mx.abs(ref).max()), 1e-6)
        rel = float(mx.abs(got - ref).max()) / denom
        if rel > _SELF_TEST_MAX_REL:
            return f"self-test rel diff {rel:.3e} > {_SELF_TEST_MAX_REL:.1e} ({dtype})"
        _LAST_STATUS[f"self_test_rel_{dtype}"] = rel
    return None


def dsv4_indexed_decode_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: Any,
    topk: mx.array,
    *,
    scale: float,
    sinks: mx.array,
) -> Optional[mx.array]:
    """Kernel path for the CSA decode topk branch; None -> caller uses stock.

    Layout contract (validated, no fallback surprises):
    B == 1, L == 1, H == 64, D == 512, dtype f16/bf16, plain
    (non-quantized-view) pooled of shape (1, P, D), topk shape (1, 1, K).
    Returns (1, H, 1, D) matching the stock gather + SDPA-with-sinks output
    (pre inverse-rope), or None when disabled/unsupported.
    """
    global _DISABLED, _SELF_TESTED
    if _DISABLED or not _enabled():
        return None
    if q.ndim != 4:
        return None
    batch, heads, seq_len, head_dim = map(int, q.shape)
    if batch != 1 or heads != _N_HEADS or seq_len != 1 or head_dim != _HEAD_DIM:
        return None
    if q.dtype not in (mx.float16, mx.bfloat16):
        return None
    if getattr(pooled, "is_dsv4_quantized_pool_view", False):
        return None
    if getattr(pooled, "ndim", 0) != 3 or int(pooled.shape[0]) != 1:
        return None
    if topk is None or topk.ndim != 3:
        return None
    if int(topk.shape[0]) != 1 or int(topk.shape[1]) != 1:
        return None
    selected = int(topk.shape[2])
    if selected <= 0:
        return None
    if local_kv.ndim != 4 or int(local_kv.shape[0]) != 1:
        return None
    if int(local_kv.shape[1]) != 1 or int(local_kv.shape[3]) != _HEAD_DIM:
        return None
    rows = int(local_kv.shape[2])
    pool_rows = int(pooled.shape[1])
    if pool_rows <= 0 or rows < 0:
        return None

    if not _SELF_TESTED:
        _SELF_TESTED = True
        err = _self_test()
        if err is not None:
            _DISABLED = True
            _LAST_STATUS["self_test"] = "failed"
            _LAST_STATUS["reason"] = err
            _LOGGER.warning(
                "DSV4 fused indexed-decode kernel self-test FAILED (%s); "
                "using stock indexed decode for this process",
                err,
            )
            return None
        _LAST_STATUS["self_test"] = "passed"
        _LOGGER.info(
            "DSV4 fused indexed-decode Metal kernel ACTIVE "
            "(self-test passed, rel_bf16=%s)",
            _LAST_STATUS.get("self_test_rel_bfloat16", "n/a"),
        )

    kv2d = local_kv.reshape(rows, head_dim)
    pool2d = pooled.reshape(pool_rows, head_dim).astype(q.dtype)
    topk1d = topk.reshape(selected).astype(mx.int32)
    sinks32 = sinks.astype(mx.float32)
    out = _run_kernel(q, kv2d, pool2d, topk1d, sinks32, scale=float(scale))
    _LAST_STATUS["calls"] += 1
    return out


def dsv4_indexed_decode_status() -> dict[str, Any]:
    return dict(_LAST_STATUS)
