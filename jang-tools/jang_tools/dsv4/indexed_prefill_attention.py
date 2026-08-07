"""DSV4 heads16 indexed-attention prefill kernel (Metal, env-gated, default ON).

Replaces the CSA prefill dense-mask branch (topk selected pool rows) in
``DeepseekV4Attention.__call__`` with a single custom Metal kernel that
attends each query token to only:

* its sliding-window local KV rows (<= ``window`` = 128), and
* its indexer-selected compressed pool rows (<= ``index_topk`` = 512,
  filtered by causal pool visibility ``(idx + 1) * ratio <= qpos + 1``).

The stock path builds a (B, 1, S, P) bool membership mask from topk, concats
local + full pool into one KV tensor and runs one masked SDPA over all P pool
rows — cost grows with context. This kernel touches at most
``window + index_topk`` rows per query regardless of context length.

Kernel geometry mirrors antirez/ds4 ``heads16_dual`` (222b2cb9f): one
256-thread threadgroup per (query token, 16-head group); 8 simdgroups, each
owning heads ``h0 = g*16 + sg`` and ``h1 = h0 + 8``; each KV row is staged
once into threadgroup memory and consumed by all 16 heads (valid because DSV4
is MQA — one shared KV head, K == V). fp32 online softmax with the attention
sink folded into the initial state (M = sink, S = 1, O = 0), matching
``_dsv4_tiled_pool_attention``. Unlike ds4 (which sorts topk chronologically
and ``break``s), our topk indices are unsorted, so invisible rows are skipped
with ``continue`` — the branch is threadgroup-uniform (idx depends only on
the token), so the staging barriers stay convergent.

Dynamic dims (S, R, P, K, offset, window, ratio) arrive via an int32 params
buffer so pool-shape-dependent math never enters the compiled source: one
compile total, no per-shape recompilation. H = 64 and D = 512 are baked.

Env gate: ``VMLX_DSV4_HEADS16_PREFILL`` (default ON; set 0 to disable). Read per call so a
single process can A/B. First enabled use runs a live self-test against fp32
reference math; on any failure the module permanently disables itself and the
caller falls back to the stock branch.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import mlx.core as mx

_HEAD_DIM = 512
_N_HEADS = 64
_HEADS_PER_TG = 16
_TG_THREADS = 256
_SELF_TEST_MAX_REL = 2.5e-2

_SOURCE = """
    uint t = threadgroup_position_in_grid.x;   // query token in [0, S)
    uint g = threadgroup_position_in_grid.y;   // 16-head group in [0, H/16)
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;  // 0..7
    uint lane = thread_index_in_simdgroup;     // 0..31

    const int S = params[0];
    const int R = params[1];
    const int P = params[2];
    const int K = params[3];
    const int offset = params[4];
    const int window = params[5];
    const int ratio = params[6];
    const float scale = fscale[0];

    const int qpos = offset + int(t);
    const int h0 = int(g) * 16 + int(sg);
    const int h1 = h0 + 8;

    // Per-lane query fragments: unit u covers elements [4*lane + 128*u, +4).
    const device T* q0p = q + ((size_t)h0 * (size_t)S + (size_t)t) * 512;
    const device T* q1p = q + ((size_t)h1 * (size_t)S + (size_t)t) * 512;
    float4 q0[4];
    float4 q1[4];
    for (int u = 0; u < 4; ++u) {
        int e = 4 * int(lane) + 128 * u;
        q0[u] = float4(float(q0p[e]), float(q0p[e + 1]),
                       float(q0p[e + 2]), float(q0p[e + 3]));
        q1[u] = float4(float(q1p[e]), float(q1p[e + 1]),
                       float(q1p[e + 2]), float(q1p[e + 3]));
    }

    // Online softmax state, sink folded in: M = sink, S = 1, O = 0.
    float M0 = sinks[h0];
    float M1 = sinks[h1];
    float S0 = 1.0f;
    float S1 = 1.0f;
    float4 O0[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};
    float4 O1[4] = {float4(0.0f), float4(0.0f), float4(0.0f), float4(0.0f)};

    threadgroup float4 kv_shared[128];

    // ---- local sliding-window rows -------------------------------------
    // Row j sits at absolute position (offset + S) - R + j; visible iff
    // pos <= qpos && pos > qpos - window  =>  contiguous j range.
    int j_lo = int(t) + R - S - (window - 1);
    if (j_lo < 0) j_lo = 0;
    int j_hi = int(t) + R - S;
    if (j_hi > R - 1) j_hi = R - 1;
    for (int j = j_lo; j <= j_hi; ++j) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128) {
            const device T* src = kv + (size_t)j * 512 + (size_t)tid * 4;
            kv_shared[tid] = float4(float(src[0]), float(src[1]),
                                    float(src[2]), float(src[3]));
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
            float c = exp(M0 - mn);
            float p = exp(s0 - mn);
            S0 = S0 * c + p;
            O0[0] = O0[0] * c + p * k0;
            O0[1] = O0[1] * c + p * k1;
            O0[2] = O0[2] * c + p * k2;
            O0[3] = O0[3] * c + p * k3;
            M0 = mn;
        }
        {
            float mn = max(M1, s1);
            float c = exp(M1 - mn);
            float p = exp(s1 - mn);
            S1 = S1 * c + p;
            O1[0] = O1[0] * c + p * k0;
            O1[1] = O1[1] * c + p * k1;
            O1[2] = O1[2] * c + p * k2;
            O1[3] = O1[3] * c + p * k3;
            M1 = mn;
        }
    }

    // ---- indexer-selected compressed pool rows -------------------------
    const device int* trow = topk + (size_t)t * (size_t)K;
    for (int i = 0; i < K; ++i) {
        int idx = trow[i];
        // Threadgroup-uniform skip (idx depends only on t): argpartition
        // fills short visible sets with causally invisible rows; drop them.
        if (idx < 0 || idx >= P) continue;
        if ((idx + 1) * ratio > qpos + 1) continue;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128) {
            const device T* src = pool + (size_t)idx * 512 + (size_t)tid * 4;
            kv_shared[tid] = float4(float(src[0]), float(src[1]),
                                    float(src[2]), float(src[3]));
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
            float c = exp(M0 - mn);
            float p = exp(s0 - mn);
            S0 = S0 * c + p;
            O0[0] = O0[0] * c + p * k0;
            O0[1] = O0[1] * c + p * k1;
            O0[2] = O0[2] * c + p * k2;
            O0[3] = O0[3] * c + p * k3;
            M0 = mn;
        }
        {
            float mn = max(M1, s1);
            float c = exp(M1 - mn);
            float p = exp(s1 - mn);
            S1 = S1 * c + p;
            O1[0] = O1[0] * c + p * k0;
            O1[1] = O1[1] * c + p * k1;
            O1[2] = O1[2] * c + p * k2;
            O1[3] = O1[3] * c + p * k3;
            M1 = mn;
        }
    }

    // ---- normalize + write ---------------------------------------------
    device T* o0p = out + ((size_t)h0 * (size_t)S + (size_t)t) * 512;
    device T* o1p = out + ((size_t)h1 * (size_t)S + (size_t)t) * 512;
    float inv0 = 1.0f / S0;
    float inv1 = 1.0f / S1;
    for (int u = 0; u < 4; ++u) {
        int e = 4 * int(lane) + 128 * u;
        float4 v0 = O0[u] * inv0;
        float4 v1 = O1[u] * inv1;
        o0p[e] = T(v0.x);
        o0p[e + 1] = T(v0.y);
        o0p[e + 2] = T(v0.z);
        o0p[e + 3] = T(v0.w);
        o1p[e] = T(v1.x);
        o1p[e + 1] = T(v1.y);
        o1p[e + 2] = T(v1.z);
        o1p[e + 3] = T(v1.w);
    }
"""

_KERNEL: Optional[Any] = None
_LAST_STATUS: dict[str, Any] = {
    "enabled_env": None,
    "self_test": "not-run",
    "reason": None,
    "calls": 0,
}
_DISABLED = False
_SELF_TESTED = False


def _enabled() -> bool:
    value = os.environ.get("VMLX_DSV4_HEADS16_PREFILL", "1").strip().lower()
    on = value in {"1", "on", "true", "yes"}
    _LAST_STATUS["enabled_env"] = value
    return on


def _get_kernel() -> Any:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = mx.fast.metal_kernel(
            name="vmlx_dsv4_heads16_prefill",
            input_names=["q", "kv", "pool", "topk", "sinks", "fscale", "params"],
            output_names=["out"],
            source=_SOURCE,
        )
    return _KERNEL


def _run_kernel(
    q: mx.array,
    kv2d: mx.array,
    pool2d: mx.array,
    topk2d: mx.array,
    sinks32: mx.array,
    *,
    offset: int,
    window: int,
    ratio: int,
    scale: float,
) -> mx.array:
    heads, seq_len, head_dim = int(q.shape[1]), int(q.shape[2]), int(q.shape[3])
    rows = int(kv2d.shape[0])
    pool_rows = int(pool2d.shape[0])
    k = int(topk2d.shape[1])
    params = mx.array(
        [seq_len, rows, pool_rows, k, int(offset), int(window), int(ratio)],
        dtype=mx.int32,
    )
    fscale = mx.array([float(scale)], dtype=mx.float32)
    kernel = _get_kernel()
    (out,) = kernel(
        inputs=[q, kv2d, pool2d, topk2d, sinks32, fscale, params],
        template=[("T", q.dtype)],
        grid=(_TG_THREADS * seq_len, heads // _HEADS_PER_TG, 1),
        threadgroup=(_TG_THREADS, 1, 1),
        output_shapes=[(1, heads, seq_len, head_dim)],
        output_dtypes=[q.dtype],
    )
    return out


def _reference(
    q: mx.array,
    kv2d: mx.array,
    pool2d: mx.array,
    topk2d: mx.array,
    sinks32: mx.array,
    *,
    offset: int,
    window: int,
    ratio: int,
    scale: float,
) -> mx.array:
    """fp32 replica of the stock dense-mask branch (mask + softmax + sink)."""
    heads, seq_len = int(q.shape[1]), int(q.shape[2])
    rows = int(kv2d.shape[0])
    pool_rows = int(pool2d.shape[0])
    q32 = q.astype(mx.float32)
    keys = mx.concatenate(
        [kv2d.astype(mx.float32), pool2d.astype(mx.float32)], axis=0
    )
    scores = (q32 @ keys.T[None, None]) * scale  # (1, H, S, R+P)

    q_pos = offset + mx.arange(seq_len)
    k_pos = (offset + seq_len) - rows + mx.arange(rows)
    local_vis = (k_pos[None, :] <= q_pos[:, None]) & (
        k_pos[None, :] > (q_pos[:, None] - window)
    )
    k_idx = mx.arange(pool_rows)
    comp_vis = ((k_idx[None, :] + 1) * ratio) <= (q_pos[:, None] + 1)
    selected = (topk2d[:, :, None] == k_idx[None, None, :]).any(axis=1)
    comp_vis = comp_vis & selected
    mask = mx.concatenate([local_vis, comp_vis], axis=-1)[None, None]

    neg = mx.full(scores.shape, -mx.inf, dtype=mx.float32)
    scores = mx.where(mask, scores, neg)
    sink = sinks32.reshape(1, heads, 1, 1)
    m = mx.maximum(scores.max(axis=-1, keepdims=True), sink)
    p = mx.exp(scores - m)
    denom = p.sum(axis=-1, keepdims=True) + mx.exp(sink - m)
    return ((p @ keys[None, None]) / denom).astype(q.dtype)


def _self_test() -> Optional[str]:
    """Randomized small-shape kernel vs fp32 reference check."""
    mx.random.seed(7)
    seq_len, rows, pool_rows, k = 8, 16, 32, 8
    offset, window, ratio = 100, 12, 4
    scale = float(_HEAD_DIM) ** -0.5
    for dtype in (mx.bfloat16, mx.float16):
        q = (mx.random.normal((1, _N_HEADS, seq_len, _HEAD_DIM)) * 0.3).astype(dtype)
        kv2d = (mx.random.normal((rows, _HEAD_DIM)) * 0.3).astype(dtype)
        pool2d = (mx.random.normal((pool_rows, _HEAD_DIM)) * 0.3).astype(dtype)
        # Distinct random rows incl. some causally invisible for early queries.
        perm = mx.argsort(mx.random.uniform(shape=(seq_len, pool_rows)), axis=-1)
        topk2d = perm[:, :k].astype(mx.int32)
        sinks32 = (mx.random.normal((_N_HEADS,)) * 0.5).astype(mx.float32)
        try:
            got = _run_kernel(
                q, kv2d, pool2d, topk2d, sinks32,
                offset=offset, window=window, ratio=ratio, scale=scale,
            ).astype(mx.float32)
            ref = _reference(
                q, kv2d, pool2d, topk2d, sinks32,
                offset=offset, window=window, ratio=ratio, scale=scale,
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


def dsv4_heads16_prefill_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled: Any,
    topk: mx.array,
    *,
    offset: int,
    window: int,
    ratio: int,
    scale: float,
    sinks: mx.array,
) -> Optional[mx.array]:
    """Kernel path for the CSA prefill topk branch; None -> caller uses stock.

    Layout contract (validated, no fallback surprises):
    B == 1, H == 64, D == 512, dtype f16/bf16, plain (non-quantized-view)
    pool, topk shape (1, S, K). Returns (1, H, S, D) matching stock SDPA
    output (pre inverse-rope), or None when disabled/unsupported.
    """
    global _DISABLED, _SELF_TESTED
    if _DISABLED or not _enabled():
        return None
    batch, heads, seq_len, head_dim = map(int, q.shape)
    if batch != 1 or heads != _N_HEADS or head_dim != _HEAD_DIM:
        return None
    if seq_len <= 0:
        return None
    if q.dtype not in (mx.float16, mx.bfloat16):
        return None
    if getattr(pooled, "is_dsv4_quantized_pool_view", False):
        return None
    if topk is None or topk.ndim != 3 or int(topk.shape[0]) != 1:
        return None
    if int(topk.shape[1]) != seq_len or int(topk.shape[2]) <= 0:
        return None
    rows = int(local_kv.shape[2])
    pool_rows = int(pooled.shape[1])
    if rows <= 0 or pool_rows <= 0:
        return None

    if not _SELF_TESTED:
        _SELF_TESTED = True
        err = _self_test()
        if err is not None:
            _DISABLED = True
            _LAST_STATUS["self_test"] = "failed"
            _LAST_STATUS["reason"] = err
            return None
        _LAST_STATUS["self_test"] = "passed"

    kv2d = local_kv.reshape(rows, head_dim)
    pool2d = pooled.reshape(pool_rows, head_dim).astype(q.dtype)
    topk2d = topk.reshape(seq_len, int(topk.shape[2])).astype(mx.int32)
    sinks32 = sinks.astype(mx.float32)
    out = _run_kernel(
        q, kv2d, pool2d, topk2d, sinks32,
        offset=int(offset), window=int(window), ratio=int(ratio),
        scale=float(scale),
    )
    _LAST_STATUS["calls"] += 1
    return out


def dsv4_heads16_status() -> dict[str, Any]:
    return dict(_LAST_STATUS)
