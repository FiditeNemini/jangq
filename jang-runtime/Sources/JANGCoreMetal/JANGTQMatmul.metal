//
// JANGTQ Metal kernels — codebook-quantized matmul + Hadamard butterfly.
// Created by Jinho Jang (eric@jangq.ai).
//
// Three kernels matching the Python jang_tools.turboquant package:
//
//   1. jangtq_hadamard_multiblock   — P3, single-dispatch multi-block Hadamard
//                                     for non-pow2 dims (e.g. 3072=2048+1024).
//
//   2. jangtq_fused_gate_up_swiglu  — P2+P7+P8+P12 → P17 OPT=10. Computes
//                                     SiLU(gate(x_rot)) * up(x_rot) per (token, k)
//                                     in one Metal dispatch with codebook lookup.
//
//   3. jangtq_gather_tq_matmul      — P9+P12 → P17 OPT=20. Per-row mode
//                                     (down_proj path): each (token, k) gets
//                                     its own input row.
//
// Conventions (must match Python kernels exactly):
//
//   packed[expert, out_idx, pack_idx] : uint32 holding 16 × 2-bit codebook indices
//                                       LSB-first. pack_idx covers in_features/16.
//   norms[expert, out_idx]            : half — per-row L2 norm
//   codebook[c]                       : float32 — 4 entries (2-bit Lloyd-Max)
//   signs[i]                          : float32 — ±1 random sign for Hadamard
//
// The matmul math (per output element):
//
//   y[token, k, out] = norms[expert, out] *
//                      Σᵢ x_rot[token, i] * codebook[ unpack(packed[expert, out, i]) ]
//
//   where x_rot = H @ (signs * x), H = randomized Hadamard, applied once.
//
// CRITICAL: the codebook is keyed on (in_features, bits). gate/up uses the
// hidden codebook, down_proj uses the intermediate codebook. They differ by
// exactly sqrt(inter / hidden) and MUST be passed as separate kernel args.
// Mixing them silently scales outputs by 1/sqrt(2) and the model emits EOS
// at 41-57 tokens. See research/JANGTQ-REFERENCE.md §10.
//

#include <metal_stdlib>
using namespace metal;

// ============================================================================
//  P3: hadamard_multiblock — single-dispatch butterfly for non-pow2 dims
// ============================================================================
//
// Layout: meta = [total_d, n_blocks, d_b0, log_b0, d_b1, log_b1, ...]
//
// Decomposes a non-pow2 dim into a sum of pow2 blocks (e.g.
// 3072 = 2048 + 1024). All blocks processed in ONE kernel launch with
// threadgroup barriers between stages.
//
// Threadgroup shmem: up to 4096 floats (16 KB). Each thread handles up
// to 4 elements per block via the `ept` (elements per thread) loop.
//
kernel void jangtq_hadamard_multiblock(
    device const half *x          [[buffer(0)]],
    device const float *signs     [[buffer(1)]],
    device const uint  *meta      [[buffer(2)]],
    device float *out             [[buffer(3)]],
    uint2 gid                     [[thread_position_in_grid]],
    uint2 tid_local               [[thread_position_in_threadgroup]],
    uint2 threads_per_tg          [[threads_per_threadgroup]])
{
    uint batch_idx = gid.y;
    uint tid = tid_local.x;
    uint threads_per_tg_x = threads_per_tg.x;
#define threads_per_tg threads_per_tg_x

    uint total_d  = meta[0];
    uint n_blocks = meta[1];

    threadgroup float shmem[4096];

    // Phase 1: load x with signs into shmem
    for (uint i = tid; i < total_d; i += threads_per_tg) {
        shmem[i] = static_cast<float>(x[batch_idx * total_d + i]) * signs[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: process each pow2 block in place
    uint offset = 0;
    for (uint b = 0; b < n_blocks; b++) {
        uint d_b   = meta[2u + b * 2u];
        uint log_b = meta[3u + b * 2u];

        uint ept_b = (d_b + threads_per_tg - 1u) / threads_per_tg;
        if (ept_b == 0u) ept_b = 1u;

        for (uint stage = 0; stage < log_b; stage++) {
            uint h = 1u << stage;
            uint two_h = 2u * h;

            // Stage new values into per-thread registers (max 4 elems/thread)
            float newv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (uint k = 0; k < ept_b; k++) {
                uint i_local = tid * ept_b + k;
                if (i_local < d_b) {
                    uint bs  = (i_local / two_h) * two_h;
                    uint pos = i_local - bs;
                    float a  = shmem[offset + bs + pos];
                    if (pos < h) {
                        newv[k] = a + shmem[offset + bs + pos + h];
                    } else {
                        newv[k] = shmem[offset + bs + pos - h] - a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < ept_b; k++) {
                uint i_local = tid * ept_b + k;
                if (i_local < d_b) {
                    shmem[offset + i_local] = newv[k];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Normalize block by 1/sqrt(d_b)
        float norm_b = 1.0f / sqrt(static_cast<float>(d_b));
        for (uint k = 0; k < ept_b; k++) {
            uint i_local = tid * ept_b + k;
            if (i_local < d_b) {
                shmem[offset + i_local] *= norm_b;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        offset += d_b;
    }

    // Phase 3: write back to global memory
    for (uint i = tid; i < total_d; i += threads_per_tg) {
        out[batch_idx * total_d + i] = shmem[i];
    }
}


// ============================================================================
//  P17: jangtq_fused_gate_up_swiglu — OPT=10 outputs per thread
// ============================================================================
//
// One kernel computes BOTH gate_proj(x_rot) and up_proj(x_rot), then
// applies SwiGLU activation: out = SiLU(gate * norm_g) * (up * norm_u).
//
// Each thread handles 10 consecutive output rows. SIMD-group reduction
// at the end. Sweet spot from M3 Ultra sweep (P12 found 4 on M4, but
// modern Apple GPUs benefit from much higher tiling).
//
// Grid layout:
//   grid.x = ceil(out_features / 10) * 32   (32 lanes per simd-group)
//   grid.y = K (broadcast experts per token)
//   threadgroup = (min(grid.x, 256), 1, 1)
//
// meta[0]=K, meta[1]=in_features, meta[2]=out_features,
// meta[3]=packed_cols (=in_features/16 for 2-bit), meta[4]=bits (=2)
//
constant constexpr uint JANGTQ_FUSED_OPT = 10;

kernel void jangtq_fused_gate_up_swiglu(
    device const float  *x_rot       [[buffer(0)]],
    device const uint   *packed_gate [[buffer(1)]],
    device const half   *norms_gate  [[buffer(2)]],
    device const uint   *packed_up   [[buffer(3)]],
    device const half   *norms_up    [[buffer(4)]],
    device const float  *codebook    [[buffer(5)]],
    device const uint   *rhs_indices [[buffer(6)]],
    device const uint   *meta        [[buffer(7)]],
    device float        *out_act     [[buffer(8)]],
    uint2 gid                         [[thread_position_in_grid]])
{
    uint global_x     = gid.x;
    uint dispatch_idx = gid.y;

    uint out_group = global_x / 32u;
    uint lane      = global_x % 32u;
    uint out_idx_0 = out_group * JANGTQ_FUSED_OPT;

    uint K            = meta[0];
    uint in_features  = meta[1];
    uint out_features = meta[2];
    uint packed_cols  = meta[3];
    uint bits         = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx     = dispatch_idx % K;
    uint expert    = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask         = (1u << bits) - 1u;

    float acc_g[JANGTQ_FUSED_OPT];
    float acc_u[JANGTQ_FUSED_OPT];
    #pragma unroll
    for (uint o = 0; o < JANGTQ_FUSED_OPT; o++) {
        acc_g[o] = 0.0f;
        acc_u[o] = 0.0f;
    }

    uint expert_base = expert * out_features * packed_cols;
    uint x_off       = token_idx * in_features;

    uint n_outs = JANGTQ_FUSED_OPT;
    if (out_idx_0 + JANGTQ_FUSED_OPT > out_features) {
        n_outs = out_features - out_idx_0;
    }

    // Vectorized 2-bit unpack (P8): one uint32 per thread per pack_idx,
    // extract all 16 indices via shift+mask, accumulate against all 10
    // weight rows in registers.
    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;

        uint pvg[JANGTQ_FUSED_OPT];
        uint pvu[JANGTQ_FUSED_OPT];
        #pragma unroll
        for (uint o = 0; o < JANGTQ_FUSED_OPT; o++) {
            if (o < n_outs) {
                uint row_off = expert_base + (out_idx_0 + o) * packed_cols + pack_idx;
                pvg[o] = packed_gate[row_off];
                pvu[o] = packed_up[row_off];
            } else {
                pvg[o] = 0u;
                pvu[o] = 0u;
            }
        }

        #pragma unroll
        for (uint k = 0; k < 16; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = x_rot[x_off + i];
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < JANGTQ_FUSED_OPT; o++) {
                float w_g = codebook[(pvg[o] >> shift) & mask];
                float w_u = codebook[(pvu[o] >> shift) & mask];
                acc_g[o] += xv * w_g;
                acc_u[o] += xv * w_u;
            }
        }
    }

    // SIMD-group reduction across the 32 lanes
    #pragma unroll
    for (uint o = 0; o < JANGTQ_FUSED_OPT; o++) {
        acc_g[o] = simd_sum(acc_g[o]);
        acc_u[o] = simd_sum(acc_u[o]);
    }

    // Lane 0 writes the n_outs results
    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint  oi = out_idx_0 + o;
            float ng = static_cast<float>(norms_gate[expert * out_features + oi]);
            float nu = static_cast<float>(norms_up  [expert * out_features + oi]);
            float gv = acc_g[o] * ng;
            float uv = acc_u[o] * nu;
            // SwiGLU = SiLU(gv) * uv
            out_act[base_off + oi] = (gv / (1.0f + fast::exp(-gv))) * uv;
        }
    }
}


// ============================================================================
//  P17: jangtq_gather_tq_matmul — OPT=20 outputs per thread (down_proj)
// ============================================================================
//
// Per-row mode: each (token, k) dispatch has its own input row in x_rot.
// Used for down_proj where x has shape (K, intermediate) — each expert's
// activation is rotated independently and feeds a separate matmul.
//
// Higher OPT (20) than fused gate+up (10) because down_proj has
// more outputs per row (3072 vs 1536), so the register tile sweet spot
// is wider.
//
// meta[0]=K_meta (=1 for per_row), meta[1]=in_features, meta[2]=out_features,
// meta[3]=packed_cols, meta[4]=bits
//
constant constexpr uint JANGTQ_GATHER_OPT = 20;

kernel void jangtq_gather_tq_matmul(
    device const float *x_rot       [[buffer(0)]],
    device const uint  *packed      [[buffer(1)]],
    device const half  *norms       [[buffer(2)]],
    device const float *codebook    [[buffer(3)]],
    device const uint  *rhs_indices [[buffer(4)]],
    device const uint  *meta        [[buffer(5)]],
    device float       *out         [[buffer(6)]],
    uint2 gid                        [[thread_position_in_grid]])
{
    uint global_x     = gid.x;
    uint dispatch_idx = gid.y;

    uint out_group = global_x / 32u;
    uint lane      = global_x % 32u;
    uint out_idx_0 = out_group * JANGTQ_GATHER_OPT;

    uint K            = meta[0];
    uint in_features  = meta[1];
    uint out_features = meta[2];
    uint packed_cols  = meta[3];
    uint bits         = meta[4];

    if (out_idx_0 >= out_features) return;

    uint token_idx = dispatch_idx / K;
    uint k_idx     = dispatch_idx % K;
    uint expert    = rhs_indices[token_idx * K + k_idx];

    uint vals_per_u32 = 32u / bits;
    uint mask         = (1u << bits) - 1u;

    float acc[JANGTQ_GATHER_OPT];
    #pragma unroll
    for (uint o = 0; o < JANGTQ_GATHER_OPT; o++) acc[o] = 0.0f;

    uint expert_base = expert * out_features * packed_cols;
    uint x_offset    = token_idx * in_features;

    uint n_outs = JANGTQ_GATHER_OPT;
    if (out_idx_0 + JANGTQ_GATHER_OPT > out_features) {
        n_outs = out_features - out_idx_0;
    }

    for (uint pack_idx = lane; pack_idx < packed_cols; pack_idx += 32u) {
        uint i_base = pack_idx * vals_per_u32;
        uint pv[JANGTQ_GATHER_OPT];
        #pragma unroll
        for (uint o = 0; o < JANGTQ_GATHER_OPT; o++) {
            pv[o] = (o < n_outs)
                ? packed[expert_base + (out_idx_0 + o) * packed_cols + pack_idx]
                : 0u;
        }
        #pragma unroll
        for (uint k = 0; k < 16; k++) {
            uint i = i_base + k;
            if (i >= in_features) break;
            float xv = x_rot[x_offset + i];
            uint shift = k * bits;
            #pragma unroll
            for (uint o = 0; o < JANGTQ_GATHER_OPT; o++) {
                float w = codebook[(pv[o] >> shift) & mask];
                acc[o] += xv * w;
            }
        }
    }

    #pragma unroll
    for (uint o = 0; o < JANGTQ_GATHER_OPT; o++) {
        acc[o] = simd_sum(acc[o]);
    }

    if (lane == 0) {
        uint base_off = (token_idx * K + k_idx) * out_features;
        for (uint o = 0; o < n_outs; o++) {
            uint  oi = out_idx_0 + o;
            float n_v = static_cast<float>(norms[expert * out_features + oi]);
            out[base_off + oi] = acc[o] * n_v;
        }
    }
}
