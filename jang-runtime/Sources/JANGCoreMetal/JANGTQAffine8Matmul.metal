//
// JANGTQ 8-bit MLX-format quantized GEMV — for attention / embed / lm_head.
// Created by Jinho Jang (eric@jangq.ai).
//
// Used by the JANGTQ Swift inference engine for the non-MoE weights:
//   - self_attn.{q,k,v,o}_proj
//   - embed_tokens
//   - lm_head
//
// Format (matches mx.quantize(bits=8, group_size=64) exactly, verified
// against MiniMax M2.7 JANGTQ_2L shards):
//
//   weight : uint32, shape (out, in/4). One uint32 = 4 × 8-bit values
//            packed LSB-first. Position k (0..3) at bit k*8.
//   scales : half, shape (out, in/group_size).
//   biases : half, shape (out, in/group_size).
//   x      : half, shape (in,)
//   y      : float, shape (out,) — fp32 accum to avoid drift.
//
//   Dequant: val = q_int * scale + bias  (per group of `group_size` cols)
//
// One thread per output row. Same naive shape as the 4-bit reference; perf
// tuning (SIMD reduction, register tiling) is a follow-up.
//

#include <metal_stdlib>
using namespace metal;

struct QuantMatmul8Params {
    uint in_features;
    uint out_features;
    uint group_size;
    uint n_groups;     // in_features / group_size
    uint packed_in;    // in_features / 4   (8-bit: 4 per uint32)
};

// Naive GEMV — one thread per output row. Kept for reference/correctness.
kernel void jangtq_quant_matmul_8bit_gemv(
    device const uint32_t*         qweight [[buffer(0)]],
    device const half*             scales  [[buffer(1)]],
    device const half*             biases  [[buffer(2)]],
    device const half*             x       [[buffer(3)]],
    device       float*            y       [[buffer(4)]],
    constant QuantMatmul8Params&   p       [[buffer(5)]],
    uint tid                                [[thread_position_in_grid]]
) {
    if (tid >= p.out_features) return;

    const uint out = tid;
    const uint row_q_offset = out * p.packed_in;
    const uint row_s_offset = out * p.n_groups;

    float acc = 0.0f;

    for (uint g = 0; g < p.n_groups; g++) {
        const float scale = float(scales[row_s_offset + g]);
        const float bias  = float(biases[row_s_offset + g]);
        const uint  g_start = g * p.group_size;
        const uint  words_per_group = p.group_size / 4u;

        for (uint w = 0; w < words_per_group; w++) {
            const uint i_base = g_start + w * 4u;
            const uint32_t word = qweight[row_q_offset + (g_start / 4u) + w];

            for (uint k = 0; k < 4; k++) {
                const uint q = (word >> (k * 8u)) & 0xFFu;
                const float dq = float(q) * scale + bias;
                const float xv = float(x[i_base + k]);
                acc = fma(dq, xv, acc);
            }
        }
    }

    y[out] = acc;
}


// SIMD-reduced GEMV — 32 threads cooperate on one output row.
// Each thread handles `packed_in / 32` words, then simd_sum reduces.
//
// Grid: (out_features * 32, 1, 1)
// Threadgroup: (256, 1, 1) → 8 simd-groups → 8 output rows per TG
//
// This matches the design of the JANGTQ matmul kernels and gives a
// ~10-30× speedup over the naive version for typical attention shapes
// because each row's 768-word loop is split across 32 threads with
// minimal extra coordination cost.
//
kernel void jangtq_quant_matmul_8bit_gemv_simd(
    device const uint32_t*         qweight [[buffer(0)]],
    device const half*             scales  [[buffer(1)]],
    device const half*             biases  [[buffer(2)]],
    device const half*             x       [[buffer(3)]],
    device       float*            y       [[buffer(4)]],
    constant QuantMatmul8Params&   p       [[buffer(5)]],
    uint global_id                          [[thread_position_in_grid]]
) {
    const uint out = global_id / 32u;
    const uint lane = global_id % 32u;
    if (out >= p.out_features) return;

    const uint row_q_offset = out * p.packed_in;
    const uint row_s_offset = out * p.n_groups;

    // Each thread strides through packed_in by 32. Per word: 4 8-bit values.
    float acc = 0.0f;
    for (uint pack_idx = lane; pack_idx < p.packed_in; pack_idx += 32u) {
        const uint i_base = pack_idx * 4u;
        const uint  g     = i_base / p.group_size;
        const float scale = float(scales[row_s_offset + g]);
        const float bias  = float(biases[row_s_offset + g]);
        const uint32_t word = qweight[row_q_offset + pack_idx];

        // Unpack 4 × 8-bit and fma into the running sum
        #pragma unroll
        for (uint k = 0; k < 4; k++) {
            const uint q = (word >> (k * 8u)) & 0xFFu;
            const float dq = float(q) * scale + bias;
            const float xv = float(x[i_base + k]);
            acc = fma(dq, xv, acc);
        }
    }

    acc = simd_sum(acc);
    if (lane == 0) {
        y[out] = acc;
    }
}
