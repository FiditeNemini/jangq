//
// JANG v2 quantized matmul — 4-bit GEMV (y = W @ x, single token).
// Created by Eric Jang (eric@jangq.ai).
//
// Convention (verified against mx.quantize on 2026-04-13):
//
//   W  packed as uint32, shape (out, in/8). Each uint32 holds 8 consecutive
//      4-bit values, LSB-first: position k (0..7) at bit k*4.
//   scales / biases: half, shape (out, in/group_size). One value per group.
//   x: half, shape (in,).
//   y: float, shape (out,) — kept in fp32 to avoid accumulation drift.
//
//   Dequant formula: val = q_int * scale + bias, per-group.
//
// This first implementation is intentionally naive: one thread per output
// row, no SIMD reduction, no threadgroup memory. Correctness first, perf
// in a later pass.
//

#include <metal_stdlib>
using namespace metal;

struct QuantMatmul4Params {
    uint in_features;
    uint out_features;
    uint group_size;
    uint n_groups;         // in_features / group_size
    uint packed_in;        // in_features / 8   (4-bit: 8 per uint32)
};

kernel void jang_v2_quant_matmul_4bit_gemv(
    device const uint32_t*      qweight [[buffer(0)]],  // [out * packed_in]
    device const half*          scales  [[buffer(1)]],  // [out * n_groups]
    device const half*          biases  [[buffer(2)]],  // [out * n_groups]
    device const half*          x       [[buffer(3)]],  // [in]
    device       float*         y       [[buffer(4)]],  // [out]
    constant QuantMatmul4Params& p      [[buffer(5)]],
    uint tid                            [[thread_position_in_grid]]
) {
    if (tid >= p.out_features) {
        return;
    }

    const uint out = tid;
    const uint row_q_offset = out * p.packed_in;
    const uint row_s_offset = out * p.n_groups;

    float acc = 0.0f;

    for (uint g = 0; g < p.n_groups; g++) {
        const float scale = float(scales[row_s_offset + g]);
        const float bias  = float(biases[row_s_offset + g]);
        const uint  g_start = g * p.group_size;

        const uint words_per_group = p.group_size / 8u;

        for (uint w = 0; w < words_per_group; w++) {
            const uint i_base = g_start + w * 8u;
            const uint32_t word = qweight[row_q_offset + (g_start / 8u) + w];

            for (uint k = 0; k < 8; k++) {
                const uint q = (word >> (k * 4u)) & 0xFu;
                const float dq = float(q) * scale + bias;
                const float xv = float(x[i_base + k]);
                acc = fma(dq, xv, acc);
            }
        }
    }

    y[out] = acc;
}
