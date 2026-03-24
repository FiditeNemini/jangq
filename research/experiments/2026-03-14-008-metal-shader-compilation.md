# Experiment 008: Metal Shader Compilation

**Date**: 2026-03-14
**Author**: Eric Jang (eric@vmlx.net)
**Status**: PASS — all shaders compile clean

## Setup

- **Xcode**: macOS Tahoe, Metal 3.0 target
- **Metal Toolchain**: 17C7003j (704.6 MB, downloaded fresh)
- **Shaders**: JANGDequant.metal, JANGCompute.metal

## Results

| Shader | Kernels | Warnings | Errors | Status |
|--------|---------|----------|--------|--------|
| JANGDequant.metal | 3 (dequant, gemv, gemm) | 0 | 0 | PASS |
| JANGCompute.metal | 7 (rmsnorm, rope, softmax, silu, silu_mul, add, embedding) | 0 | 0 | PASS |
| jang.metallib | 10 total | 0 | 0 | 44,993 bytes |

### Kernels Compiled

**JANGDequant.metal:**
1. `jang_dequantize` — standalone dequant to float16 buffer
2. `jang_dequant_gemv` — fused dequant + matrix-vector multiply (token generation)
3. `jang_dequant_gemm` — fused dequant + matrix-matrix multiply (prefill)

**JANGCompute.metal:**
4. `jang_rms_norm` — RMSNorm
5. `jang_rope` — Rotary Position Embeddings
6. `jang_softmax` — Numerically stable softmax
7. `jang_silu` — SiLU activation
8. `jang_silu_mul` — Fused SiLU + multiply (SwiGLU)
9. `jang_add` — Residual connection
10. `jang_embedding` — Token embedding lookup

## Implementation Notes

- All kernels use float32 accumulation for numerical stability
- Dequant uses fast paths for 2-bit and 4-bit (direct bit shift, no general extraction)
- General extraction handles 3, 5, 6-bit via bit offset calculation
- GEMV uses SIMD reduction (simd_sum) for thread cooperation
- GEMM uses tiled approach with threadgroup shared memory
- RoPE supports position offset for KV cache continuation

## Not Yet Tested

- Actual GPU execution (compilation only verifies syntax and types)
- Performance benchmarking
- Numerical correctness vs CPU reference
- These are next steps — requires Swift runtime to dispatch kernels

## Required for Metal Toolchain

Had to run `xcodebuild -downloadComponent MetalToolchain` (704.6 MB download)
before `xcrun metal` would work. This is a macOS Tahoe requirement.
