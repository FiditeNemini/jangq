"""Opt-in MPP/TensorOps lane for TurboQuant dense matmul.

This is intentionally not the default routed-expert path. JANGTQ stores
codebook indices + per-row norms, while Metal tensor_ops consume dense half,
bfloat, float, or int8 tensors. The bridge here materializes a dense half
matrix in the rotated-weight domain and then uses MPP ``matmul2d``.

Use it to prove correctness and speed characteristics before porting the same
idea to routed ``gather_tq`` / fused gate-up paths.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .hadamard_kernel import hadamard_rotate_metal


_MPP_HEADER = """#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp;
"""


@lru_cache(maxsize=1)
def mpp_tensorops_available() -> bool:
    """Return true when MLX custom kernels can compile and run MPP matmul2d."""
    try:
        kernel = mx.fast.metal_kernel(
            name="jangtq_mpp_tensorops_smoke",
            input_names=["inp", "w"],
            output_names=["out"],
            header=_MPP_HEADER,
            source="""
device half* xh = const_cast<device half*>(reinterpret_cast<const device half*>(inp));
device half* wh = const_cast<device half*>(reinterpret_cast<const device half*>(w));
device float* of = out;
constexpr auto desc = tensor_ops::matmul2d_descriptor(1, 8, 16, false, false, false);
tensor_ops::matmul2d<desc, execution_thread> op;
auto X = tensor(xh, extents<int, 16, 1>(), array<int, 2>{1, 16});
auto W = tensor(wh, extents<int, 8, 16>(), array<int, 2>{1, 8});
auto O = tensor(of, extents<int, 8, 1>(), array<int, 2>{1, 8});
op.run(X, W, O);
""",
        )
        x = mx.ones((1, 16), dtype=mx.float16)
        w = mx.ones((16, 8), dtype=mx.float16)
        out = kernel(
            inputs=[x, w],
            output_shapes=[(1, 8)],
            output_dtypes=[mx.float32],
            grid=(1, 1, 1),
            threadgroup=(1, 1, 1),
        )[0]
        mx.eval(out)
        return bool(mx.all(mx.abs(out - 16.0) < 1e-3).item())
    except Exception:
        return False


@lru_cache(maxsize=64)
def _make_dequant_kernel(in_features: int, out_features: int, bits: int):
    vals_per_u32 = 32 // bits
    source = f"""
uint out_idx = thread_position_in_grid.x;
uint in_idx = thread_position_in_grid.y;
if (out_idx >= {out_features}u || in_idx >= {in_features}u) return;

uint pack_idx = in_idx / {vals_per_u32}u;
uint bit_offset = (in_idx % {vals_per_u32}u) * {bits}u;
uint mask = (1u << {bits}u) - 1u;
uint packed_val = packed[out_idx * {((in_features + vals_per_u32 - 1) // vals_per_u32)}u + pack_idx];
uint cb_idx = (packed_val >> bit_offset) & mask;
float value = codebook[cb_idx] * static_cast<float>(norms[out_idx]);

// Dense layout is (K, N) so MPP can multiply X(B,K) by W(K,N).
dense[in_idx * {out_features}u + out_idx] = static_cast<half>(value);
"""
    return mx.fast.metal_kernel(
        name=f"jangtq_dequant_dense_half_b{bits}_i{in_features}_o{out_features}",
        input_names=["packed", "norms", "codebook"],
        output_names=["dense"],
        source=source,
    )


@lru_cache(maxsize=64)
def _make_mpp_thread_matmul_kernel(
    batch_size: int, in_features: int, out_features: int
):
    source = f"""
device half* xh = const_cast<device half*>(reinterpret_cast<const device half*>(x_rot));
device half* wh = const_cast<device half*>(reinterpret_cast<const device half*>(dense));
device float* of = out;

constexpr auto desc = tensor_ops::matmul2d_descriptor(
    {batch_size}, {out_features}, static_cast<int>(dynamic_extent),
    false, false, false
);
tensor_ops::matmul2d<desc, execution_thread> op;

// MPP tensor coordinate order here is (K, M), (N, K), (N, M).
auto X = tensor(
    xh,
    dextents<int, 2>{{{in_features}, {batch_size}}},
    array<int, 2>{{1, {in_features}}}
);
auto W = tensor(
    wh,
    dextents<int, 2>{{{out_features}, {in_features}}},
    array<int, 2>{{1, {out_features}}}
);
auto O = tensor(
    of,
    dextents<int, 2>{{{out_features}, {batch_size}}},
    array<int, 2>{{1, {out_features}}}
);
op.run(X, W, O);
"""
    return mx.fast.metal_kernel(
        name=f"jangtq_mpp_dense_matmul_b{batch_size}_i{in_features}_o{out_features}",
        input_names=["x_rot", "dense"],
        output_names=["out"],
        header=_MPP_HEADER,
        source=source,
    )


def _dequantize_dense_half(
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
    in_features: int,
    bits: int,
) -> mx.array:
    out_features = int(packed.shape[0])
    kernel = _make_dequant_kernel(in_features, out_features, bits)
    dense = kernel(
        inputs=[packed, norms, codebook],
        output_shapes=[(in_features, out_features)],
        output_dtypes=[mx.float16],
        grid=(out_features, in_features, 1),
        threadgroup=(16, 16, 1),
    )[0]
    return dense


def tq_matmul_mpp_dense(
    x: mx.array,
    packed: mx.array,
    norms: mx.array,
    codebook: mx.array,
    signs: mx.array,
    in_features: int,
    bits: int,
) -> mx.array:
    """Run TQ matmul via dequantized dense half + MPP tensor_ops matmul2d.

    This preserves the existing TQ math domain:
    ``x_rot @ (codebook[packed] * norm).T``.
    """
    if not mpp_tensorops_available():
        raise RuntimeError("MPP tensor_ops unavailable for MLX custom kernels")

    squeeze = False
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True

    orig_shape = x.shape
    if x.ndim > 2:
        x_flat = x.reshape(-1, in_features)
    else:
        x_flat = x

    out_features = int(packed.shape[0])
    batch_size = int(x_flat.shape[0])
    x_rot = hadamard_rotate_metal(x_flat.astype(mx.float32), signs).astype(mx.float16)
    dense = _dequantize_dense_half(packed, norms, codebook, in_features, bits)
    kernel = _make_mpp_thread_matmul_kernel(batch_size, in_features, out_features)
    out = kernel(
        inputs=[x_rot, dense],
        output_shapes=[(batch_size, out_features)],
        output_dtypes=[mx.float32],
        grid=(1, 1, 1),
        threadgroup=(1, 1, 1),
    )[0]

    if x.ndim > 2:
        out = out.reshape(*orig_shape[:-1], out_features)
    if squeeze:
        out = out.squeeze(0)
    if out.dtype != x.dtype:
        out = out.astype(x.dtype)
    return out
