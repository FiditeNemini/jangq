"""
TurboQuantLinear — Metal kernel for fused TQ dequant + matmul.
Created by Jinho Jang (eric@jangq.ai)

Performs codebook-quantized matmul directly on packed weights without
decompressing to affine form. Keeps the compact TQ representation in
GPU memory (packed uint32 + float16 norms + small codebook).

## Math

Weights are quantized as: w_rot = H @ diag(signs) @ w (row-wise on in dim)
Then quantized: w_rot[r, i] ≈ codebook[packed_idx(r, i)] * norms[r]

At inference: y = x @ w^T. Using H symmetric, H @ H = I:
  y[b, r] = norms[r] * sum_i x_rot[b, i] * codebook[packed_idx(r, i)]
  where x_rot = H @ (signs * x)   ← small input rotation, O(d log d) once

This "rotate-x-once" approach (from QuIP#) moves the Hadamard off the weight
(huge) onto the input (small), making the matmul kernel trivially fused.

## Format per weight matrix
  - packed: (out_features, packed_cols) uint32 — codebook indices at N bits
  - norms: (out_features,) float16 — per-row L2 norm
  - codebook: (2^bits,) float32 — Lloyd-Max centroids
  - signs: (in_features,) float32 — random signs for Hadamard rotation (stored per layer)

## Memory savings vs affine quantization
  - Affine 2-bit: packed + scales + biases = weight + 2 * (weight_rows * groups) float16
  - TQ 2-bit: packed + norms + codebook = weight + weight_rows * 1 float16 + 4 float32
  - For GLM-5.1 (744B): affine=234 GB vs TQ=191 GB in GPU memory
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional

from .codebook import compute_codebook
from .rotation import generate_random_signs
from .hadamard_kernel import hadamard_rotate_metal as hadamard_rotate


# P1: shared signs/codebook cache keyed by (in_features, bits, seed).
# Critical for gate/up to share rotated x: if both modules reference the SAME
# Python object for `signs`, the rotation cache hits.
_SIGNS_CACHE = {}
_CODEBOOK_CACHE = {}


def _get_signs(in_features, seed):
    key = (in_features, seed)
    if key not in _SIGNS_CACHE:
        _SIGNS_CACHE[key] = mx.array(
            generate_random_signs(in_features, seed=seed), dtype=mx.float32
        )
    return _SIGNS_CACHE[key]


def _get_codebook(in_features, bits):
    key = (in_features, bits)
    if key not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[key] = mx.array(compute_codebook(in_features, bits), dtype=mx.float32)
    return _CODEBOOK_CACHE[key]


# Metal kernel source for fused TQ dequant + matmul
# Each thread computes one output element: dot(x, dequant(w_row))
_TQ_MATMUL_SOURCE = '''
    // Fused TQ matmul: y[b, r] = norms[r] * sum_i x_rot[b, i] * codebook[unpack(packed[r, i])]
    // The input x must already be rotated (x_rot = H @ (signs * x)) before calling this.
    uint batch_idx = thread_position_in_grid.y;
    uint out_idx = thread_position_in_grid.x;

    uint out_features = tq_meta[0];
    uint packed_cols = tq_meta[1];
    uint in_features = tq_meta[2];
    uint bits = tq_meta[3];
    uint vals_per_u32 = 32 / bits;
    uint mask = (1u << bits) - 1u;

    if (out_idx >= out_features) return;

    // Accumulate: sum_i x_rot[b, i] * codebook[unpack(packed[r, i])]
    float acc = 0.0f;

    for (uint i = 0; i < in_features; i++) {
        // Unpack codebook index for weight[out_idx, i]
        uint pack_idx = i / vals_per_u32;
        uint bit_offset = (i % vals_per_u32) * bits;
        uint packed_val = packed[out_idx * packed_cols + pack_idx];
        uint cb_idx = (packed_val >> bit_offset) & mask;

        // Codebook lookup
        float w_val = codebook[cb_idx];

        // Dot product accumulation
        float x_val = static_cast<float>(x_rot[batch_idx * in_features + i]);
        acc += x_val * w_val;
    }

    // Scale by per-row norm
    float norm = static_cast<float>(norms[out_idx]);
    out[batch_idx * out_features + out_idx] = acc * norm;
'''


def _create_tq_matmul_kernel():
    """Create the Metal kernel for TQ matmul (input assumed pre-rotated)."""
    return mx.fast.metal_kernel(
        name="tq_matmul",
        input_names=["x_rot", "packed", "norms", "codebook", "tq_meta"],
        output_names=["out"],
        source=_TQ_MATMUL_SOURCE,
    )


# Cache the compiled kernel
_kernel = None


def tq_matmul(x: mx.array, packed: mx.array, norms: mx.array,
              codebook: mx.array, signs: mx.array,
              in_features: int, bits: int) -> mx.array:
    """Fused TQ dequant + matmul using Metal kernel.

    Rotates x once (H @ (signs * x)) then calls fused kernel for
    unpack + codebook + matmul + norm scale.

    Args:
        x: (batch, in_features) or (in_features,) input (unrotated)
        packed: (out_features, packed_cols) uint32
        norms: (out_features,) float16
        codebook: (2^bits,) float32
        signs: (in_features,) float32 (+1/-1)
        in_features: int
        bits: int

    Returns:
        (batch, out_features) float32
    """
    global _kernel
    if _kernel is None:
        _kernel = _create_tq_matmul_kernel()

    # Handle 1D input
    squeeze = False
    if x.ndim == 1:
        x = x[None, :]
        squeeze = True

    # Rotate input: x_rot = H @ (signs * x)
    # Since H is symmetric, this is the same as hadamard_rotate applied to x.
    x_rot = hadamard_rotate(x.astype(mx.float32), signs)

    batch_size = x.shape[0]
    out_features = packed.shape[0]

    # Metadata: [out_features, packed_cols, in_features, bits]
    tq_meta = mx.array([packed.shape[0], packed.shape[1], in_features, bits],
                        dtype=mx.uint32)

    out = _kernel(
        inputs=[x_rot, packed, norms, codebook, tq_meta],
        output_shapes=[(batch_size, out_features)],
        output_dtypes=[mx.float32],
        grid=(out_features, batch_size, 1),
        threadgroup=(min(out_features, 256), 1, 1),
    )

    result = out[0]
    if result.dtype != x.dtype:
        result = result.astype(x.dtype)

    if squeeze:
        return result.squeeze(0)
    return result


class TurboQuantLinear(nn.Module):
    """Linear layer using fused Metal TQ kernel — no dequant to affine.

    Drop-in replacement for nn.QuantizedLinear. Keeps TQ format
    (packed + norms + codebook) in GPU memory instead of expanding
    to affine (packed + scales + biases).

    AWQ support: if `awq_scale` attribute is set (shape (in_features,)),
    input x is divided by the scale before the matmul.  This recovers
    correct output from weights that were pre-scaled by W * awq_scale
    at convert time.
    """

    def __init__(self, in_features: int, out_features: int, bits: int = 2,
                 bias: bool = False, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self._seed = seed

        vals_per_u32 = 32 // bits
        packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
        self.packed = mx.zeros((out_features, packed_cols), dtype=mx.uint32)
        self.norms = mx.zeros((out_features,), dtype=mx.float16)

        if bias:
            self.bias = mx.zeros((out_features,))

        self.freeze()

    @property
    def codebook(self):
        return _get_codebook(self.in_features, self.bits)

    @property
    def signs(self):
        return _get_signs(self.in_features, self._seed)

    def __call__(self, x: mx.array) -> mx.array:
        # AWQ support: if `awq_scale` attribute exists (attached by loader
        # for AWQ-enabled bundles), divide x by it before matmul. Use
        # getattr to avoid side effects when attribute was never set.
        awq = getattr(self, "awq_scale", None)
        if awq is not None:
            x = x / awq.astype(x.dtype)
        y = tq_matmul(x, self.packed, self.norms, self.codebook, self.signs,
                       self.in_features, self.bits)
        if "bias" in self:
            y = y + self.bias
        return y


class TurboQuantSwitchLinear(nn.Module):
    """MoE switch layer using fused Metal TQ kernel.

    Only dequants active experts via the kernel — inactive experts
    stay packed in GPU memory.

    AWQ support: same as TurboQuantLinear — if `awq_scale` is set, x
    is divided by it before matmul.  For MoE all experts share the
    same per-layer scale (captured at shared-input point).
    """

    def __init__(self, in_features: int, out_features: int, num_experts: int,
                 bits: int = 2, bias: bool = False, seed: int = 42):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.bits = bits
        self._seed = seed

        vals_per_u32 = 32 // bits
        packed_cols = (in_features + vals_per_u32 - 1) // vals_per_u32
        self.packed = mx.zeros((num_experts, out_features, packed_cols), dtype=mx.uint32)
        self.norms = mx.zeros((num_experts, out_features), dtype=mx.float16)

        if bias:
            self.bias = mx.zeros((num_experts, out_features))

        self.freeze()

    @property
    def codebook(self):
        return _get_codebook(self.in_features, self.bits)

    @property
    def signs(self):
        return _get_signs(self.in_features, self._seed)

    def __call__(self, x: mx.array, indices: mx.array, sorted_indices: bool = False) -> mx.array:
        """Fused gather + TQ matmul via Metal kernel."""
        awq = getattr(self, "awq_scale", None)
        if awq is not None:
            x = x / awq.astype(x.dtype)
        return _gather_tq_matmul(
            x, self.packed, self.norms, self.codebook, self.signs,
            indices, bits=self.bits, sorted_indices=sorted_indices,
        )


# Module-level import — avoid doing this inside __call__ on every decode step.
from .gather_tq_kernel import gather_tq_matmul as _gather_tq_matmul
