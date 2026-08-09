"""
TurboQuant encode/decode pipeline.

Combines rotation, optimal codebook, and QJL into complete Key and Value
compression/decompression with full bit-packing for maximum memory savings.

Keys:   TurboQuant_prod -- (b-1)-bit MSE + 1-bit QJL -> unbiased inner products
Values: TurboQuant_mse  -- b-bit MSE -> minimum reconstruction error

Storage (per coordinate):
  Key:   (b-1) bits index + 1 bit QJL sign = b bits total
  Value: b bits index
  + 1 float16 norm per vector (negligible)

Reference: TurboQuant Algorithms 1 & 2 (arXiv:2504.19874)
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from .rotation import generate_random_signs, hadamard_rotate, hadamard_inverse
from .codebook import compute_codebook, quantize_scalar, dequantize_scalar
from .qjl import generate_qjl_projection


# ── Bit packing ────────────────────────────────────────────────────────────

def pack_bits(values: mx.array, bits: int) -> mx.array:
    """Pack integer values into uint32, fitting 32//bits values per uint32.

    Args:
        values: Integer array (uint8), any shape. Values must be in [0, 2^bits-1].
        bits: Bits per value (1, 2, 3, 4, 8).

    Returns:
        Packed uint32 array. Last dim shrinks by factor of (32//bits).
    """
    vals_per_u32 = 32 // bits
    flat = values.reshape(-1).astype(mx.uint32)
    # Pad to multiple of vals_per_u32
    n = flat.shape[0]
    pad = (vals_per_u32 - n % vals_per_u32) % vals_per_u32
    if pad > 0:
        flat = mx.concatenate([flat, mx.zeros(pad, dtype=mx.uint32)])
    flat = flat.reshape(-1, vals_per_u32)
    packed = mx.zeros(flat.shape[0], dtype=mx.uint32)
    for i in range(vals_per_u32):
        packed = packed | (flat[:, i] << (i * bits))
    return packed


def unpack_bits(packed: mx.array, bits: int, n_elements: int) -> mx.array:
    """Unpack uint32 back to integer values.

    Args:
        packed: Packed uint32 array from pack_bits().
        bits: Bits per value (must match pack_bits).
        n_elements: Total number of values to unpack.

    Returns:
        uint8 array of length n_elements.
    """
    vals_per_u32 = 32 // bits
    mask = (1 << bits) - 1
    # Broadcast unpack: one shift/mask over a (..., vals_per_u32) view replaces
    # the per-lane Python loop (vals_per_u32 kernel launches -> 2).
    shifts = mx.arange(vals_per_u32, dtype=mx.uint32) * bits
    unpacked = (packed[..., None] >> shifts) & mask
    return unpacked.astype(mx.uint8).reshape(-1)[:n_elements]


def pack_signs(signs: mx.array) -> mx.array:
    """Pack {-1,+1} float signs into uint32 bitfield. 32 signs per uint32."""
    bits = ((signs.reshape(-1) + 1) / 2).astype(mx.uint32)
    n = bits.shape[0]
    pad = (32 - n % 32) % 32
    if pad > 0:
        bits = mx.concatenate([bits, mx.zeros(pad, dtype=mx.uint32)])
    bits = bits.reshape(-1, 32)
    packed = mx.zeros(bits.shape[0], dtype=mx.uint32)
    for i in range(32):
        packed = packed | (bits[:, i] << i)
    return packed


def unpack_signs(packed: mx.array, n_elements: int) -> mx.array:
    """Unpack uint32 bitfield back to {-1,+1} float array."""
    # Broadcast unpack: the 32-iteration Python loop dispatched ~100 small
    # kernels per call; this is the block-restore decode hot path (vmlx#91).
    shifts = mx.arange(32, dtype=mx.uint32)
    bits_ = (packed[..., None] >> shifts) & 1
    flat = bits_.reshape(-1)[:n_elements].astype(mx.float32)
    return flat * 2 - 1


_METAL_INVERSE = None  # lazily resolved: False = unavailable


def _hadamard_inverse_fast(y: mx.array, signs: mx.array) -> mx.array:
    """Inverse RHT via the fused Metal butterfly when available.

    H is self-inverse up to sign order: inverse = H(y) * signs. The Metal
    kernel computes H(s ⊙ y); with s = ones that is exactly H(y). The Python
    butterfly dispatches ~6 ops per stage per pow2 block at full element
    count — the dominant cost of TQ block-restore decode (vmlx#91).
    """
    global _METAL_INVERSE
    if _METAL_INVERSE is None:
        try:
            from .hadamard_kernel import hadamard_rotate_metal
            _METAL_INVERSE = hadamard_rotate_metal if mx.metal.is_available() else False
        except Exception:
            _METAL_INVERSE = False
    if _METAL_INVERSE is False:
        return hadamard_inverse(y, signs)
    flat = y.reshape(-1, y.shape[-1])
    out = _METAL_INVERSE(flat, mx.ones_like(signs))
    return (out * signs).reshape(y.shape)


# ── Data structures ────────────────────────────────────────────────────────

class EncodedKeys(NamedTuple):
    """Compressed key representation — fully packed."""
    indices_packed: mx.array   # Codebook indices packed into uint32
    qjl_packed: mx.array       # QJL signs packed into uint32 (1 bit each)
    residual_norms: mx.array   # ||residual||_2 per vector (float16)
    vector_norms: mx.array     # ||key||_2 per vector (float16)
    shape: tuple               # Original shape for unpacking
    index_bits: int            # Bits per index (for unpacking)


class EncodedValues(NamedTuple):
    """Compressed value representation — fully packed."""
    indices_packed: mx.array   # Codebook indices packed into uint32
    vector_norms: mx.array     # ||value||_2 per vector (float16)
    shape: tuple               # Original shape for unpacking
    index_bits: int            # Bits per index


# ── Encoder ────────────────────────────────────────────────────────────────

@dataclass
class TurboQuantEncoder:
    """Precomputed rotation signs, codebooks, and QJL projection."""
    dim: int
    key_bits: int
    value_bits: int
    seed: int

    def __post_init__(self):
        self.rotation_signs = generate_random_signs(self.dim, seed=self.seed)
        k_mse_bits = max(self.key_bits - 1, 1)
        self.key_codebook_list = compute_codebook(self.dim, k_mse_bits)
        self.key_codebook = mx.array(np.array(self.key_codebook_list, dtype=np.float32))
        self.key_index_bits = k_mse_bits
        self.value_codebook_list = compute_codebook(self.dim, self.value_bits)
        self.value_codebook = mx.array(np.array(self.value_codebook_list, dtype=np.float32))
        self.value_index_bits = self.value_bits
        self.qjl_S = generate_qjl_projection(self.dim, seed=self.seed + 1000)


# ── Encode / Decode ────────────────────────────────────────────────────────

def encode_keys(keys: mx.array, enc: TurboQuantEncoder) -> EncodedKeys:
    """Compress keys: rotate → MSE quantize → QJL residual → pack."""
    orig_shape = keys.shape
    dim = orig_shape[-1]

    vector_norms = mx.sqrt(mx.sum(keys * keys, axis=-1, keepdims=True))
    keys_unit = keys / (vector_norms + 1e-8)
    keys_rotated = hadamard_rotate(keys_unit, enc.rotation_signs)

    flat_rotated = keys_rotated.reshape(-1, dim)
    mse_indices = quantize_scalar(flat_rotated, enc.key_codebook)
    mse_dequant = dequantize_scalar(mse_indices, enc.key_codebook)

    residual = flat_rotated - mse_dequant
    projected = residual @ enc.qjl_S.T
    qjl_signs = mx.where(projected >= 0, mx.array(1.0), mx.array(-1.0))
    residual_norms = mx.sqrt(mx.sum(residual * residual, axis=-1, keepdims=True))

    return EncodedKeys(
        indices_packed=pack_bits(mse_indices.reshape(-1), enc.key_index_bits),
        qjl_packed=pack_signs(qjl_signs.reshape(-1)),
        residual_norms=residual_norms.reshape(orig_shape[:-1] + (1,)).astype(mx.float16),
        vector_norms=vector_norms.astype(mx.float16),
        shape=orig_shape,
        index_bits=enc.key_index_bits,
    )


def decode_keys(encoded: EncodedKeys, enc: TurboQuantEncoder) -> mx.array:
    """Decompress keys: unpack → dequant → QJL correct → inverse rotate → scale."""
    orig_shape = encoded.shape
    dim = orig_shape[-1]
    n_elements = 1
    for s in orig_shape:
        n_elements *= s

    flat_indices = unpack_bits(encoded.indices_packed, encoded.index_bits, n_elements).reshape(-1, dim)
    flat_qjl = unpack_signs(encoded.qjl_packed, n_elements).reshape(-1, dim)
    flat_res_norms = encoded.residual_norms.astype(mx.float32).reshape(-1, 1)
    flat_vec_norms = encoded.vector_norms.astype(mx.float32).reshape(-1, 1)

    mse_dequant = dequantize_scalar(flat_indices, enc.key_codebook)

    qjl_scale = math.sqrt(math.pi / 2.0) / dim
    qjl_dequant = qjl_scale * flat_res_norms * (flat_qjl @ enc.qjl_S)

    reconstructed_rotated = (mse_dequant + qjl_dequant).reshape(orig_shape)
    reconstructed_unit = _hadamard_inverse_fast(reconstructed_rotated, enc.rotation_signs)

    return reconstructed_unit * flat_vec_norms.reshape(orig_shape[:-1] + (1,))


def encode_values(values: mx.array, enc: TurboQuantEncoder) -> EncodedValues:
    """Compress values: rotate → MSE quantize → pack."""
    orig_shape = values.shape
    dim = orig_shape[-1]

    vector_norms = mx.sqrt(mx.sum(values * values, axis=-1, keepdims=True))
    values_unit = values / (vector_norms + 1e-8)
    values_rotated = hadamard_rotate(values_unit, enc.rotation_signs)

    flat_rotated = values_rotated.reshape(-1, dim)
    mse_indices = quantize_scalar(flat_rotated, enc.value_codebook)

    return EncodedValues(
        indices_packed=pack_bits(mse_indices.reshape(-1), enc.value_index_bits),
        vector_norms=vector_norms.astype(mx.float16),
        shape=orig_shape,
        index_bits=enc.value_index_bits,
    )


def decode_values(encoded: EncodedValues, enc: TurboQuantEncoder) -> mx.array:
    """Decompress values: unpack → dequant → inverse rotate → scale."""
    orig_shape = encoded.shape
    dim = orig_shape[-1]
    n_elements = 1
    for s in orig_shape:
        n_elements *= s

    flat_indices = unpack_bits(encoded.indices_packed, encoded.index_bits, n_elements).reshape(-1, dim)
    flat_vec_norms = encoded.vector_norms.astype(mx.float32).reshape(-1, 1)

    mse_dequant = dequantize_scalar(flat_indices, enc.value_codebook)
    mse_dequant = mse_dequant.reshape(orig_shape)
    reconstructed_unit = _hadamard_inverse_fast(mse_dequant, enc.rotation_signs)

    return reconstructed_unit * flat_vec_norms.reshape(orig_shape[:-1] + (1,))
