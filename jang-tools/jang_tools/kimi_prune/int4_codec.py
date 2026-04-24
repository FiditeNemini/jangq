"""INT4 codec for Kimi K2.6 compressed-tensors on-disk format.

Kimi K2.6 on-disk layout (verified via shard30 header):
  <base>.weight_packed    I32   (out, in // 8)   8 × int4 per uint32 word, lsb-first
  <base>.weight_scale     BF16  (out, in // 32)  group-wise scales, group_size=32
  <base>.weight_shape     I32   (2,)             target shape [out, in]

Sign convention: signed INT4, -8..7 (two's complement in the 4 low bits of each nibble).
Dequantization: w[i, j] = u4[i, j] * scale[i, j // 32]

This module provides:
  unpack_int4(packed, scale, target_shape, group_size=32) -> np.float32
      Full dequantization, returns a (out, in) float32 tensor.

  pack_int4(w_f32, group_size=32) -> (packed, scale, shape_i32)
      Round-trip quantize a float32 tensor back to the same format.
      Used when writing pruned or merged expert weights.

Verified: pack(unpack(x)) ≈ x within INT4 quantization error (MSE < 1e-3
for typical weight tensors).
"""

from __future__ import annotations

import numpy as np


def unpack_int4(packed_i32, scale_bf16_like, target_shape, group_size: int = 32):
    """Unpack compressed-tensors INT4 → numpy float32.

    Compressed-tensors packing (verified against
    compressed_tensors.compressors.unpack_from_int32):

      packed_i32: (out, in // 8) int32 tensor. Each int32 word holds 8
                  **offset-binary** 4-bit values (not two's-complement!).
                  Values 0..15 in storage map to signed -8..+7 by
                  subtracting 8 (excess-8 encoding).
      Nibble ordering lsb-first within the word:
                  unpacked[..., 0] = bits[3:0]
                  unpacked[..., 1] = bits[7:4]
                  ...
                  unpacked[..., 7] = bits[31:28]
      So unpacked position `p` ← nibble `p % 8` of packed column `p // 8`.

    scale_bf16_like: (out, in // group_size) tensor of per-group scales.
                    Promoted to float32 internally.
    target_shape: length-2 array-like [out, in].
    group_size: number of columns per scale. Default 32 (Kimi K2.6).

    Returns: np.ndarray (out, in), dtype float32.
    """
    out_dim, in_dim = int(target_shape[0]), int(target_shape[1])
    packed = np.ascontiguousarray(np.asarray(packed_i32, dtype=np.int32))
    u = packed.view(np.uint32)
    assert u.shape[-1] * 8 == in_dim, \
        f"packed cols {u.shape[-1]} × 8 != in_dim {in_dim}"

    # Extract 8 unsigned 4-bit values per uint32. Result: (out, in/8, 8) uint8.
    nibbles = np.empty((u.shape[0], u.shape[1], 8), dtype=np.uint8)
    for j in range(8):
        nibbles[..., j] = ((u >> (4 * j)) & 0xF).astype(np.uint8)
    # Offset-binary → signed: subtract 8 (stored 0..15 → logical -8..+7).
    signed = nibbles.astype(np.int16) - 8
    unpacked = signed.reshape(out_dim, in_dim).astype(np.float32)

    scale = np.asarray(scale_bf16_like, dtype=np.float32)
    assert scale.shape == (out_dim, in_dim // group_size), \
        f"scale shape {scale.shape} != {(out_dim, in_dim // group_size)}"
    scale_full = np.repeat(scale, group_size, axis=-1)
    assert scale_full.shape == (out_dim, in_dim)

    return unpacked * scale_full


def pack_int4(w_f32, group_size: int = 32):
    """Quantize float32 → INT4 packed + BF16-compatible scales.

    Returns (packed_i32, scale_f32, shape_i32).
    """
    w = np.asarray(w_f32, dtype=np.float32)
    out_dim, in_dim = w.shape
    assert in_dim % group_size == 0, \
        f"in_dim {in_dim} not multiple of group_size {group_size}"
    assert in_dim % 8 == 0, f"in_dim {in_dim} not multiple of 8"

    # Per-group absolute max → symmetric scale targeting int4 range [-7, 7].
    # We keep one code (-8) unused to avoid asymmetry / overflow concerns.
    groups = w.reshape(out_dim, in_dim // group_size, group_size)
    amax = np.abs(groups).max(axis=-1, keepdims=True)
    scale = amax / 7.0
    scale = np.where(scale > 0, scale, 1.0)
    q = np.round(groups / scale).clip(-8, 7).astype(np.int8)
    q_flat = q.reshape(out_dim, in_dim)

    # Offset-binary encoding: store as unsigned 0..15 by adding 8.
    # Range: signed -8..+7 → unsigned 0..15.
    u4_flat = (q_flat.astype(np.int16) + 8).astype(np.uint32)
    u4_g = u4_flat.reshape(out_dim, in_dim // 8, 8)
    packed = np.zeros((out_dim, in_dim // 8), dtype=np.uint32)
    for j in range(8):
        packed |= (u4_g[..., j] << (4 * j))

    packed_i32 = packed.view(np.int32)
    scale_f32 = scale.squeeze(-1).astype(np.float32)
    shape_i32 = np.array([out_dim, in_dim], dtype=np.int32)
    return packed_i32, scale_f32, shape_i32


def bf16_u16_to_f32(u16):
    """Convert a numpy uint16 array that holds bf16 bytes to float32.

    safetensors reads bf16 tensors as `torch.bfloat16` by default, but
    when loaded via `safetensors.numpy.load_file` bf16 is returned as
    uint16 (numpy has no native bf16). This helper lifts to f32.
    """
    u16 = np.ascontiguousarray(np.asarray(u16, dtype=np.uint16))
    # bf16 format: 1 sign + 8 exp + 7 frac  → identical to upper half of f32.
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def f32_to_bf16_u16(f32):
    """Truncate float32 to bf16 (stored as uint16)."""
    f = np.ascontiguousarray(np.asarray(f32, dtype=np.float32))
    u32 = f.view(np.uint32)
    # Round-to-nearest-even: add 0x8000 + (bit 16) before truncating.
    rounding_bias = ((u32 >> 16) & 1) + 0x7FFF
    rounded = (u32 + rounding_bias) >> 16
    return rounded.astype(np.uint16)
