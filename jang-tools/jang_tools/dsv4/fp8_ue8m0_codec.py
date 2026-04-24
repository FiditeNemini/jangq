"""FP8 e4m3fn with UE8M0 block scales (DSV4 non-expert weights).

Input format per DSV4-Flash:
  weight: float8_e4m3fn, shape (out_dim, in_dim)
  scale:  float8_e8m0fnu (UE8M0), shape (out_dim // 128, in_dim // 128)
          each scale covers a 128 x 128 block.

Dequant:
  weight_bf16 = fp8_e4m3_to_bf16(w_fp8) * ue8m0_to_fp32(scale_block)
"""

from __future__ import annotations

import torch


def _ue8m0_to_fp32(s: torch.Tensor) -> torch.Tensor:
    return s.float()


def dequant_fp8_ue8m0_blockwise(
    w_fp8: torch.Tensor,
    scale_ue8m0: torch.Tensor,
    *,
    fp8_block: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a FP8 e4m3fn matrix to bf16 (or fp32) via UE8M0 block scales.

    Arguments
    ---------
    w_fp8 : float8_e4m3fn tensor, shape (out_dim, in_dim)
    scale_ue8m0 : float8_e8m0fnu tensor, shape (out_dim // b0, in_dim // b1)
    fp8_block : block size (default (128, 128) per DSV4 spec)
    """
    assert w_fp8.dtype == torch.float8_e4m3fn, \
        f"expected float8_e4m3fn, got {w_fp8.dtype}"
    assert w_fp8.ndim == 2
    out_dim, in_dim = w_fp8.shape
    b0, b1 = fp8_block
    assert out_dim % b0 == 0 and in_dim % b1 == 0, \
        f"shape {(out_dim, in_dim)} not divisible by block {fp8_block}"
    assert scale_ue8m0.shape == (out_dim // b0, in_dim // b1), \
        f"scale shape mismatch: {scale_ue8m0.shape} vs " \
        f"({out_dim // b0}, {in_dim // b1})"

    w_fp32 = w_fp8.float()  # torch handles fp8 decode
    scale_fp32 = _ue8m0_to_fp32(scale_ue8m0)  # (nb0, nb1)
    # Expand scale to per-element
    scale_expanded = scale_fp32.repeat_interleave(b0, dim=0).repeat_interleave(b1, dim=1)
    out = w_fp32 * scale_expanded

    return out.to(out_dtype)
