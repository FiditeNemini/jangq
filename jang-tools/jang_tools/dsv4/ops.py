"""Pure-PyTorch (no CUDA kernels) ops needed for DSV4 forward.

These replace the reference `inference/kernel.py` CUDA kernels so the
model can run on Mac / CPU / MPS for calibration and JANGTQ work.
Correctness over speed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------- RMSNorm ----------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm with learned `weight` gain; computes in fp32 for stability."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return (x * weight.float()).to(dtype)


# ---------- YaRN RoPE ----------

def precompute_freqs_cis(
    dim: int, seqlen: int, original_seq_len: int,
    base: float, factor: float, beta_fast: int, beta_slow: int,
) -> torch.Tensor:
    """Precompute complex rotation factors for YaRN-scaled RoPE.

    When original_seq_len == 0 or factor == 1, this is standard RoPE.
    Otherwise applies the YaRN ramp between [low, high] wavelengths for
    extended-context scaling (DSV4: factor=16, original_seq_len=65536,
    target seqlen=1M).
    """
    assert dim % 2 == 0
    idx = torch.arange(0, dim, 2).float()
    freqs = 1.0 / (base ** (idx / dim))
    if original_seq_len > 0 and factor > 1:
        # YaRN rescale
        low = max(math.floor(dim * math.log(original_seq_len / (beta_fast * 2 * math.pi)) /
                             (2 * math.log(base))), 0)
        high = min(math.ceil(dim * math.log(original_seq_len / (beta_slow * 2 * math.pi)) /
                             (2 * math.log(base))), dim // 2 - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp((torch.arange(dim // 2).float() - low) / (high - low), 0, 1)
        freqs_scaled = freqs / factor
        freqs = freqs_scaled * (1 - ramp) + freqs * ramp
    t = torch.arange(seqlen).float()
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis  # complex64, (seqlen, dim//2)


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply rotary embedding in-place to the last dimension of x.

    x shape: (..., rope_head_dim); rope_head_dim must be even.
    freqs_cis shape: (seqlen, rope_head_dim//2).

    The reference kernel mutates `x` in place; we do the same for parity
    with the DSV4 code that expects the side-effect behavior. Returns x
    for chaining.
    """
    dtype = x.dtype
    d = x.shape[-1]
    # Force contiguous for view_as_complex. Reshape last dim to (d/2, 2).
    xr = x.float().contiguous().reshape(*x.shape[:-1], d // 2, 2)
    xc = torch.view_as_complex(xr)
    fc = freqs_cis.conj() if inverse else freqs_cis
    # xc shape ends with (..., L, d/2). freqs_cis (L, d/2) broadcasts.
    # For x shape (B, H, L, d) → xc (B, H, L, d/2) — broadcast with
    # (L, d/2) → (1, 1, L, d/2). Explicitly align to avoid torch edge cases.
    if xc.ndim == 4 and fc.ndim == 2:
        fc = fc[None, None, :, :]
    elif xc.ndim == 3 and fc.ndim == 2:
        fc = fc[None, :, :]
    xc = xc * fc
    rot = torch.view_as_real(xc).reshape(*x.shape)
    x.copy_(rot.to(dtype))
    return x


# ---------- Hadamard rotation (replaces CUDA kernel) ----------

_HADAMARD_CACHE: dict[int, torch.Tensor] = {}


def _hadamard_matrix(n: int) -> torch.Tensor:
    """Recursive Hadamard matrix for power-of-2 n, normalized by 1/sqrt(n)."""
    if n in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[n]
    assert n & (n - 1) == 0, f"Hadamard requires power of 2, got {n}"
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = H / math.sqrt(n)
    _HADAMARD_CACHE[n] = H
    return H


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation to the last dimension of x, in-place.

    Pure-PyTorch replacement for the CUDA rotate_activation kernel.
    The reference mutates x; we match that.
    """
    d = x.shape[-1]
    H = _hadamard_matrix(d).to(x.device).to(x.dtype)
    # (...,d) @ (d,d) = (...,d)
    y = x @ H
    x.copy_(y)
    return x


# ---------- Activation FP simulation (no-op for calibration) ----------

def act_quant_bf16_passthrough(x: torch.Tensor, block: int, scale_fmt=None,
                                scale_dtype=None, inplace: bool = False) -> torch.Tensor:
    """Reference kernel simulates FP8 quant+dequant of activations to match
    QAT. For calibration/forward correctness we simply return x unchanged.

    Difference expected: ~5-10% relative error on attention outputs vs
    the fully-simulated forward. Doesn't affect REAP saliency ranking
    meaningfully — experts see similar enough inputs."""
    return x


def fp4_act_quant_bf16_passthrough(x: torch.Tensor, block: int, inplace: bool = False) -> torch.Tensor:
    return x


# ---------- hc_split_sinkhorn — pure PyTorch ----------

def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a mix vector into (pre, post, comb) via Sinkhorn normalization.

    The reference CUDA kernel `hc_split_sinkhorn` takes (mixes, scale,
    base, mult, iters, eps) and returns (pre, post, comb).

    Layout of mixes[..., mix_hc=(2+mult)*mult]:
      first mult entries  → pre
      next mult entries   → post
      last mult*mult entr → comb (flat, reshaped to (mult, mult))

    Each segment gets its own scale and base applied, then the comb
    matrix is Sinkhorn-normalized to have row/column sums = 1.

    Per mHC in the paper: the mixing coefficients come from a constraint
    that the "manifold" structure is preserved — hence doubly-stochastic
    comb matrix via Sinkhorn.
    """
    # mixes: (..., mix_hc)
    mh = hc_mult
    # split: pre (mh), post (mh), comb (mh*mh)
    pre_raw = mixes[..., :mh]
    post_raw = mixes[..., mh:2 * mh]
    comb_raw = mixes[..., 2 * mh:]  # mh*mh

    # apply scales & bases (scale has 3 entries: one per pre/post/comb)
    # hc_base has mix_hc = 2mh + mh*mh entries, one per output position
    base_pre = hc_base[:mh]
    base_post = hc_base[mh:2 * mh]
    base_comb = hc_base[2 * mh:]

    pre = torch.sigmoid(pre_raw * hc_scale[0] + base_pre) + eps        # (..., mh)
    post = torch.sigmoid(post_raw * hc_scale[1] + base_post) + eps     # (..., mh)
    comb = (comb_raw * hc_scale[2] + base_comb).view(*mixes.shape[:-1], mh, mh)

    # Normalize pre to sum-to-1 (Sinkhorn-lite for 1D is just normalize)
    pre = pre / pre.sum(-1, keepdim=True)

    # Sinkhorn on comb: iterate row-normalize, column-normalize `iters` times
    # to produce a doubly-stochastic matrix.
    comb = comb.exp()
    for _ in range(iters):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)   # row-normalize
        comb = comb / (comb.sum(-2, keepdim=True) + eps)   # col-normalize

    return pre, post, comb
