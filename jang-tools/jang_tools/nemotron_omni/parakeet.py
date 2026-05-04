"""Native MLX implementation of the Parakeet Conformer encoder.

Mirrors `transformers.ParakeetEncoder` for the Nemotron-3-Nano-Omni-30B audio
path. Architecture (per the safetensors keys + transformers source):

  - `subsampling`: 3-stage 2-D conv subsampling (factor=8) + Linear projection
      layers.0  Conv2d 1   → 256, kernel=3×3, stride=2, padding=1
      layers.1  ReLU
      layers.2  Conv2d 256 → 256, kernel=3×3, stride=1, padding=1
      layers.3  Conv2d 256 → 256, kernel=1×1 (depthwise project)
      layers.4  ReLU
      layers.5  Conv2d 256 → 256, kernel=3×3, stride=2, padding=1
      layers.6  Conv2d 256 → 256, kernel=1×1
      layers.7  ReLU
      layers.8  Conv2d 256 → 256, kernel=3×3, stride=2, padding=1
      linear    Linear 4096 → hidden=1024  (4096 = 16 mel-frames × 256 ch)

  - 24× ConformerBlock:
      ½ × FF + LN     (linear1 1024→4096, SiLU, linear2 4096→1024, residual ×0.5)
      MHA + LN         (q/k/v/o projections + relative_k_proj for rel-pos, bias_u, bias_v)
      Conv module + LN (pointwise_conv1 1024→2048, GLU, depthwise_conv 1024×9
                        kernel + BN, SiLU, pointwise_conv2 1024→1024)
      ½ × FF + LN     (same as first)
      LayerNorm        (final)

Output: (batch, n_frames, 1024) — feeds into sound_projection.
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


# ── Subsampling ────────────────────────────────────────────────────────────

class ParakeetSubsampling(nn.Module):
    """3-stage Conv2d subsampling with factor=8 (matches transformers
    `ParakeetEncoderSubsamplingConv2D`).

    Layers:
        0  Conv2d(1, 256, k=3, s=2)        — full conv
        1  ReLU                             — (no params)
        2  Conv2d(256, 256, k=3, s=2, groups=256)  — depthwise
        3  Conv2d(256, 256, k=1)            — pointwise
        4  ReLU
        5  Conv2d(256, 256, k=3, s=2, groups=256)  — depthwise
        6  Conv2d(256, 256, k=1)            — pointwise
        7  ReLU
        linear  Linear(4096, hidden)        — flatten 256 ch × 16 mel bins

    Mel axis: 128 → 64 → 32 → 16 (3× stride-2). Time axis: T → T/8.
    """

    def __init__(self, hidden: int = 1024, channels: int = 256):
        super().__init__()
        self.layers_0 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1)
        # layers_1 = ReLU (no params)
        self.layers_2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1, groups=channels,
        )
        self.layers_3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        # layers_4 = ReLU
        self.layers_5 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=2, padding=1, groups=channels,
        )
        self.layers_6 = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        # layers_7 = ReLU
        self.linear = nn.Linear(channels * 16, hidden)

    def __call__(self, mel: mx.array) -> mx.array:
        """mel: (B, n_frames, n_mels=128) → (B, n_subsampled_frames, hidden=1024).

        IMPORTANT: PyTorch reference treats `n_frames` as the H axis and
        `n_mels` as the W axis (input shape after `unsqueeze(1)` is
        `(B, C=1, T, M)`). After convs the layout is `(B, C=256, T/8, M/8)`;
        the final flatten before the linear projector concatenates `(C, M)`
        in that order — NOT `(M, C)`. We mirror exactly: transpose to NCHW
        layout effectively by routing T as H and M as W.
        """
        B, T, M = mel.shape
        # MLX Conv2d input format: (B, H, W, C_in). Match PyTorch by using
        # T as H (rows) and M as W (cols), single channel C_in=1.
        x = mel[..., None]  # (B, T, M, 1)
        x = nn.relu(self.layers_0(x))      # (B, T/2, M/2, 256)
        x = self.layers_2(x)                # (B, T/4, M/4, 256) — depthwise
        x = nn.relu(self.layers_3(x))       # (B, T/4, M/4, 256) — pointwise
        x = self.layers_5(x)                # (B, T/8, M/8, 256) — depthwise
        x = nn.relu(self.layers_6(x))       # (B, T/8, M/8, 256) — pointwise
        # x is now (B, T_sub, M_sub, C). PyTorch's flatten order after the
        # transpose(1,2) is (C, M_sub) per time-step, so we transpose to
        # (B, T_sub, C, M_sub) and flatten the last two.
        B2, T_sub, M_sub, C = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(B2, T_sub, C * M_sub)
        x = self.linear(x)
        return x


# ── Conformer FF (half) ────────────────────────────────────────────────────

class ConformerFeedForward(nn.Module):
    """Macaron-style half feed-forward: Linear → SiLU → Linear."""

    def __init__(self, dim: int = 1024, hidden: int = 4096):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden, bias=False)
        self.linear2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


# ── Multi-head self-attention with relative positional encoding ────────────

def _build_rel_pos_embeddings(
    seq_len: int, hidden_size: int, base: float = 10000.0,
) -> mx.array:
    """Sinusoidal relative-position embeddings, mirroring
    `transformers.ParakeetEncoderRelPositionalEncoding`.

    Returns: (1, 2*seq_len-1, hidden_size) where positions go from
    `seq_len-1` down to `-(seq_len-1)` (inclusive).
    """
    half = hidden_size // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, hidden_size, 2, dtype=mx.float32) / hidden_size))
    # position_ids: [seq_len-1, seq_len-2, ..., 0, -1, ..., -(seq_len-1)]
    position_ids = mx.arange(seq_len - 1, -seq_len, -1, dtype=mx.float32)  # (2*seq_len-1,)
    # freqs: (2*seq_len-1, half)
    freqs = position_ids[:, None] * inv_freq[None, :]
    sin = mx.sin(freqs)
    cos = mx.cos(freqs)
    # Interleave sin and cos along the last dim → (2*seq_len-1, hidden_size)
    pos_embed = mx.stack([sin, cos], axis=-1).reshape(2 * seq_len - 1, hidden_size)
    return pos_embed[None, :, :]  # (1, 2*seq_len-1, hidden_size)


def _rel_shift(scores: mx.array, seq_len: int) -> mx.array:
    """Transformer-XL relative-position shift.

    Input:  (B, H, T, 2T-1)  scores from Q · R^T
    Output: (B, H, T, T)     causally-aligned relative scores
    """
    B, H, T, _ = scores.shape
    # Pad with one zero column on the left → (B, H, T, 2T)
    zeros = mx.zeros((B, H, T, 1), dtype=scores.dtype)
    scores = mx.concatenate([zeros, scores], axis=-1)
    # Reshape and slice (the "skewing" trick)
    scores = scores.reshape(B, H, 2 * T, T)
    scores = scores[:, :, 1:, :]               # (B, H, 2T-1, T)
    scores = scores.reshape(B, H, T, 2 * T - 1)
    # Take only the first T columns
    scores = scores[..., :seq_len]
    return scores


class RelativeMultiHeadAttention(nn.Module):
    """Full Transformer-XL relative-position attention.

    Mirrors `transformers.ParakeetEncoderAttention`:
      • content-content + content-position score = (Q + bias_u) · K^T
      • position-content + position-position score = (Q + bias_v) · R^T,
        then `_rel_shift` to align relative offsets, then take first T cols
      • Final attention: softmax(QK^T·s + matrix_bd) · V

    Args:
        dim: hidden size (1024 for parakeet)
        num_heads: 8 for parakeet
    """

    def __init__(self, dim: int = 1024, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 128
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.relative_k_proj = nn.Linear(dim, dim, bias=False)
        self.bias_u = mx.zeros((num_heads, self.head_dim))
        self.bias_v = mx.zeros((num_heads, self.head_dim))

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, T, D = x.shape
        H = self.num_heads
        Hd = self.head_dim

        # Projections — shape (B, H, T, Hd) after transpose
        q = self.q_proj(x).reshape(B, T, H, Hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, Hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, H, Hd).transpose(0, 2, 1, 3)

        # Build sinusoidal rel-pos embeddings: (1, 2T-1, D)
        pos_embed = _build_rel_pos_embeddings(T, D)
        # Project to relative-key space: (1, 2T-1, D) → (1, 2T-1, H, Hd)
        rel_k = self.relative_k_proj(pos_embed).reshape(1, 2 * T - 1, H, Hd)

        # Term (b)+(d): (Q + bias_v) · R^T
        # (Q+v) shape (B, H, T, Hd); rel_k.permute(0,2,3,1) shape (1, H, Hd, 2T-1)
        q_with_v = q + self.bias_v[None, :, None, :]                # (B, H, T, Hd)
        rel_k_t = rel_k.transpose(0, 2, 3, 1)                       # (1, H, Hd, 2T-1)
        matrix_bd = q_with_v @ rel_k_t                              # (B, H, T, 2T-1)
        matrix_bd = _rel_shift(matrix_bd, T)                        # (B, H, T, T)
        matrix_bd = matrix_bd * self.scale

        # Term (a)+(c): (Q + bias_u) · K^T (handled inside SDPA via scale)
        q_with_u = q + self.bias_u[None, :, None, :]                # (B, H, T, Hd)
        # Pass matrix_bd as additive mask so it sums into attention scores
        # before softmax.
        if mask is not None:
            matrix_bd = matrix_bd + mask
        out = mx.fast.scaled_dot_product_attention(
            q_with_u, k, v, scale=self.scale, mask=matrix_bd,
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return self.o_proj(out)


# ── Conformer Conv module ──────────────────────────────────────────────────

class ConformerConvModule(nn.Module):
    """Pointwise → GLU → Depthwise (kernel=9) → BN → SiLU → Pointwise."""

    def __init__(self, dim: int = 1024, kernel_size: int = 9):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1, bias=False)
        # depthwise conv: groups=dim, but MLX Conv1d may not have groups arg —
        # implement manually via per-channel gather.
        self.kernel_size = kernel_size
        self.depthwise_conv_weight = mx.zeros((dim, 1, kernel_size))
        self.norm = BatchNorm1d(dim)  # tracks running mean/var
        self.pointwise_conv2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.dim = dim

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D) — Conv1d expects (B, T, C_in) channels-last in MLX
        # but pointwise has kernel=1, so it's just a per-frame matmul.
        x = self.pointwise_conv1(x)  # (B, T, 2*D)
        # GLU split + sigmoid gating
        a, b = mx.split(x, 2, axis=-1)
        x = a * mx.sigmoid(b)         # (B, T, D)
        # Depthwise causal conv: pad+slide kernel=9 along T per channel.
        x = self._depthwise(x)
        x = self.norm(x)
        x = nn.silu(x)
        x = self.pointwise_conv2(x)
        return x

    def _depthwise(self, x: mx.array) -> mx.array:
        """Manual depthwise 1-D conv: each of D channels has its own k=9 kernel."""
        B, T, D = x.shape
        K = self.kernel_size
        pad = (K - 1) // 2  # symmetric padding
        x_pad = mx.pad(x, [(0, 0), (pad, pad), (0, 0)])
        # weight shape (D, 1, K) → reshape for elementwise per-channel mult
        w = self.depthwise_conv_weight.reshape(D, K)  # (D, K)
        # Build (B, T, D, K) by sliding
        out = mx.zeros((B, T, D))
        for i in range(K):
            out = out + x_pad[:, i:i + T, :] * w[None, None, :, i]
        return out


class BatchNorm1d(nn.Module):
    """Inference-only BatchNorm1d using stored running mean/var.

    Source weights:
      norm.weight        γ (D,)
      norm.bias          β (D,)
      norm.running_mean  μ (D,)
      norm.running_var   σ² (D,)

    Forward: y = γ * (x - μ) / sqrt(σ² + eps) + β
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.bias = mx.zeros((dim,))
        self.running_mean = mx.zeros((dim,))
        self.running_var = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, D) → normalize per-channel
        return (x - self.running_mean) / mx.sqrt(self.running_var + self.eps) \
               * self.weight + self.bias


# ── Conformer Block ────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """Macaron-style Conformer: ½FF + Attn + Conv + ½FF + LN."""

    def __init__(self, dim: int = 1024, num_heads: int = 8,
                 ff_hidden: int = 4096, conv_kernel: int = 9):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(dim)
        self.feed_forward1 = ConformerFeedForward(dim, ff_hidden)

        self.norm_self_att = nn.LayerNorm(dim)
        self.self_attn = RelativeMultiHeadAttention(dim, num_heads)

        self.norm_conv = nn.LayerNorm(dim)
        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel)

        self.norm_feed_forward2 = nn.LayerNorm(dim)
        self.feed_forward2 = ConformerFeedForward(dim, ff_hidden)

        self.norm_out = nn.LayerNorm(dim)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        # ½ × FF residual
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        # MHA residual
        x = x + self.self_attn(self.norm_self_att(x), mask=mask)
        # Conv residual
        x = x + self.conv(self.norm_conv(x))
        # ½ × FF residual
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        # Final LN
        x = self.norm_out(x)
        return x


# ── Full Encoder ───────────────────────────────────────────────────────────

class ParakeetEncoder(nn.Module):
    """Full Parakeet Conformer encoder (subsampling + 24 blocks)."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 8,
        ff_hidden: int = 4096,
        conv_kernel: int = 9,
        n_mels: int = 128,
    ):
        super().__init__()
        self.subsampling = ParakeetSubsampling(hidden_size)
        self.layers = [
            ConformerBlock(hidden_size, num_heads, ff_hidden, conv_kernel)
            for _ in range(num_layers)
        ]

    def __call__(
        self, mel_features: mx.array, attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """mel_features: (B, n_frames, n_mels) → (B, n_subsampled, hidden)."""
        x = self.subsampling(mel_features)
        for layer in self.layers:
            # NOTE: full attention_mask wiring deferred — short audio clips
            # don't need it; long-form ASR will require it.
            x = layer(x, mask=None)
        return x


def map_parakeet_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap on-disk `sound_encoder.encoder.*` keys to our nn.Module attribute names."""
    rename: dict[str, str] = {}
    for k in weights:
        if not k.startswith("sound_encoder.encoder."):
            continue
        suffix = k[len("sound_encoder.encoder."):]

        # Skip the inline featurizer (we use our own audio_features.py)
        if suffix.startswith("feature_extractor."):
            continue

        # Subsampling: layers.{N}.{weight,bias} → subsampling.layers_{N}.{weight,bias}
        if suffix.startswith("subsampling.layers."):
            inner = suffix[len("subsampling.layers."):]
            n, rest = inner.split(".", 1)
            rename[k] = f"subsampling.layers_{n}.{rest}"
            continue
        if suffix.startswith("subsampling.linear."):
            rename[k] = "subsampling.linear." + suffix[len("subsampling.linear."):]
            continue

        # Conformer blocks: layers.N.…
        if suffix.startswith("layers."):
            rename[k] = suffix
            continue
    out = {}
    for ok, nk in rename.items():
        v = weights[ok]
        # Fix the depthwise conv weight reshape: source shape (1024, 1, 9),
        # we store as (1024, 1, 9) too — but our class uses
        # `depthwise_conv_weight` not `depthwise_conv.weight`.
        nk = nk.replace("conv.depthwise_conv.weight", "conv.depthwise_conv_weight")
        # BN: norm.num_batches_tracked is a counter — drop
        if nk.endswith(".num_batches_tracked"):
            continue
        # MLX Conv layer weight ordering: (out_channels, kernel_h, kernel_w, in_channels)
        # PyTorch ordering: (out_channels, in_channels, kernel_h, kernel_w).
        # Transpose for subsampling Conv2d weights (4-D shape, OIHW → OHWI).
        if "subsampling.layers_" in nk and nk.endswith(".weight") and v.ndim == 4:
            import mlx.core as _mx
            v = _mx.transpose(v, (0, 2, 3, 1))
        # Conv1d (pointwise): (out_channels, in_channels, kernel) → (out_channels, kernel, in_channels)
        if (("pointwise_conv" in nk) and nk.endswith(".weight") and v.ndim == 3):
            import mlx.core as _mx
            v = _mx.transpose(v, (0, 2, 1))
        out[nk] = v
    return out
