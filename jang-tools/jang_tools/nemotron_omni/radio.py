"""Native MLX implementation of the RADIO ViT vision tower.

Mirrors `nvidia/C-RADIOv2-H` (radio_v2.5-h variant) for the
Nemotron-3-Nano-Omni-30B model:

  - Standard timm vit_huge_patch16_224 body (32 blocks, embed_dim=1280, 16 heads)
  - NVIDIA's CPE patch_generator replacing patch_embed/cls_token/pos_embed
  - Bilinear pos_embed interpolation for non-default input dims (eval mode)
  - 10 cls/register tokens (num_cls_tokens=1 + 9 registers from register_multiple)
  - Final norm = Identity (timm strips it via `model.norm = nn.Identity()`)

Usage:
    from jang_tools.nemotron_omni.radio import RADIOVisionModel
    rm = RADIOVisionModel.from_safetensors(bundle_path)
    feats = rm(pixel_values_mx)          # (B, num_patches, 1280)

Tensor naming on disk:
  vision_model.radio_model.input_conditioner.norm_mean        (3, 1, 1)
  vision_model.radio_model.input_conditioner.norm_std         (3, 1, 1)
  vision_model.radio_model.model.patch_generator.cls_token.token       (10, 1280)
  vision_model.radio_model.model.patch_generator.embedder.weight        (1280, 768)
  vision_model.radio_model.model.patch_generator.pos_embed              (1, 16384, 1280)
  vision_model.radio_model.model.patch_generator.video_embedder.weight  (1280, 1536)
  vision_model.radio_model.model.blocks.{0..31}.attn.qkv.{weight,bias}  (3840, 1280) (3840,)
  vision_model.radio_model.model.blocks.{0..31}.attn.proj.{weight,bias} (1280, 1280) (1280,)
  vision_model.radio_model.model.blocks.{0..31}.norm{1,2}.{weight,bias} (1280,) (1280,)
  vision_model.radio_model.model.blocks.{0..31}.mlp.fc1.{weight,bias}   (5120, 1280) (5120,)
  vision_model.radio_model.model.blocks.{0..31}.mlp.fc2.{weight,bias}   (1280, 5120) (1280,)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def _bilinear_resize_2d(x: mx.array, target_h: int, target_w: int) -> mx.array:
    """Bilinear-interpolate (1, C, H, W) → (1, C, target_h, target_w).

    MLX has no built-in `F.interpolate`; we implement basic bilinear resize via
    grid sampling math. align_corners=False (matches the patched RADIO eval
    path and Megatron/vLLM convention).

    The pos_embed at our preferred resolution (512×512, patch=16) is 32×32,
    which is smaller than the stored 128×128 pos_embed. We bilinear-downsample.
    """
    _, C, H, W = x.shape
    if H == target_h and W == target_w:
        return x

    # Build sampling grid in [-1, 1] coords, align_corners=False
    # For align_corners=False: sample at grid spacing 2/target_dim with
    # offset (1/target_dim - 1) so first sample center is half-pixel from edge.
    def _grid(target: int, source: int) -> mx.array:
        # output index i ∈ [0, target-1] maps to source coordinate
        # (i + 0.5) * source / target - 0.5 (the OpenCV / PyTorch convention
        # for align_corners=False)
        idx = mx.arange(target, dtype=mx.float32)
        return (idx + 0.5) * source / target - 0.5

    src_y = _grid(target_h, H)        # (target_h,)
    src_x = _grid(target_w, W)        # (target_w,)

    # Clamp to valid integer range
    y0 = mx.floor(src_y).astype(mx.int32)
    y1 = y0 + 1
    x0 = mx.floor(src_x).astype(mx.int32)
    x1 = x0 + 1
    wy = src_y - y0.astype(mx.float32)
    wx = src_x - x0.astype(mx.float32)

    y0 = mx.clip(y0, 0, H - 1); y1 = mx.clip(y1, 0, H - 1)
    x0 = mx.clip(x0, 0, W - 1); x1 = mx.clip(x1, 0, W - 1)

    # Gather: (1, C, target_h, target_w) per corner
    # x has shape (1, C, H, W) — gather along H then W.
    # Use take which works on mx arrays.
    # We'll do row-wise gather:
    #   x[:, :, y, x] = x.take(y, axis=2).take(x, axis=3)
    def _gather(arr, ys, xs):
        # arr: (1,C,H,W); ys: (Th,); xs: (Tw,)
        return arr[..., ys, :][..., xs]

    f00 = _gather(x, y0, x0)
    f01 = _gather(x, y0, x1)
    f10 = _gather(x, y1, x0)
    f11 = _gather(x, y1, x1)

    wx_b = wx[None, None, None, :]  # (1,1,1,Tw)
    wy_b = wy[None, None, :, None]  # (1,1,Th,1)
    out = (
        f00 * (1 - wx_b) * (1 - wy_b)
        + f01 * wx_b * (1 - wy_b)
        + f10 * (1 - wx_b) * wy_b
        + f11 * wx_b * wy_b
    )
    return out


class InputConditioner(nn.Module):
    """Normalize pixel_values: (x - mean) / std with broadcast over (C, 1, 1).

    Note: when our `image_processor.preprocess_images` already normalizes,
    the InputConditioner can be replaced with `make_preprocessor_external`
    (ie. become identity). We default to applying it; callers override if
    they pre-normalize.
    """

    def __init__(self):
        super().__init__()
        self.norm_mean = mx.zeros((3, 1, 1))
        self.norm_std = mx.ones((3, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        # x shape (B, 3, H, W)
        return (x - self.norm_mean[None]) / self.norm_std[None]


class ViTPatchGenerator(nn.Module):
    """RADIO's CPE patch generator: Im2Patches + Linear + pos_embed + cls_token.

    The pos_embed is stored at max input resolution (1, 16384=128*128, 1280)
    and bilinear-interpolated to the actual input grid at forward time.
    """

    def __init__(
        self,
        *,
        patch_size: int = 16,
        embed_dim: int = 1280,
        num_cls_tokens: int = 10,
        max_grid: int = 128,  # 128*16 = 2048 max input
        video_temporal_patch: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
        self.max_grid = max_grid
        self.num_skip = num_cls_tokens  # alias used by the source code

        # patch_generator.embedder.weight (embed_dim, 3*P*P)
        self.embedder = nn.Linear(3 * patch_size * patch_size, embed_dim, bias=False)
        # video_embedder for T*3*P*P input (when stacked T frames)
        self.video_embedder = nn.Linear(
            video_temporal_patch * 3 * patch_size * patch_size, embed_dim, bias=False,
        )
        # ClsToken: shape (num_cls_tokens, embed_dim)
        self.cls_token = mx.zeros((num_cls_tokens, embed_dim))
        # pos_embed: (1, max_grid*max_grid, embed_dim)
        self.pos_embed = mx.zeros((1, max_grid * max_grid, embed_dim))

    def _im_to_patches(self, x: mx.array) -> mx.array:
        """(B, 3, H, W) → (B, num_patches, 3*P*P)."""
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, \
            f"Input dims {H}x{W} must be divisible by patch_size {P}"
        # rearrange b c (py p1) (px p2) -> b (py px) (c p1 p2)
        py = H // P
        px = W // P
        x = x.reshape(B, C, py, P, px, P)              # (B,C,py,P,px,P)
        x = x.transpose(0, 2, 4, 1, 3, 5)              # (B,py,px,C,P,P)
        x = x.reshape(B, py * px, C * P * P)
        return x

    def _get_pos_embed(self, input_h: int, input_w: int) -> mx.array:
        """Bilinear-interpolate pos_embed from (max_grid, max_grid) to (input_h, input_w)."""
        gy = input_h // self.patch_size
        gx = input_w // self.patch_size
        if (gy, gx) == (self.max_grid, self.max_grid):
            return self.pos_embed
        # Reshape pos_embed (1, max*max, D) → (1, D, max, max)
        pe = self.pos_embed.reshape(1, self.max_grid, self.max_grid, self.embed_dim)
        pe = pe.transpose(0, 3, 1, 2)
        # Eval-time CPE: interpolate to max(gy, gx) square then window-select
        max_dim = max(gy, gx)
        pe = _bilinear_resize_2d(pe, max_dim, max_dim)
        # Window-select to (gy, gx)
        pe = pe[:, :, :gy, :gx]
        # If still mismatched (rectangular target with max==gx but gy<max),
        # already handled by the slice above.
        # Flatten back to (1, gy*gx, D)
        pe = pe.transpose(0, 2, 3, 1).reshape(1, gy * gx, self.embed_dim)
        return pe

    def __call__(self, x: mx.array, *, video: bool = False) -> mx.array:
        B, _, H, W = x.shape
        patches = self._im_to_patches(x)                          # (B, N, 3*P*P)
        embedder = self.video_embedder if video else self.embedder
        patches = embedder(patches)                                # (B, N, embed_dim)
        pos = self._get_pos_embed(H, W)                            # (1, N, embed_dim)
        patches = patches + pos
        # Concat cls tokens at front
        cls = mx.broadcast_to(
            self.cls_token[None],
            (B, self.num_cls_tokens, self.embed_dim),
        )
        patches = mx.concatenate([cls, patches], axis=1)
        return patches


class ViTAttention(nn.Module):
    """Standard timm-style ViT attention block.

    Forward:
        qkv = Linear(x)  # 1280 → 3*1280
        q, k, v = split(qkv)
        out = softmax(QK^T / sqrt(d)) @ V
        out = Linear(out)  # 1280 → 1280
    """

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        # (3, B, num_heads, N, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=None,
        )
        # (B, num_heads, N, head_dim) → (B, N, dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class ViTMLP(nn.Module):
    """Standard timm-style ViT MLP: fc1 → GELU → fc2."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class ViTBlock(nn.Module):
    """Pre-norm transformer block: x = x + Attn(LN1(x)); x = x + MLP(LN2(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, int(dim * mlp_ratio))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class RADIOVisionModel(nn.Module):
    """Full RADIO ViT body for Nemotron-3-Nano-Omni.

    Skips the input_conditioner + adaptors. Returns the full (B, num_total, D)
    feature tensor — the wrapper code splits off cls tokens from patch tokens
    and applies pixel_shuffle + mlp1.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 1280,
        num_blocks: int = 32,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        num_cls_tokens: int = 10,
        max_grid: int = 128,
        video_temporal_patch: int = 2,
        apply_input_conditioner: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_cls_tokens = num_cls_tokens
        self.apply_input_conditioner = apply_input_conditioner

        if apply_input_conditioner:
            self.input_conditioner = InputConditioner()
        self.patch_generator = ViTPatchGenerator(
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_cls_tokens=num_cls_tokens,
            max_grid=max_grid,
            video_temporal_patch=video_temporal_patch,
        )
        self.blocks = [
            ViTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_blocks)
        ]
        # NOTE: timm sets `model.norm = nn.Identity()` via radio_model.py L329
        # so there is NO final LayerNorm. The `vit_huge_patch16_224` body's
        # norm weight is dropped — confirm by absence of `vision_model.radio_model.model.norm.*`
        # in the safetensors index.
        # (We also disable "summary_idxs" — that's for adaptors which Nemotron doesn't use.)

    def __call__(self, x: mx.array, *, video: bool = False) -> mx.array:
        """x: (B, 3, H, W) image tensor.

        Returns (B, num_cls + num_patches, embed_dim) feature tokens.
        Caller must split off cls tokens (first num_cls_tokens) before
        applying pixel_shuffle.
        """
        if self.apply_input_conditioner:
            x = self.input_conditioner(x)
        x = self.patch_generator(x, video=video)
        for block in self.blocks:
            x = block(x)
        return x


def map_radio_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Remap on-disk `vision_model.radio_model.*` keys → our nn.Module attribute names."""
    rename: dict[str, str] = {}
    for k in weights:
        if not k.startswith("vision_model.radio_model."):
            continue
        suffix = k[len("vision_model.radio_model."):]
        if suffix.startswith("input_conditioner."):
            rename[k] = "input_conditioner." + suffix[len("input_conditioner."):]
        elif suffix.startswith("model.patch_generator.cls_token.token"):
            rename[k] = "patch_generator.cls_token"
        elif suffix.startswith("model.patch_generator."):
            inner = suffix[len("model.patch_generator."):]
            rename[k] = f"patch_generator.{inner}"
        elif suffix.startswith("model.blocks."):
            inner = suffix[len("model."):]  # "blocks.N.…"
            rename[k] = inner
        elif suffix == "summary_idxs":
            # Buffer for adaptor head selection — not used in our path
            continue
        else:
            # Unknown key, skip
            continue
    return {nk: weights[ok] for ok, nk in rename.items()}


def pixel_shuffle(x: mx.array, scale_factor: float = 0.5) -> mx.array:
    """Pixel-shuffle for vision tokens.

    Input  (B, H, W, C) → Output (B, H*scale, W*scale, C/(scale**2))

    With scale_factor=0.5:
        (B, 32, 32, 1280) → (B, 16, 16, 5120)
    Then reshape (B, 256, 5120) → projector input.
    """
    B, H, W, C = x.shape
    # N, W, H, C -> N, W, H*scale, C//scale
    s = scale_factor
    x = x.reshape(B, W, int(H * s), int(C / s))
    # N, W, H*scale, C//scale -> N, H*scale, W, C//scale
    x = x.transpose(0, 2, 1, 3)
    # N, H*scale, W, C//scale -> N, H*scale, W*scale, C//(scale**2)
    x = x.reshape(B, int(H * s), int(W * s), int(C / (s * s)))
    # ps_version v2: swap spatial dims back
    x = x.transpose(0, 2, 1, 3)
    return x
