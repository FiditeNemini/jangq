"""JANG-DFlash drafter: 5-layer block-diffusion transformer with KV injection.

Architecture follows DFlash (arXiv 2602.06036) and BD3-LM (arXiv 2503.09573):
- Absorbing-state masked diffusion (MaskGIT-style), 1 denoising step at inference
- Target hidden states injected directly into each drafter attention layer's
  K and V cache (NOT concatenated to input embeddings)
- Weighted masked CE loss with exponential decay: w_k = exp(-(k-1)/gamma)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import JangDFlashConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)      # [L, head_dim]
    # Broadcast over batch and head axes: [1, L, 1, head_dim]
    cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(2)
    sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(2)
    return cos, sin


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class JangDFlashAttention(nn.Module):
    """Attention with KV injection.

    The block's own token-derived K/V (from ``wk``/``wv``) and the injected
    context K/V (from ``wk_ctx``/``wv_ctx``) are concatenated on the sequence
    axis. Block positions are causal among themselves and attend bidirectionally
    to the full context. RoPE applies only to the block positions; the injected
    context carries its positional information baked in via the fusion MLP.
    """

    def __init__(self, cfg: JangDFlashConfig):
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        q_out = cfg.num_heads * cfg.head_dim
        kv_out = cfg.num_kv_heads * cfg.head_dim
        self.wq = nn.Linear(cfg.hidden_dim, q_out, bias=False)
        self.wk = nn.Linear(cfg.hidden_dim, kv_out, bias=False)
        self.wv = nn.Linear(cfg.hidden_dim, kv_out, bias=False)
        self.wo = nn.Linear(q_out, cfg.hidden_dim, bias=False)
        self.wk_ctx = nn.Linear(cfg.hidden_dim, kv_out, bias=False)
        self.wv_ctx = nn.Linear(cfg.hidden_dim, kv_out, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, h_ctx_kv: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        _, T_ctx, _ = h_ctx_kv.shape

        q = self.wq(x).view(B, L, self.num_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        k_ctx = self.wk_ctx(h_ctx_kv).view(B, T_ctx, self.num_kv_heads, self.head_dim)
        v_ctx = self.wv_ctx(h_ctx_kv).view(B, T_ctx, self.num_kv_heads, self.head_dim)
        k_ctx = self.k_norm(k_ctx)

        cos, sin = _build_rope_cache(L, self.head_dim, self.cfg.rope_theta, x.device, x.dtype)
        q, k = _apply_rope(q, k.contiguous(), cos, sin)

        k_full = torch.cat([k_ctx, k], dim=1)
        v_full = torch.cat([v_ctx, v], dim=1)

        rep = self.num_heads // self.num_kv_heads
        k_full = k_full.repeat_interleave(rep, dim=2)
        v_full = v_full.repeat_interleave(rep, dim=2)

        mask = torch.zeros(L, T_ctx + L, device=x.device, dtype=x.dtype)
        causal = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device), diagonal=1
        )
        mask[:, T_ctx:] = causal.to(x.dtype)

        q_ = q.transpose(1, 2)
        k_ = k_full.transpose(1, 2)
        v_ = v_full.transpose(1, 2)

        out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class JangDFlashFFN(nn.Module):
    def __init__(self, cfg: JangDFlashConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)
        self.w3 = nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class JangDFlashBlock(nn.Module):
    def __init__(self, cfg: JangDFlashConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.attn = JangDFlashAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.ffn = JangDFlashFFN(cfg)

    def forward(self, x: torch.Tensor, h_ctx_kv: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), h_ctx_kv)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class JangDFlashDrafter(nn.Module):
    def __init__(self, cfg: JangDFlashConfig):
        super().__init__()
        self.cfg = cfg
        # Vocab extended by 1 to hold MASK token at index vocab_size.
        self.embed = nn.Embedding(cfg.vocab_size + 1, cfg.hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(cfg.tap_dim, cfg.hidden_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim, bias=False),
            RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps),
        )
        self.layers = nn.ModuleList([JangDFlashBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_dim, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(
        self,
        block_ids: torch.Tensor,
        h_taps: torch.Tensor | None = None,
        h_ctx_kv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One-step masked-block forward.

        block_ids: [B, L] int64, anchor at position 0, MASK at 1..L-1
        h_taps:    [B, T_ctx, tap_dim] — raw concat of tap-layer hiddens
        h_ctx_kv:  [B, T_ctx, hidden_dim] — pre-fused (skip fusion MLP)

        Exactly one of h_taps / h_ctx_kv must be provided.
        """
        if (h_taps is None) == (h_ctx_kv is None):
            raise ValueError("pass exactly one of h_taps or h_ctx_kv")
        if h_ctx_kv is None:
            h_ctx_kv = self.fusion_mlp(h_taps)

        x = self.embed(block_ids)
        for layer in self.layers:
            x = layer(x, h_ctx_kv)
        return self.lm_head(self.norm(x))


def dflash_loss(
    logits: torch.Tensor, targets: torch.Tensor, cfg: JangDFlashConfig
) -> torch.Tensor:
    """Weighted masked cross-entropy (DFlash Eq. 4).

    logits:  [B, L, V]
    targets: [B, L]

    Position 0 is the anchor (always clean), skipped in the loss.
    Weights w_k = exp(-(k - 1) / gamma) for k in 1..L-1.
    """
    B, L, V = logits.shape
    if L != cfg.block_size:
        raise ValueError(f"expected block size {cfg.block_size}, got {L}")

    pred = logits[:, 1:, :].reshape(-1, V)
    tgt = targets[:, 1:].reshape(-1)
    ks = torch.arange(1, L, device=logits.device, dtype=logits.dtype)
    w = torch.exp(-(ks - 1) / cfg.loss_gamma)
    w = w.unsqueeze(0).expand(B, -1).reshape(-1)
    per_tok = F.cross_entropy(pred, tgt, reduction="none")
    return (per_tok * w).sum() / w.sum()
