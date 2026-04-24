"""Pure-PyTorch DSV4-Flash forward for calibration.

Simplifications vs inference/model.py:
  - No CUDA kernels (sparse_attn, fp8_gemm, fp4_gemm, hc_split_sinkhorn)
    → replaced with pure PyTorch equivalents in ops.py.
  - No FP4/FP8 activation simulation (act_quant* are bf16 passthroughs).
  - No tensor parallelism (world_size=1).
  - Simplified attention: full attention over all KV positions (no
    sparse top-k, no window-mask, no compressor). For calibration REAP
    saliency this is fine — experts still see meaningful hidden state.
  - mHC residuals are fully implemented (essential for signal
    fidelity across layers).
  - Gate supports both hash (first n_hash_layers) and score-based
    (sqrtsoftplus + noaux_tc bias) routing, per real model.

This is Phase 7.5A. Phase 7.5B = full CSA/HCA/indexer port.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from jang_tools.dsv4.ops import (
    rms_norm,
    precompute_freqs_cis,
    apply_rotary_emb,
    hc_split_sinkhorn,
)


@dataclass
class DSV4Config:
    """Minimal config mirror — only fields we use for calibration."""
    vocab_size: int = 129280
    dim: int = 4096
    n_layers: int = 43
    n_hash_layers: int = 3
    n_mtp_layers: int = 1
    n_heads: int = 64
    n_kv_heads: int = 1
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    # MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    n_activated_experts: int = 6
    moe_inter_dim: int = 2048
    score_func: str = "sqrtsoftplus"
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    norm_topk_prob: bool = True
    # RoPE
    rope_theta: float = 10000.0
    rope_factor: float = 16.0
    original_seq_len: int = 65536
    compress_rope_theta: float = 160000.0
    beta_fast: int = 32
    beta_slow: int = 1
    # mHC
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # misc
    norm_eps: float = 1e-6
    window_size: int = 128
    routed_scaling_factor: float = 1.5

    @classmethod
    def from_config_json(cls, config_dict: dict) -> "DSV4Config":
        """Parse HF config.json into our local config."""
        return cls(
            vocab_size=config_dict["vocab_size"],
            dim=config_dict["hidden_size"],
            n_layers=config_dict["num_hidden_layers"],
            n_hash_layers=config_dict.get("num_hash_layers", 0),
            n_mtp_layers=config_dict.get("num_nextn_predict_layers", 1),
            n_heads=config_dict["num_attention_heads"],
            n_kv_heads=config_dict["num_key_value_heads"],
            head_dim=config_dict["head_dim"],
            rope_head_dim=config_dict["qk_rope_head_dim"],
            q_lora_rank=config_dict["q_lora_rank"],
            o_lora_rank=config_dict["o_lora_rank"],
            o_groups=config_dict["o_groups"],
            n_routed_experts=config_dict["n_routed_experts"],
            n_shared_experts=config_dict["n_shared_experts"],
            n_activated_experts=config_dict["num_experts_per_tok"],
            moe_inter_dim=config_dict["moe_intermediate_size"],
            score_func=config_dict.get("scoring_func", "sqrtsoftplus"),
            route_scale=config_dict.get("routed_scaling_factor", 1.5),
            swiglu_limit=config_dict.get("swiglu_limit", 10.0),
            norm_topk_prob=config_dict.get("norm_topk_prob", True),
            rope_theta=config_dict["rope_theta"],
            rope_factor=config_dict["rope_scaling"]["factor"],
            original_seq_len=config_dict["rope_scaling"]["original_max_position_embeddings"],
            compress_rope_theta=config_dict.get("compress_rope_theta", 160000.0),
            beta_fast=config_dict["rope_scaling"]["beta_fast"],
            beta_slow=config_dict["rope_scaling"]["beta_slow"],
            hc_mult=config_dict.get("hc_mult", 4),
            hc_sinkhorn_iters=config_dict.get("hc_sinkhorn_iters", 20),
            hc_eps=config_dict.get("hc_eps", 1e-6),
            norm_eps=config_dict.get("rms_norm_eps", 1e-6),
            window_size=config_dict.get("sliding_window", 128),
            routed_scaling_factor=config_dict.get("routed_scaling_factor", 1.5),
        )


# ---------- attention (simplified: full attention) ----------

class Attention(nn.Module):
    def __init__(self, cfg: DSV4Config):
        super().__init__()
        self.cfg = cfg
        self.wq_a = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.q_norm = nn.RMSNorm(cfg.q_lora_rank, eps=cfg.norm_eps, dtype=torch.bfloat16)
        self.wq_b = nn.Linear(cfg.q_lora_rank, cfg.n_heads * cfg.head_dim, bias=False, dtype=torch.bfloat16)
        self.wkv = nn.Linear(cfg.dim, cfg.head_dim, bias=False, dtype=torch.bfloat16)
        self.kv_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.norm_eps, dtype=torch.bfloat16)
        self.wo_a = nn.Linear(
            cfg.n_heads * cfg.head_dim // cfg.o_groups,
            cfg.o_groups * cfg.o_lora_rank,
            bias=False, dtype=torch.bfloat16,
        )
        self.wo_b = nn.Linear(cfg.o_groups * cfg.o_lora_rank, cfg.dim, bias=False, dtype=torch.bfloat16)
        self.attn_sink = nn.Parameter(torch.empty(cfg.n_heads, dtype=torch.float32))
        self.softmax_scale = cfg.head_dim ** -0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        cfg = self.cfg
        # Q low-rank
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (cfg.n_heads, cfg.head_dim)).transpose(1, 2)  # (B, H, L, d)
        # per-head RMSNorm (applied on last dim)
        dtype = q.dtype
        q_f = q.float()
        q = (q_f * torch.rsqrt(q_f.square().mean(-1, keepdim=True) + cfg.norm_eps)).to(dtype)
        apply_rotary_emb(q[..., -cfg.rope_head_dim:], freqs_cis)
        # KV: single head (B, L, D) → treat as B=1, L=L, H=1, D
        kv = self.kv_norm(self.wkv(x))  # (B, L, D)
        kv = kv.unsqueeze(1)  # (B, 1, L, D) — H=1
        apply_rotary_emb(kv[..., -cfg.rope_head_dim:], freqs_cis)
        # Broadcast single KV head to all n_heads
        k = kv.expand(-1, cfg.n_heads, -1, -1)  # (B, H, L, D)
        # Attention scores (B, H, sq, sk)
        scores = (q @ k.transpose(-1, -2)) * self.softmax_scale
        # causal mask
        mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=x.device), diagonal=1)
        scores = scores + mask
        # attn sink: prepend a virtual logit of attn_sink per head
        sink = self.attn_sink.view(1, cfg.n_heads, 1, 1).expand(bsz, -1, seqlen, 1).to(scores.dtype)
        scores = torch.cat([sink, scores], dim=-1)
        attn = scores.softmax(dim=-1)
        attn = attn[..., 1:]
        attn = attn.to(kv.dtype)
        v = k  # single KV head already broadcast
        o = attn @ v  # (B, H, L, d)
        # apply inverse RoPE while still in (B, H, L, d) layout
        apply_rotary_emb(o[..., -cfg.rope_head_dim:], freqs_cis, inverse=True)
        o = o.transpose(1, 2).contiguous()  # (B, L, H, d)
        # Grouped low-rank O
        o = o.view(bsz, seqlen, cfg.o_groups, -1)  # (b, s, g, h*d/g)
        wo_a = self.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        return self.wo_b(o.flatten(2))


# ---------- MoE ----------

class Gate(nn.Module):
    def __init__(self, layer_id: int, cfg: DSV4Config):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.hash = layer_id < cfg.n_hash_layers
        self.weight = nn.Parameter(torch.empty(cfg.n_routed_experts, cfg.dim, dtype=torch.bfloat16))
        if self.hash:
            self.register_buffer("tid2eid", torch.empty(cfg.vocab_size, cfg.n_activated_experts, dtype=torch.int32))
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(cfg.n_routed_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        scores = F.linear(x.float(), self.weight.float())  # (..., n_routed)
        if cfg.score_func == "softmax":
            scores = scores.softmax(-1)
        elif cfg.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:  # sqrtsoftplus
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        if not self.hash:
            scores = scores + self.bias
            indices = scores.topk(cfg.n_activated_experts, dim=-1)[1]
        else:
            indices = self.tid2eid[input_ids]
        weights = original_scores.gather(-1, indices.long())
        if cfg.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * cfg.route_scale
        return weights, indices


class Expert(nn.Module):
    def __init__(self, cfg: DSV4Config):
        super().__init__()
        self.cfg = cfg
        self.w1 = nn.Linear(cfg.dim, cfg.moe_inter_dim, bias=False, dtype=torch.bfloat16)
        self.w2 = nn.Linear(cfg.moe_inter_dim, cfg.dim, bias=False, dtype=torch.bfloat16)
        self.w3 = nn.Linear(cfg.dim, cfg.moe_inter_dim, bias=False, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.cfg.swiglu_limit > 0:
            up = up.clamp(min=-self.cfg.swiglu_limit, max=self.cfg.swiglu_limit)
            gate = gate.clamp(max=self.cfg.swiglu_limit)
        y = F.silu(gate) * up
        if weights is not None:
            y = weights * y
        return self.w2(y.to(dtype))


class MoE(nn.Module):
    def __init__(self, layer_id: int, cfg: DSV4Config):
        super().__init__()
        self.cfg = cfg
        self.gate = Gate(layer_id, cfg)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.n_routed_experts)])
        self.shared_experts = Expert(cfg)  # 1 shared expert

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, self.cfg.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        for i in range(self.cfg.n_routed_experts):
            mask_idx, top = torch.where(indices == i)
            if len(mask_idx) == 0:
                continue
            expert = self.experts[i]
            y[mask_idx] += expert(x[mask_idx], weights[mask_idx, top, None].to(x.dtype))
        y = y + self.shared_experts(x)
        return y.type_as(x).view(shape)


# ---------- Block with mHC ----------

class Block(nn.Module):
    def __init__(self, layer_id: int, cfg: DSV4Config):
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.attn = Attention(cfg)
        self.ffn = MoE(layer_id, cfg)
        self.attn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps, dtype=torch.bfloat16)
        self.ffn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps, dtype=torch.bfloat16)
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def _hc_pre(self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
        # x: (b, s, hc_mult, d)
        shape, dtype = x.shape, x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.cfg.norm_eps)
        mixes = F.linear(x_flat, fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, scale, base, self.cfg.hc_mult,
                                             self.cfg.hc_sinkhorn_iters, self.cfg.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def _hc_post(self, x: torch.Tensor, residual: torch.Tensor,
                 post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        # x: (b,s,d); residual: (b,s,hc,d); out: (b,s,hc,d)
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + \
            torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        residual = x
        x, post, comb = self._hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, freqs_cis)
        x = self._hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self._hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self._hc_post(x, residual, post, comb)
        return x


def load_block_from_shards(layer_id: int, cfg: DSV4Config, shard_idx) -> Block:
    """Build a Block and populate its weights from the DSV4 shard index
    (ShardIndex). Dequantizes FP4/FP8 source weights to bf16 on the fly."""
    blk = Block(layer_id, cfg)
    prefix = f"layers.{layer_id}"

    def _set(param_key: str, tensor_key: str, dtype: torch.dtype | None = None):
        w = shard_idx.read_tensor(tensor_key, out_dtype=dtype or torch.bfloat16)
        p = blk
        for seg in param_key.split("."):
            p = getattr(p, seg)
        p.data.copy_(w)

    # Attention
    _set("attn.wq_a.weight", f"{prefix}.attn.wq_a.weight")
    _set("attn.q_norm.weight", f"{prefix}.attn.q_norm.weight")
    _set("attn.wq_b.weight", f"{prefix}.attn.wq_b.weight")
    _set("attn.wkv.weight", f"{prefix}.attn.wkv.weight")
    _set("attn.kv_norm.weight", f"{prefix}.attn.kv_norm.weight")
    _set("attn.wo_a.weight", f"{prefix}.attn.wo_a.weight")
    _set("attn.wo_b.weight", f"{prefix}.attn.wo_b.weight")
    _set("attn.attn_sink", f"{prefix}.attn.attn_sink", torch.float32)

    # FFN norm
    _set("attn_norm.weight", f"{prefix}.attn_norm.weight")
    _set("ffn_norm.weight", f"{prefix}.ffn_norm.weight")

    # Gate
    _set("ffn.gate.weight", f"{prefix}.ffn.gate.weight")
    if not blk.ffn.gate.hash:
        _set("ffn.gate.bias", f"{prefix}.ffn.gate.bias", torch.float32)
    else:
        _set("ffn.gate.tid2eid", f"{prefix}.ffn.gate.tid2eid", torch.int32)

    # Routed experts
    for e in range(cfg.n_routed_experts):
        for w in ("w1", "w2", "w3"):
            _set(f"ffn.experts.{e}.{w}.weight",
                 f"{prefix}.ffn.experts.{e}.{w}.weight")

    # Shared expert
    for w in ("w1", "w2", "w3"):
        _set(f"ffn.shared_experts.{w}.weight",
             f"{prefix}.ffn.shared_experts.{w}.weight")

    # mHC
    _set("hc_attn_fn", f"{prefix}.hc_attn_fn", torch.float32)
    _set("hc_ffn_fn", f"{prefix}.hc_ffn_fn", torch.float32)
    _set("hc_attn_base", f"{prefix}.hc_attn_base", torch.float32)
    _set("hc_ffn_base", f"{prefix}.hc_ffn_base", torch.float32)
    _set("hc_attn_scale", f"{prefix}.hc_attn_scale", torch.float32)
    _set("hc_ffn_scale", f"{prefix}.hc_ffn_scale", torch.float32)

    return blk
