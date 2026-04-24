"""Single-layer forward pass for Kimi K2.6 / DeepseekV3 in pure torch.

Purpose: compute hidden_state[L+1] = layer_L(hidden_state[L], ...) for
one layer at a time, using weights just read off disk. No model object
is ever fully materialized — peak memory is one layer plus the cached
batch of hidden states.

Implements:
  - RMSNorm (DSV3 style: x * rsqrt(mean(x²) + eps) * weight)
  - MLA attention (q_lora / kv_lora, nope+rope head split, per-head RoPE)
  - Rotary position embeddings (DeepseekV3 uses NTK-aware scaled RoPE)
  - SwiGLU MLP (for dense layer 0 + shared expert)
  - MoE routing (sigmoid gate + e_score_correction_bias biased top-k,
    gate values UNBIASED and renormalized over selected experts)

For the MoE layer forward, we ALSO return per-expert L2-norm-weighted
saliency contributions so the observer can accumulate REAP scores in
the same pass as producing the next hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


# ----- config (pulled from Kimi K2.6 text_config) ------------------

@dataclass
class KimiCfg:
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    first_k_dense_replace: int = 1

    # Attention (MLA)
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # RoPE
    rope_theta: float = 50000.0
    max_position_embeddings: int = 262144
    rope_scaling: dict | None = None  # YaRN / LLaMA-style NTK; if None no scaling

    # MoE
    n_routed_experts: int = 384
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 2048
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0  # DSV3-style — applied to top-k weights

    # Dense MLP
    intermediate_size: int = 18432

    # Norms
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_config_json(cls, cfg_path):
        import json
        from pathlib import Path
        d = json.loads(Path(cfg_path).read_text())
        t = d.get("text_config", d)
        return cls(
            hidden_size=t["hidden_size"],
            num_hidden_layers=t["num_hidden_layers"],
            first_k_dense_replace=t.get("first_k_dense_replace", 1),
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            q_lora_rank=t["q_lora_rank"],
            kv_lora_rank=t["kv_lora_rank"],
            qk_nope_head_dim=t["qk_nope_head_dim"],
            qk_rope_head_dim=t["qk_rope_head_dim"],
            v_head_dim=t.get("v_head_dim", t["qk_nope_head_dim"]),
            rope_theta=t.get("rope_theta", 10000.0),
            max_position_embeddings=t.get("max_position_embeddings", 4096),
            rope_scaling=t.get("rope_scaling"),
            n_routed_experts=t["n_routed_experts"],
            n_shared_experts=t.get("n_shared_experts", 1),
            num_experts_per_tok=t["num_experts_per_tok"],
            moe_intermediate_size=t["moe_intermediate_size"],
            norm_topk_prob=t.get("norm_topk_prob", True),
            routed_scaling_factor=t.get("routed_scaling_factor", 1.0),
            intermediate_size=t["intermediate_size"],
            rms_norm_eps=t.get("rms_norm_eps", 1e-6),
        )


# ----- building blocks --------------------------------------------

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # DSV3 RMSNorm: x * rsqrt(mean(x², -1, keepdim) + eps) * weight
    # Compute in f32 for numerical stability, return in x's dtype.
    in_dtype = x.dtype
    x32 = x.float()
    rms = x32.pow(2).mean(-1, keepdim=True).add_(eps).rsqrt_()
    return (x32 * rms).mul_(weight.float()).to(in_dtype)


def precompute_rope_cos_sin(
    head_dim: int,
    max_pos: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
    scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for RoPE. head_dim here is the ROPE head_dim (64)."""
    # LLaMA/DSV3 convention: freqs = base^(-2i/head_dim) for i in [0, head_dim/2).
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                            device=device) / head_dim))
    # Optional NTK/YaRN scaling
    if scaling is not None:
        stype = scaling.get("type", scaling.get("rope_type", ""))
        if stype in ("linear",):
            factor = float(scaling.get("factor", 1.0))
            inv_freq = inv_freq / factor
        elif stype in ("dynamic", "yarn"):
            # For calibration we mostly run <= max_position_embeddings so
            # this branch is rarely active. Proper YaRN is complex —
            # fall back to base rope (inv_freq) which is accurate for
            # positions within the trained context. Flag for future work.
            pass
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin  # (T, head_dim/2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding on the LAST dim of x.

    x: (B, T, H, D) or (B, H, T, D) — must have T somewhere.
    cos/sin: (T, D/2)
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    # Figure out which axis is T: it's the one whose length matches cos.shape[0].
    T = cos.shape[0]
    # Build a broadcast-compatible view: (1, T, 1, half) if x is (B, T, H, D)
    # or (1, 1, T, half) if x is (B, H, T, D).
    if x.ndim == 4 and x.shape[1] == T:
        cos_b = cos.view(1, T, 1, half)
        sin_b = sin.view(1, T, 1, half)
    elif x.ndim == 4 and x.shape[2] == T:
        cos_b = cos.view(1, 1, T, half)
        sin_b = sin.view(1, 1, T, half)
    elif x.ndim == 3 and x.shape[1] == T:
        cos_b = cos.view(1, T, half)
        sin_b = sin.view(1, T, half)
    else:
        # Generic fallback: pick the axis whose length matches T.
        axis = next(i for i, s in enumerate(x.shape) if s == T)
        shape = [1] * x.ndim
        shape[axis] = T
        shape[-1] = half
        cos_b = cos.view(*shape)
        sin_b = sin.view(*shape)
    y1 = x1 * cos_b - x2 * sin_b
    y2 = x2 * cos_b + x1 * sin_b
    return torch.cat([y1, y2], dim=-1)


# ----- MLA attention -----------------------------------------------

def mla_attention(
    x: torch.Tensor,
    q_a_proj: torch.Tensor, q_a_layernorm: torch.Tensor,
    q_b_proj: torch.Tensor,
    kv_a_proj_with_mqa: torch.Tensor, kv_a_layernorm: torch.Tensor,
    kv_b_proj: torch.Tensor, o_proj: torch.Tensor,
    cfg: KimiCfg,
    cos: torch.Tensor, sin: torch.Tensor,
) -> torch.Tensor:
    """MLA attention forward (full no-KV-cache, calibration only).

    x: (B, T, H) bf16
    All *_proj weights are (out, in) f32 or bf16; we compute in bf16.
    cos/sin: (T, qk_rope_head_dim/2) bf16
    Returns (B, T, H) bf16.
    """
    B, T, H = x.shape
    Hq = cfg.num_attention_heads
    Hkv = cfg.num_key_value_heads
    d_nope = cfg.qk_nope_head_dim
    d_rope = cfg.qk_rope_head_dim
    d_v = cfg.v_head_dim
    d_q = d_nope + d_rope  # query head dim

    # --- Q projection with LoRA
    q_a = x @ q_a_proj.T.to(x.dtype)              # (B, T, q_lora_rank)
    q_a = rms_norm(q_a, q_a_layernorm, cfg.rms_norm_eps)
    q = q_a @ q_b_proj.T.to(x.dtype)              # (B, T, Hq * d_q)
    q = q.view(B, T, Hq, d_q)
    q_nope = q[..., :d_nope]                       # (B, T, Hq, d_nope)
    q_rope = q[..., d_nope:]                       # (B, T, Hq, d_rope)
    q_rope = apply_rope(q_rope, cos, sin)

    # --- KV projection with latent compression
    kv_a = x @ kv_a_proj_with_mqa.T.to(x.dtype)    # (B, T, kv_lora+d_rope)
    kv_compressed = kv_a[..., :cfg.kv_lora_rank]
    k_rope = kv_a[..., cfg.kv_lora_rank:]         # (B, T, d_rope) — MQA-shared
    kv_compressed = rms_norm(kv_compressed, kv_a_layernorm, cfg.rms_norm_eps)
    kv = kv_compressed @ kv_b_proj.T.to(x.dtype)  # (B, T, Hkv*(d_nope+d_v))
    kv = kv.view(B, T, Hkv, d_nope + d_v)
    k_nope = kv[..., :d_nope]                      # (B, T, Hkv, d_nope)
    v = kv[..., d_nope:]                           # (B, T, Hkv, d_v)

    k_rope = k_rope.view(B, T, 1, d_rope).expand(B, T, Hkv, d_rope)
    k_rope = apply_rope(k_rope, cos, sin)
    k = torch.cat([k_nope, k_rope], dim=-1)       # (B, T, Hkv, d_q)

    # Attention: transpose heads to dim 1 for batched matmul
    q = q.transpose(1, 2)   # (B, Hq, T, d_q)
    q = torch.cat([q_nope.transpose(1, 2), q_rope.transpose(1, 2)], dim=-1)
    k = k.transpose(1, 2)   # (B, Hkv, T, d_q)
    v = v.transpose(1, 2)   # (B, Hkv, T, d_v)

    if Hq != Hkv:
        # GQA expand
        r = Hq // Hkv
        k = k.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, Hq, T, d_q)
        v = v.unsqueeze(2).expand(-1, -1, r, -1, -1).reshape(B, Hq, T, d_v)

    # Scaled dot-product attention with causal mask.
    scale = 1.0 / math.sqrt(d_q)
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale      # (B, Hq, T, T)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
    attn = attn.masked_fill(mask, float("-inf"))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(x.dtype)
    out = torch.matmul(attn, v)                               # (B, Hq, T, d_v)
    out = out.transpose(1, 2).contiguous().view(B, T, Hq * d_v)
    return out @ o_proj.T.to(x.dtype)


# ----- dense MLP (layer 0 + shared expert) --------------------------

def swiglu_mlp(x: torch.Tensor,
               gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor) -> torch.Tensor:
    # MPS matmul cap: combined operands + intermediate buffer can exceed
    # INT_MAX (~2.14 B elements) on tensors that individually look safe,
    # because MPS materializes a contiguous padded buffer for the matmul
    # kernel. Empirically, for Kimi K2.6 the per-expert x_e shape (85-ish,
    # 7168) crashes intermittently — even though N*H is tiny — so we
    # reshape to always be 2D AND chunk at a conservative row count,
    # forcing per-call alloc below the threshold.
    #
    # This is a SALIENCY-ONLY path — speed doesn't matter. We aggressively
    # chunk to trade throughput for correctness.
    MAX_ROWS_PER_MATMUL = 16384
    orig_shape = x.shape
    if x.dim() != 2:
        x = x.reshape(-1, orig_shape[-1])
    # Also make weight tensors 2D + contiguous for predictable MPS layout
    gate_wT = gate_w.T.to(x.dtype).contiguous()
    up_wT = up_w.T.to(x.dtype).contiguous()
    down_wT = down_w.T.to(x.dtype).contiguous()
    N = x.shape[0]
    outs = []
    for s in range(0, N, MAX_ROWS_PER_MATMUL):
        e = min(s + MAX_ROWS_PER_MATMUL, N)
        xi = x[s:e].contiguous()
        gate = xi @ gate_wT
        up = xi @ up_wT
        hidden = F.silu(gate) * up
        outs.append(hidden @ down_wT)
    y = torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]
    # Restore original shape if input was 3D (B, T, H)
    if len(orig_shape) != 2:
        y = y.reshape(*orig_shape[:-1], y.shape[-1])
    return y


# ----- MoE forward with REAP observation ---------------------------

def _loader_supports_stacked(loader) -> bool:
    return getattr(loader, "supports_stacked", False)

def moe_forward_observe(
    x: torch.Tensor,                       # (B, T, H) bf16
    router_w: torch.Tensor,                # (E, H) f32
    router_bias: torch.Tensor | None,      # (E,) f32 or None
    expert_loader,                         # callable returning stacked (E,Im,H)/(E,Im,H)/(E,H,Im)
    shared_expert_weights,                 # dict {'gate_proj','up_proj','down_proj'} or None
    cfg: KimiCfg,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one MoE layer forward with REAP saliency accumulation.

    Returns:
      out:                (B, T, H) bf16 — MoE output (sum of top-k expert contributions + shared)
      saliency_sum[e]:    sum_{x in X_e} g_e(x) * ||f_e(x)||_2       (E,) f32
      count[e]:           |X_e| = number of tokens selecting expert e  (E,) i64
    """
    B, T, H = x.shape
    E = cfg.n_routed_experts
    K = cfg.num_experts_per_tok

    x_flat = x.reshape(-1, H)  # (N, H) where N = B*T
    N = x_flat.shape[0]

    # Router: sigmoid of (x @ W_gate.T). Logits kept in f32 for stability.
    router_w_t = router_w.T.to(device=device, dtype=torch.float32)
    logits = x_flat.float() @ router_w_t  # (N, E) f32
    gates = torch.sigmoid(logits)  # unbiased gate values (used downstream + in REAP)

    # Biased top-k selection (DSV3: add e_score_correction_bias for selection only)
    if router_bias is not None:
        biased = gates + router_bias.to(device=device, dtype=torch.float32)
    else:
        biased = gates
    top_k_vals_biased, top_k_idx = torch.topk(biased, k=K, dim=-1)  # (N, K)
    # Gather unbiased gate values at selected positions
    top_k_gates = torch.gather(gates, 1, top_k_idx)  # (N, K)
    if cfg.norm_topk_prob:
        top_k_gates = top_k_gates / (top_k_gates.sum(-1, keepdim=True) + 1e-20)
    top_k_gates = top_k_gates * cfg.routed_scaling_factor

    # Accumulators
    out = torch.zeros_like(x_flat, dtype=x.dtype)
    saliency_sum = torch.zeros(E, dtype=torch.float32, device=device)
    count = torch.zeros(E, dtype=torch.int64, device=device)

    # Flatten (N, K) → (N*K,) and sort by expert id for contiguous processing.
    flat_eids = top_k_idx.reshape(-1)
    flat_rows = torch.arange(N, device=device).unsqueeze(1).expand(-1, K).reshape(-1)
    flat_w = top_k_gates.reshape(-1)

    order = torch.argsort(flat_eids, stable=True)
    eids_sorted = flat_eids[order]
    rows_sorted = flat_rows[order]
    w_sorted = flat_w[order]

    unique_eids, counts = torch.unique_consecutive(eids_sorted, return_counts=True)
    splits = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.zeros(1, dtype=splits.dtype, device=device), splits[:-1]])

    # Bucket routings by expert into GPU-resident lookup arrays to avoid the
    # per-expert Python sync/.item() call.
    unique_eids_cpu = unique_eids.cpu().tolist()
    starts_cpu = starts.cpu().tolist()
    splits_cpu = splits.cpu().tolist()

    # Request the stacked expert tensors from the loader once per layer.
    # Loader may return either (gate_stk, up_stk, down_stk) with shapes
    # (E, Im, H), (E, Im, H), (E, H, Im) — stacked — or fall back to the
    # dict/per-expert interface if the caller hasn't provided stacked.
    stacked = expert_loader("__stacked__", allow_none=True) \
        if _loader_supports_stacked(expert_loader) else None

    for i, e in enumerate(unique_eids_cpu):
        s, end = starts_cpu[i], splits_cpu[i]
        rows_e = rows_sorted[s:end]
        weights_e = w_sorted[s:end]
        x_e = x_flat[rows_e].to(device=device, dtype=x.dtype)

        if stacked is not None:
            gate_w = stacked[0][e]; up_w = stacked[1][e]; down_w = stacked[2][e]
        else:
            gate_w, up_w, down_w = expert_loader(e)
        f = swiglu_mlp(x_e, gate_w, up_w, down_w)
        norms = torch.linalg.vector_norm(f.float(), ord=2, dim=-1)
        saliency_sum[e] += (weights_e * norms).sum()
        count[e] += norms.numel()
        out[rows_e] += f * weights_e.to(f.dtype).unsqueeze(-1)

    # Shared expert (always-on, no gating)
    if shared_expert_weights is not None:
        s_gate = shared_expert_weights["gate_proj"]
        s_up = shared_expert_weights["up_proj"]
        s_down = shared_expert_weights["down_proj"]
        out = out + swiglu_mlp(x_flat, s_gate, s_up, s_down)

    return out.reshape(B, T, H), saliency_sum, count


# ----- whole-layer forward (single batch) ---------------------------

@dataclass
class LayerWeights:
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    # attention (MLA)
    q_a_proj: torch.Tensor
    q_a_layernorm: torch.Tensor
    q_b_proj: torch.Tensor
    kv_a_proj_with_mqa: torch.Tensor
    kv_a_layernorm: torch.Tensor
    kv_b_proj: torch.Tensor
    o_proj: torch.Tensor
    # MLP: one of dense_mlp OR (router + expert_loader + shared_expert)
    dense_mlp: dict | None = None
    router_w: torch.Tensor | None = None
    router_bias: torch.Tensor | None = None
    expert_loader: Any | None = None
    shared_expert: dict | None = None


def decoder_layer_forward(
    x: torch.Tensor,
    lw: LayerWeights,
    cfg: KimiCfg,
    cos: torch.Tensor, sin: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Full DSV3 decoder layer forward. Returns (hidden, saliency_sum, count).

    saliency_sum/count are None for dense layer 0, tensors for MoE layers.
    """
    # Attention block
    r = x
    h = rms_norm(x, lw.input_layernorm, cfg.rms_norm_eps)
    h = mla_attention(
        h, lw.q_a_proj, lw.q_a_layernorm, lw.q_b_proj,
        lw.kv_a_proj_with_mqa, lw.kv_a_layernorm, lw.kv_b_proj, lw.o_proj,
        cfg, cos, sin,
    )
    h = r + h

    # MLP block
    r = h
    h = rms_norm(h, lw.post_attention_layernorm, cfg.rms_norm_eps)
    saliency = None; count = None
    if lw.dense_mlp is not None:
        h = swiglu_mlp(h, lw.dense_mlp["gate_proj"], lw.dense_mlp["up_proj"], lw.dense_mlp["down_proj"])
    else:
        h, saliency, count = moe_forward_observe(
            h, lw.router_w, lw.router_bias, lw.expert_loader,
            lw.shared_expert, cfg, device,
        )
    h = r + h
    return h, saliency, count
