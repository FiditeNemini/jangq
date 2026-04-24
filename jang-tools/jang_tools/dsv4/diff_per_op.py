"""Per-op numerical diff: torch reference vs MLX DeepseekV4DecoderLayer.

Monkey-patches both implementations to record intermediate tensors
at matching checkpoints, then prints max/mean abs diff per op.

Run at DSV4 real dimensions by default so scale-dependent bugs surface.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx.nn as mlnn

from jang_tools.dsv4.layer_forward import DSV4Config, Block as TorchBlock
from jang_tools.dsv4.mlx_model import (
    ModelArgs, DeepseekV4DecoderLayer, _mlx_apply_rotary_cis,
    hc_split_sinkhorn as mlx_hc_split_sinkhorn,
)
from jang_tools.dsv4.ops import precompute_freqs_cis, apply_rotary_emb, hc_split_sinkhorn as torch_hc_split_sinkhorn
from jang_tools.dsv4.diff_one_block import init_shared, build_configs, t2m


_t_trace: dict[str, np.ndarray] = {}
_m_trace: dict[str, np.ndarray] = {}


def _record(store, name, x):
    if isinstance(x, torch.Tensor):
        store[name] = x.detach().float().cpu().numpy()
    else:
        store[name] = np.asarray(x)


def torch_forward_traced(tb: TorchBlock, x, freqs_cis, input_ids):
    """Verbose torch forward with _t_trace dict populated at each major op."""
    cfg = tb.cfg
    _record(_t_trace, "input", x)

    # ---- _hc_pre (attention) ----
    residual = x
    shape, dtype = x.shape, x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + cfg.norm_eps)
    mixes = F.linear(x_flat, tb.hc_attn_fn) * rsqrt
    _record(_t_trace, "hc_pre_attn.mixes", mixes)
    pre, post, comb = torch_hc_split_sinkhorn(
        mixes, tb.hc_attn_scale, tb.hc_attn_base, cfg.hc_mult,
        cfg.hc_sinkhorn_iters, cfg.hc_eps,
    )
    _record(_t_trace, "hc_pre_attn.pre", pre)
    _record(_t_trace, "hc_pre_attn.post", post)
    _record(_t_trace, "hc_pre_attn.comb", comb)
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2).to(dtype)
    x = y
    _record(_t_trace, "hc_pre_attn.out", x)

    # ---- attn_norm ----
    x = tb.attn_norm(x)
    _record(_t_trace, "attn_norm", x)

    # ---- Attention (inlined) ----
    attn = tb.attn
    bsz, seqlen, _ = x.shape
    qr_in = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr_in).unflatten(-1, (cfg.n_heads, cfg.head_dim)).transpose(1, 2)
    q_f = q.float()
    q = (q_f * torch.rsqrt(q_f.square().mean(-1, keepdim=True) + cfg.norm_eps)).to(q.dtype)
    _record(_t_trace, "attn.q_pre_rope", q)
    apply_rotary_emb(q[..., -cfg.rope_head_dim:], freqs_cis)
    _record(_t_trace, "attn.q_post_rope", q)
    kv = attn.kv_norm(attn.wkv(x)).unsqueeze(1)
    apply_rotary_emb(kv[..., -cfg.rope_head_dim:], freqs_cis)
    _record(_t_trace, "attn.kv_post_rope", kv)
    k = kv.expand(-1, cfg.n_heads, -1, -1)
    scores = (q @ k.transpose(-1, -2)) * attn.softmax_scale
    mask = torch.triu(torch.full((seqlen, seqlen), float("-inf")), diagonal=1)
    scores = scores + mask
    sink = attn.attn_sink.view(1, cfg.n_heads, 1, 1).expand(bsz, -1, seqlen, 1).to(scores.dtype)
    scores = torch.cat([sink, scores], dim=-1)
    attn_w = scores.softmax(dim=-1)
    attn_w = attn_w[..., 1:].to(kv.dtype)
    v = k
    o = attn_w @ v
    _record(_t_trace, "attn.o_pre_invrope", o)
    apply_rotary_emb(o[..., -cfg.rope_head_dim:], freqs_cis, inverse=True)
    _record(_t_trace, "attn.o_post_invrope", o)
    o = o.transpose(1, 2).contiguous().view(bsz, seqlen, cfg.o_groups, -1)
    wo_a = attn.wo_a.weight.view(cfg.o_groups, cfg.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    attn_out = attn.wo_b(o.flatten(2))
    _record(_t_trace, "attn.out", attn_out)

    # ---- _hc_post (attention) ----
    y = post.unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    x = y.type_as(attn_out)
    _record(_t_trace, "hc_post_attn", x)

    # ---- _hc_pre (ffn) ----
    residual = x
    shape, dtype = x.shape, x.dtype
    x_flat = x.flatten(2).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + cfg.norm_eps)
    mixes = F.linear(x_flat, tb.hc_ffn_fn) * rsqrt
    pre, post, comb = torch_hc_split_sinkhorn(
        mixes, tb.hc_ffn_scale, tb.hc_ffn_base, cfg.hc_mult,
        cfg.hc_sinkhorn_iters, cfg.hc_eps,
    )
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2).to(dtype)
    x = y
    _record(_t_trace, "hc_pre_ffn.out", x)

    # ---- ffn_norm ----
    x = tb.ffn_norm(x)
    _record(_t_trace, "ffn_norm", x)

    # ---- MoE ----
    x = tb.ffn(x, input_ids)
    _record(_t_trace, "moe.out", x)

    # ---- _hc_post (ffn) ----
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2
    )
    x = y.type_as(x)
    _record(_t_trace, "final", x)
    return x


def mlx_forward_traced(ml: DeepseekV4DecoderLayer, x, input_ids):
    args = ml.args
    _record(_m_trace, "input", x)

    # ---- _hc_pre (attention) ----
    residual = x
    shape = x.shape
    x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(x_flat.square(), axis=-1, keepdims=True) + args.rms_norm_eps)
    mixes = (x_flat @ ml.hc_attn_fn.T) * rsqrt
    _record(_m_trace, "hc_pre_attn.mixes", mixes)
    pre, post, comb = mlx_hc_split_sinkhorn(
        mixes, ml.hc_attn_scale, ml.hc_attn_base, args.hc_mult,
        args.hc_sinkhorn_iters, args.hc_eps,
    )
    _record(_m_trace, "hc_pre_attn.pre", pre)
    _record(_m_trace, "hc_pre_attn.post", post)
    _record(_m_trace, "hc_pre_attn.comb", comb)
    y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
    x = y.astype(x.dtype)
    _record(_m_trace, "hc_pre_attn.out", x)

    # ---- attn_norm ----
    x = ml.input_layernorm(x)
    _record(_m_trace, "attn_norm", x)

    # ---- Attention (inlined, matches MLX impl but traced) ----
    attn = ml.self_attn
    B, L, D = x.shape
    qr_in = attn.q_norm(attn.wq_a(x))
    q = attn.wq_b(qr_in).reshape(B, L, attn.n_heads, attn.head_dim).transpose(0, 2, 1, 3)
    q = q * mx.rsqrt(mx.mean(q.astype(mx.float32).square(), axis=-1, keepdims=True) + args.rms_norm_eps).astype(q.dtype)
    _record(_m_trace, "attn.q_pre_rope", q)
    q_nope, q_rope = mx.split(q, [attn.nope_head_dim], axis=-1)
    fc_real = attn._freqs_cis_real[:L]
    q_rope = _mlx_apply_rotary_cis(q_rope, fc_real)
    q = mx.concatenate([q_nope, q_rope], axis=-1)
    _record(_m_trace, "attn.q_post_rope", q)
    kv = attn.kv_norm(attn.wkv(x)).reshape(B, L, 1, attn.head_dim).transpose(0, 2, 1, 3)
    kv_nope, kv_rope = mx.split(kv, [attn.nope_head_dim], axis=-1)
    kv_rope = _mlx_apply_rotary_cis(kv_rope, fc_real)
    kv = mx.concatenate([kv_nope, kv_rope], axis=-1)
    _record(_m_trace, "attn.kv_post_rope", kv)
    k = mx.broadcast_to(kv, (B, attn.n_heads, kv.shape[2], attn.head_dim))
    v = k
    q_f = q.astype(mx.float32); k_f = k.astype(mx.float32); v_f = v.astype(mx.float32)
    scores = (q_f * attn.softmax_scale) @ k_f.swapaxes(-1, -2)
    m = mx.triu(mx.full((L, L), -mx.inf, dtype=mx.float32), k=1)
    scores = scores + m
    sink = attn.attn_sink.astype(mx.float32).reshape(1, attn.n_heads, 1, 1)
    sink = mx.broadcast_to(sink, scores.shape[:-1] + (1,))
    scores = mx.concatenate([sink, scores], axis=-1)
    attn_w = mx.softmax(scores, axis=-1, precise=True)
    attn_w = attn_w[..., 1:]
    o = attn_w @ v_f
    o = o.astype(q.dtype)
    _record(_m_trace, "attn.o_pre_invrope", o)
    o_nope, o_rope = mx.split(o, [attn.nope_head_dim], axis=-1)
    cos_ir = fc_real[..., 0]; sin_ir = -fc_real[..., 1]
    fc_inv = mx.stack([cos_ir, sin_ir], axis=-1)
    o_rope = _mlx_apply_rotary_cis(o_rope, fc_inv)
    o = mx.concatenate([o_nope, o_rope], axis=-1)
    _record(_m_trace, "attn.o_post_invrope", o)
    o = o.transpose(0, 2, 1, 3).reshape(B, L, attn.o_groups, -1)
    wa = attn.wo_a.weight
    wo_a_weight = wa.reshape(attn.o_groups, attn.o_lora_rank, -1)
    o = mx.einsum("bsgd,grd->bsgr", o, wo_a_weight)
    attn_out = attn.wo_b(mx.flatten(o, start_axis=2))
    _record(_m_trace, "attn.out", attn_out)

    # ---- _hc_post (attention) using FIXED axis ----
    y = post[..., None] * attn_out[..., None, :] + mx.sum(
        comb[..., None] * residual[..., None, :], axis=2
    )
    x = y.astype(attn_out.dtype)
    _record(_m_trace, "hc_post_attn", x)

    # ---- _hc_pre (ffn) ----
    residual = x
    shape = x.shape
    x_flat = mx.flatten(x, start_axis=2).astype(mx.float32)
    rsqrt = mx.rsqrt(mx.mean(x_flat.square(), axis=-1, keepdims=True) + args.rms_norm_eps)
    mixes = (x_flat @ ml.hc_ffn_fn.T) * rsqrt
    pre, post, comb = mlx_hc_split_sinkhorn(
        mixes, ml.hc_ffn_scale, ml.hc_ffn_base, args.hc_mult,
        args.hc_sinkhorn_iters, args.hc_eps,
    )
    y = mx.sum(pre[..., None] * mx.reshape(x_flat, shape), axis=2)
    x = y.astype(x.dtype)
    _record(_m_trace, "hc_pre_ffn.out", x)

    # ---- ffn_norm ----
    x = ml.post_attention_layernorm(x)
    _record(_m_trace, "ffn_norm", x)

    # ---- MoE ----
    x = ml.mlp(x, input_ids=input_ids)
    _record(_m_trace, "moe.out", x)

    # ---- _hc_post (ffn) ----
    y = post[..., None] * x[..., None, :] + mx.sum(
        comb[..., None] * residual[..., None, :], axis=2
    )
    x = y.astype(x.dtype)
    _record(_m_trace, "final", x)
    return x


def main():
    tcfg, mcfg = build_configs()
    tb = TorchBlock(layer_id=0, cfg=tcfg); tb.train(False)
    ml = DeepseekV4DecoderLayer(mcfg, layer_id=0)
    init_shared(tb, ml)

    from jang_tools.dsv4.diff_one_block import DIM, HCM, ROPE_DIM
    B, L = 1, 8
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((B, L, HCM, DIM)).astype(np.float32) * 0.1
    x_t = torch.from_numpy(x_np).to(torch.bfloat16)
    x_m = mx.array(x_np)
    freqs_cis = precompute_freqs_cis(ROPE_DIM, L, 0, 10000.0, 1.0, 32, 1)
    iids_t = torch.zeros(B, L, dtype=torch.long)
    iids_m = mx.zeros((B, L), dtype=mx.int32)

    with torch.no_grad():
        torch_forward_traced(tb, x_t, freqs_cis, iids_t)
    mlx_forward_traced(ml, x_m, iids_m)

    keys = [
        "input", "hc_pre_attn.mixes", "hc_pre_attn.pre", "hc_pre_attn.post",
        "hc_pre_attn.comb", "hc_pre_attn.out", "attn_norm",
        "attn.q_pre_rope", "attn.q_post_rope", "attn.kv_post_rope",
        "attn.o_pre_invrope", "attn.o_post_invrope", "attn.out",
        "hc_post_attn", "hc_pre_ffn.out", "ffn_norm", "moe.out", "final",
    ]
    print(f"{'op':<28} {'max':>12} {'mean':>12} {'t_std':>10} {'m_std':>10}")
    for k in keys:
        if k not in _t_trace or k not in _m_trace:
            print(f"{k:<28} MISSING ({k in _t_trace}, {k in _m_trace})")
            continue
        t = _t_trace[k]; m = _m_trace[k]
        if t.shape != m.shape:
            print(f"{k:<28} SHAPE MISMATCH: t={t.shape} m={m.shape}")
            continue
        diff = np.abs(t - m)
        print(f"{k:<28} {diff.max():>12.5f} {diff.mean():>12.6f} "
              f"{t.std():>10.5f} {m.std():>10.5f}")


if __name__ == "__main__":
    main()
