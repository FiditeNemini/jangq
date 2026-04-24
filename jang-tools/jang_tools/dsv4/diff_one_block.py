"""Synthetic layer-0 diff: torch Block (reference) vs MLX DeepseekV4DecoderLayer.

Uses a mini-sized config with random weights shared between the two
implementations. Runs identical input through both and prints max abs
diff. If diff > 1e-2 anywhere, the MLX layer has an architecture bug.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import mlx.core as mx

from jang_tools.dsv4.layer_forward import DSV4Config, Block as TorchBlock
from jang_tools.dsv4.mlx_model import (
    ModelArgs, DeepseekV4DecoderLayer,
)
from jang_tools.dsv4.ops import precompute_freqs_cis


import os
_REAL = os.environ.get("DIFF_REAL", "0") == "1"
if _REAL:
    DIM = 4096
    HEADS = 64
    HEAD_DIM = 512
    ROPE_DIM = 64
    QR = 1024
    OR = 1024
    OG = 8
    N_ROUTED = 8       # keep small for speed
    N_SHARED = 1
    N_ACTIVE = 2
    MOE_INTER = 2048
    HCM = 4
else:
    DIM = 128
    HEADS = 4
    HEAD_DIM = 64
    ROPE_DIM = 16
    QR = 32
    OR = 32
    OG = 2
    N_ROUTED = 4
    N_SHARED = 1
    N_ACTIVE = 2
    MOE_INTER = 64
    HCM = 2


def build_configs():
    tcfg = DSV4Config(
        vocab_size=100, dim=DIM, n_layers=1, n_hash_layers=0, n_mtp_layers=1,
        n_heads=HEADS, n_kv_heads=1, head_dim=HEAD_DIM, rope_head_dim=ROPE_DIM,
        q_lora_rank=QR, o_lora_rank=OR, o_groups=OG,
        n_routed_experts=N_ROUTED, n_shared_experts=N_SHARED,
        n_activated_experts=N_ACTIVE, moe_inter_dim=MOE_INTER,
        score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=10.0,
        norm_topk_prob=True, rope_theta=10000.0, rope_factor=1.0,
        original_seq_len=0, beta_fast=32, beta_slow=1,
        hc_mult=HCM, hc_sinkhorn_iters=20, hc_eps=1e-6,
        norm_eps=1e-6, window_size=128,
    )
    mcfg = ModelArgs(
        vocab_size=100, hidden_size=DIM, num_hidden_layers=1,
        num_attention_heads=HEADS, num_key_value_heads=1, head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM, q_lora_rank=QR, o_lora_rank=OR, o_groups=OG,
        n_routed_experts=N_ROUTED, n_shared_experts=N_SHARED,
        num_experts_per_tok=N_ACTIVE, moe_intermediate_size=MOE_INTER,
        num_hash_layers=0, num_nextn_predict_layers=1,
        scoring_func="sqrtsoftplus", topk_method="noaux_tc", norm_topk_prob=True,
        routed_scaling_factor=1.5, swiglu_limit=10.0,
        hc_mult=HCM, hc_sinkhorn_iters=20, hc_eps=1e-6,
        rope_theta=10000.0, rope_scaling=None,
        max_position_embeddings=128, sliding_window=128, rms_norm_eps=1e-6,
    )
    return tcfg, mcfg


def t2m(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def rand_shape(rng, shape, scale=0.02):
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32) * scale)


def init_shared(tb: TorchBlock, ml: DeepseekV4DecoderLayer):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Attention
    tb.attn.wq_a.weight.data = rand_shape(rng, (QR, DIM)).to(torch.bfloat16)
    ml.self_attn.wq_a.weight = t2m(tb.attn.wq_a.weight)
    tb.attn.q_norm.weight.data = torch.ones(QR, dtype=torch.bfloat16)
    ml.self_attn.q_norm.weight = t2m(tb.attn.q_norm.weight)
    tb.attn.wq_b.weight.data = rand_shape(rng, (HEADS * HEAD_DIM, QR)).to(torch.bfloat16)
    ml.self_attn.wq_b.weight = t2m(tb.attn.wq_b.weight)
    tb.attn.wkv.weight.data = rand_shape(rng, (HEAD_DIM, DIM)).to(torch.bfloat16)
    ml.self_attn.wkv.weight = t2m(tb.attn.wkv.weight)
    tb.attn.kv_norm.weight.data = torch.ones(HEAD_DIM, dtype=torch.bfloat16)
    ml.self_attn.kv_norm.weight = t2m(tb.attn.kv_norm.weight)
    tb.attn.wo_a.weight.data = rand_shape(rng, (OG * OR, HEADS * HEAD_DIM // OG)).to(torch.bfloat16)
    ml.self_attn.wo_a.weight = t2m(tb.attn.wo_a.weight)
    tb.attn.wo_b.weight.data = rand_shape(rng, (DIM, OG * OR)).to(torch.bfloat16)
    ml.self_attn.wo_b.weight = t2m(tb.attn.wo_b.weight)
    tb.attn.attn_sink.data = torch.zeros(HEADS, dtype=torch.float32)
    ml.self_attn.attn_sink = t2m(tb.attn.attn_sink)

    # Norms
    tb.attn_norm.weight.data = torch.ones(DIM, dtype=torch.bfloat16)
    ml.input_layernorm.weight = t2m(tb.attn_norm.weight)
    tb.ffn_norm.weight.data = torch.ones(DIM, dtype=torch.bfloat16)
    ml.post_attention_layernorm.weight = t2m(tb.ffn_norm.weight)

    # Gate
    tb.ffn.gate.weight.data = rand_shape(rng, (N_ROUTED, DIM)).to(torch.bfloat16)
    ml.mlp.gate.weight = t2m(tb.ffn.gate.weight)
    tb.ffn.gate.bias.data = rand_shape(rng, (N_ROUTED,)).to(torch.float32)
    ml.mlp.gate.bias = t2m(tb.ffn.gate.bias)

    # Routed experts
    w1s, w2s, w3s = [], [], []
    for e in range(N_ROUTED):
        w1 = rand_shape(rng, (MOE_INTER, DIM)).to(torch.bfloat16)
        w2 = rand_shape(rng, (DIM, MOE_INTER)).to(torch.bfloat16)
        w3 = rand_shape(rng, (MOE_INTER, DIM)).to(torch.bfloat16)
        tb.ffn.experts[e].w1.weight.data = w1
        tb.ffn.experts[e].w2.weight.data = w2
        tb.ffn.experts[e].w3.weight.data = w3
        w1s.append(t2m(w1)); w2s.append(t2m(w2)); w3s.append(t2m(w3))
    ml.mlp.switch_mlp.gate_proj.weight = mx.stack(w1s)
    ml.mlp.switch_mlp.down_proj.weight = mx.stack(w2s)
    ml.mlp.switch_mlp.up_proj.weight = mx.stack(w3s)

    # Shared expert
    sw1 = rand_shape(rng, (MOE_INTER, DIM)).to(torch.bfloat16)
    sw2 = rand_shape(rng, (DIM, MOE_INTER)).to(torch.bfloat16)
    sw3 = rand_shape(rng, (MOE_INTER, DIM)).to(torch.bfloat16)
    tb.ffn.shared_experts.w1.weight.data = sw1
    tb.ffn.shared_experts.w2.weight.data = sw2
    tb.ffn.shared_experts.w3.weight.data = sw3
    ml.mlp.shared_experts.gate_proj.weight = t2m(sw1)
    ml.mlp.shared_experts.down_proj.weight = t2m(sw2)
    ml.mlp.shared_experts.up_proj.weight = t2m(sw3)

    # mHC
    mix_hc = (2 + HCM) * HCM
    hc_dim = HCM * DIM
    for name, shape in (
        ("hc_attn_fn", (mix_hc, hc_dim)),
        ("hc_ffn_fn", (mix_hc, hc_dim)),
        ("hc_attn_base", (mix_hc,)),
        ("hc_ffn_base", (mix_hc,)),
        ("hc_attn_scale", (3,)),
        ("hc_ffn_scale", (3,)),
    ):
        t = rand_shape(rng, shape).to(torch.float32)
        getattr(tb, name).data = t
        setattr(ml, name, t2m(t))


def main():
    tcfg, mcfg = build_configs()
    tb = TorchBlock(layer_id=0, cfg=tcfg)
    tb.train(False)
    ml = DeepseekV4DecoderLayer(mcfg, layer_id=0)
    init_shared(tb, ml)

    B, L = 1, 8
    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((B, L, HCM, DIM)).astype(np.float32) * 0.1
    x_t = torch.from_numpy(x_np).to(torch.bfloat16)
    x_m = mx.array(x_np)

    freqs_cis = precompute_freqs_cis(ROPE_DIM, L, 0, 10000.0, 1.0, 32, 1)
    iids_t = torch.zeros(B, L, dtype=torch.long)
    iids_m = mx.zeros((B, L), dtype=mx.int32)

    with torch.no_grad():
        y_t = tb(x_t, freqs_cis, iids_t).float().numpy()
    y_m = np.asarray(ml(x_m, input_ids=iids_m))

    print("torch out mean/std:", float(y_t.mean()), float(y_t.std()))
    print("mlx   out mean/std:", float(y_m.mean()), float(y_m.std()))
    print("max abs diff:", float(np.max(np.abs(y_t - y_m))))
    print("mean abs diff:", float(np.mean(np.abs(y_t - y_m))))
    print("shapes:", y_t.shape, y_m.shape)


if __name__ == "__main__":
    main()
