"""Load actual DSV4-Flash layer 0 weights (dequanted to bf16) into
BOTH torch Block and MLX DeepseekV4DecoderLayer. Run same input,
compare. Diagnoses real-weight divergence that synthetic random
weights might miss.
"""

from __future__ import annotations

import argparse
import numpy as np
import torch
import mlx.core as mx

from jang_tools.dsv4.layer_forward import DSV4Config, Block as TorchBlock
from jang_tools.dsv4.mlx_model import ModelArgs, DeepseekV4DecoderLayer
from jang_tools.dsv4.weight_loader import ShardIndex
from jang_tools.dsv4.ops import precompute_freqs_cis
import json


def t2m(t):
    return mx.array(t.detach().cpu().float().numpy())


def _set(obj, path, v):
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p) if not p.isdigit() else obj[int(p)]
    setattr(obj, parts[-1], v)


def load_layer(idx: ShardIndex, cfg_t: DSV4Config, cfg_m: ModelArgs, layer_id: int):
    tb = TorchBlock(layer_id=layer_id, cfg=cfg_t)
    tb.train(False)
    ml = DeepseekV4DecoderLayer(cfg_m, layer_id=layer_id)

    pfx = f"layers.{layer_id}"

    # Attention
    wq_a = idx.read_tensor(f"{pfx}.attn.wq_a.weight", out_dtype=torch.bfloat16)
    wq_b = idx.read_tensor(f"{pfx}.attn.wq_b.weight", out_dtype=torch.bfloat16)
    wkv = idx.read_tensor(f"{pfx}.attn.wkv.weight", out_dtype=torch.bfloat16)
    wo_a = idx.read_tensor(f"{pfx}.attn.wo_a.weight", out_dtype=torch.bfloat16)
    wo_b = idx.read_tensor(f"{pfx}.attn.wo_b.weight", out_dtype=torch.bfloat16)
    q_norm = idx.read_tensor(f"{pfx}.attn.q_norm.weight", out_dtype=torch.bfloat16)
    kv_norm = idx.read_tensor(f"{pfx}.attn.kv_norm.weight", out_dtype=torch.bfloat16)
    attn_sink = idx.read_tensor(f"{pfx}.attn.attn_sink", out_dtype=torch.float32)
    attn_norm = idx.read_tensor(f"{pfx}.attn_norm.weight", out_dtype=torch.bfloat16)
    ffn_norm = idx.read_tensor(f"{pfx}.ffn_norm.weight", out_dtype=torch.bfloat16)

    tb.attn.wq_a.weight.data = wq_a
    tb.attn.wq_b.weight.data = wq_b
    tb.attn.wkv.weight.data = wkv
    tb.attn.wo_a.weight.data = wo_a
    tb.attn.wo_b.weight.data = wo_b
    tb.attn.q_norm.weight.data = q_norm
    tb.attn.kv_norm.weight.data = kv_norm
    tb.attn.attn_sink.data = attn_sink
    tb.attn_norm.weight.data = attn_norm
    tb.ffn_norm.weight.data = ffn_norm

    ml.self_attn.wq_a.weight = t2m(wq_a)
    ml.self_attn.wq_b.weight = t2m(wq_b)
    ml.self_attn.wkv.weight = t2m(wkv)
    ml.self_attn.wo_a.weight = t2m(wo_a)
    ml.self_attn.wo_b.weight = t2m(wo_b)
    ml.self_attn.q_norm.weight = t2m(q_norm)
    ml.self_attn.kv_norm.weight = t2m(kv_norm)
    ml.self_attn.attn_sink = t2m(attn_sink)
    ml.input_layernorm.weight = t2m(attn_norm)
    ml.post_attention_layernorm.weight = t2m(ffn_norm)

    # Gate
    gate_w = idx.read_tensor(f"{pfx}.ffn.gate.weight", out_dtype=torch.bfloat16)
    tb.ffn.gate.weight.data = gate_w
    ml.mlp.gate.weight = t2m(gate_w)

    # Gate bias or tid2eid
    is_hash = layer_id < cfg_t.n_hash_layers
    if is_hash:
        tid2eid_key = f"{pfx}.ffn.gate.tid2eid"
        t = idx.read_tensor(tid2eid_key, out_dtype=torch.int64)
        tb.ffn.gate.tid2eid.data = t.int()
        ml.mlp.gate.tid2eid = mx.array(t.numpy().astype(np.int32))
    else:
        gate_b = idx.read_tensor(f"{pfx}.ffn.gate.bias", out_dtype=torch.float32)
        tb.ffn.gate.bias.data = gate_b
        ml.mlp.gate.bias = t2m(gate_b)

    # Routed experts — stack into switch_mlp
    w1s, w2s, w3s = [], [], []
    for e in range(cfg_t.n_routed_experts):
        w1 = idx.read_tensor(f"{pfx}.ffn.experts.{e}.w1.weight", out_dtype=torch.bfloat16)
        w2 = idx.read_tensor(f"{pfx}.ffn.experts.{e}.w2.weight", out_dtype=torch.bfloat16)
        w3 = idx.read_tensor(f"{pfx}.ffn.experts.{e}.w3.weight", out_dtype=torch.bfloat16)
        tb.ffn.experts[e].w1.weight.data = w1
        tb.ffn.experts[e].w2.weight.data = w2
        tb.ffn.experts[e].w3.weight.data = w3
        w1s.append(t2m(w1))
        w2s.append(t2m(w2))
        w3s.append(t2m(w3))
    ml.mlp.switch_mlp.gate_proj.weight = mx.stack(w1s)
    ml.mlp.switch_mlp.down_proj.weight = mx.stack(w2s)
    ml.mlp.switch_mlp.up_proj.weight = mx.stack(w3s)

    # Shared expert
    sw1 = idx.read_tensor(f"{pfx}.ffn.shared_experts.w1.weight", out_dtype=torch.bfloat16)
    sw2 = idx.read_tensor(f"{pfx}.ffn.shared_experts.w2.weight", out_dtype=torch.bfloat16)
    sw3 = idx.read_tensor(f"{pfx}.ffn.shared_experts.w3.weight", out_dtype=torch.bfloat16)
    tb.ffn.shared_experts.w1.weight.data = sw1
    tb.ffn.shared_experts.w2.weight.data = sw2
    tb.ffn.shared_experts.w3.weight.data = sw3
    ml.mlp.shared_experts.gate_proj.weight = t2m(sw1)
    ml.mlp.shared_experts.down_proj.weight = t2m(sw2)
    ml.mlp.shared_experts.up_proj.weight = t2m(sw3)

    # mHC
    for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
                 "hc_attn_scale", "hc_ffn_scale"):
        t = idx.read_tensor(f"{pfx}.{name}", out_dtype=torch.float32)
        getattr(tb, name).data = t
        setattr(ml, name, t2m(t))

    return tb, ml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Path to DeepSeek-V4-Flash source")
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()

    src_cfg = json.load(open(f"{args.src}/config.json"))
    cfg_t = DSV4Config.from_config_json(src_cfg)
    cfg_m = ModelArgs(
        vocab_size=src_cfg["vocab_size"], hidden_size=src_cfg["hidden_size"],
        num_hidden_layers=src_cfg["num_hidden_layers"],
        num_attention_heads=src_cfg["num_attention_heads"],
        num_key_value_heads=src_cfg["num_key_value_heads"],
        head_dim=src_cfg["head_dim"], qk_rope_head_dim=src_cfg["qk_rope_head_dim"],
        q_lora_rank=src_cfg["q_lora_rank"], o_lora_rank=src_cfg["o_lora_rank"],
        o_groups=src_cfg["o_groups"],
        n_routed_experts=src_cfg["n_routed_experts"],
        n_shared_experts=src_cfg["n_shared_experts"],
        num_experts_per_tok=src_cfg["num_experts_per_tok"],
        moe_intermediate_size=src_cfg["moe_intermediate_size"],
        num_hash_layers=src_cfg.get("num_hash_layers", 3),
        num_nextn_predict_layers=src_cfg.get("num_nextn_predict_layers", 1),
        scoring_func=src_cfg.get("scoring_func", "sqrtsoftplus"),
        topk_method="noaux_tc", norm_topk_prob=src_cfg["norm_topk_prob"],
        routed_scaling_factor=src_cfg.get("routed_scaling_factor", 1.5),
        swiglu_limit=src_cfg.get("swiglu_limit", 10.0),
        hc_mult=src_cfg["hc_mult"], hc_sinkhorn_iters=src_cfg["hc_sinkhorn_iters"],
        hc_eps=src_cfg["hc_eps"],
        rope_theta=src_cfg["rope_theta"], rope_scaling=src_cfg["rope_scaling"],
        max_position_embeddings=src_cfg["max_position_embeddings"],
        sliding_window=src_cfg.get("sliding_window", 128),
        rms_norm_eps=src_cfg["rms_norm_eps"],
    )
    print(f"loading layer {args.layer} real weights...", flush=True)
    idx = ShardIndex(args.src)
    tb, ml = load_layer(idx, cfg_t, cfg_m, args.layer)

    B, L = 1, 6
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((B, L, cfg_m.hc_mult, cfg_m.hidden_size)).astype(np.float32) * 0.05
    x_t = torch.from_numpy(x_np).to(torch.bfloat16)
    x_m = mx.array(x_np)
    freqs_cis = precompute_freqs_cis(
        cfg_t.rope_head_dim, L, cfg_t.original_seq_len,
        cfg_t.rope_theta, cfg_t.rope_factor,
        cfg_t.beta_fast, cfg_t.beta_slow,
    )
    iids_t = torch.zeros(B, L, dtype=torch.long)
    iids_m = mx.zeros((B, L), dtype=mx.int32)

    with torch.no_grad():
        y_t = tb(x_t, freqs_cis, iids_t).float().numpy()
    y_m = np.asarray(ml(x_m, input_ids=iids_m))

    print(f"torch mean/std: {y_t.mean():.5f} / {y_t.std():.5f}")
    print(f"mlx   mean/std: {y_m.mean():.5f} / {y_m.std():.5f}")
    print(f"max abs diff: {np.abs(y_t - y_m).max():.5f}")
    print(f"mean abs diff: {np.abs(y_t - y_m).mean():.5f}")


if __name__ == "__main__":
    main()
