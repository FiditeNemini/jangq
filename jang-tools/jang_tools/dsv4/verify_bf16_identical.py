"""Load IDENTICAL bf16 weights into Torch ref + MLX runtime; diff outputs.

This removes quantization as a variable. If the diff is still large,
the bug is in the MLX architecture (attention, mHC, gate, etc.),
not in quantization.

Run:
  python -m jang_tools.dsv4.verify_bf16_identical --source <path/to/DeepSeek-V4-Flash> --layer 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import mlx.core as mx

from jang_tools.dsv4 import mlx_register
from jang_tools.dsv4.mlx_model import ModelArgs as MLXArgs, DeepseekV4DecoderLayer
from jang_tools.dsv4.layer_forward import DSV4Config, Block, load_block_from_shards
from jang_tools.dsv4.weight_loader import ShardIndex
from jang_tools.dsv4.ops import precompute_freqs_cis


def _to_mx(t: torch.Tensor) -> mx.array:
    """Torch bf16 → mx fp16 (safetensors-compatible, close to bf16)."""
    return mx.array(t.detach().float().numpy().astype(np.float16))


def diff(name: str, a: mx.array, b: torch.Tensor, report_sample: bool = False):
    an = np.array(a, copy=False).astype(np.float32)
    bn = b.detach().float().numpy()
    if an.shape != bn.shape:
        print(f"  [{name}] SHAPE MISMATCH: mlx {an.shape} vs torch {bn.shape}")
        return
    d = np.abs(an - bn)
    rel = d / (np.abs(bn) + 1e-6)
    print(f"  [{name}]  max={d.max():.4g}  mean={d.mean():.4g}  "
          f"rel_mean={rel.mean():.4g}  "
          f"a_range=[{an.min():.3g}, {an.max():.3g}]  "
          f"b_range=[{bn.min():.3g}, {bn.max():.3g}]")
    if report_sample:
        print(f"       mlx[0,0,:3]={an.reshape(-1)[:3]}")
        print(f"       trc[0,0,:3]={bn.reshape(-1)[:3]}")


def _build_mlx_layer_from_torch(torch_blk: Block, mlx_cfg: MLXArgs) -> DeepseekV4DecoderLayer:
    """Construct MLX layer and copy torch_blk's weights directly
    (no quantization, no bundle I/O). This isolates architectural
    differences from quantization + bundle sanitize bugs."""
    mlx_layer = DeepseekV4DecoderLayer(mlx_cfg, layer_id=torch_blk.layer_id)
    import mlx.nn as nn

    def copy(mx_mod, torch_linear):
        """Copy torch nn.Linear weight → MLX Linear.weight in fp16."""
        w = torch_linear.weight.detach().float().numpy().astype(np.float16)
        mx_mod.weight = mx.array(w)
        if torch_linear.bias is not None:
            b = torch_linear.bias.detach().float().numpy().astype(np.float16)
            mx_mod.bias = mx.array(b)

    def copy_norm(mx_n, torch_n):
        w = torch_n.weight.detach().float().numpy().astype(np.float16)
        mx_n.weight = mx.array(w)

    def copy_param(mx_attr_path, torch_tensor):
        """Set an mx.array attribute on mlx_layer tree."""
        node = mlx_layer
        parts = mx_attr_path.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], mx.array(
            torch_tensor.detach().float().numpy().astype(np.float16)
        ))

    # Attention
    copy(mlx_layer.self_attn.wq_a, torch_blk.attn.wq_a)
    copy_norm(mlx_layer.self_attn.q_norm, torch_blk.attn.q_norm)
    copy(mlx_layer.self_attn.wq_b, torch_blk.attn.wq_b)
    copy(mlx_layer.self_attn.wkv, torch_blk.attn.wkv)
    copy_norm(mlx_layer.self_attn.kv_norm, torch_blk.attn.kv_norm)
    copy(mlx_layer.self_attn.wo_a, torch_blk.attn.wo_a)
    copy(mlx_layer.self_attn.wo_b, torch_blk.attn.wo_b)
    copy_param("self_attn.attn_sink", torch_blk.attn.attn_sink)

    # Norms
    copy_norm(mlx_layer.input_layernorm, torch_blk.attn_norm)
    copy_norm(mlx_layer.post_attention_layernorm, torch_blk.ffn_norm)

    # Gate
    copy_param("mlp.gate.weight", torch_blk.ffn.gate.weight)
    if torch_blk.ffn.gate.bias is not None:
        copy_param("mlp.gate.bias", torch_blk.ffn.gate.bias)
    if hasattr(torch_blk.ffn.gate, "tid2eid") and torch_blk.ffn.gate.tid2eid is not None:
        copy_param("mlp.gate.tid2eid", torch_blk.ffn.gate.tid2eid)

    # Shared expert
    copy(mlx_layer.mlp.shared_experts.gate_proj, torch_blk.ffn.shared_experts.w1)
    copy(mlx_layer.mlp.shared_experts.down_proj, torch_blk.ffn.shared_experts.w2)
    copy(mlx_layer.mlp.shared_experts.up_proj, torch_blk.ffn.shared_experts.w3)

    # Routed experts — stack w1/w2/w3 across experts for SwitchGLU
    n_e = mlx_cfg.n_routed_experts
    w1_stack = torch.stack([torch_blk.ffn.experts[e].w1.weight for e in range(n_e)])
    w2_stack = torch.stack([torch_blk.ffn.experts[e].w2.weight for e in range(n_e)])
    w3_stack = torch.stack([torch_blk.ffn.experts[e].w3.weight for e in range(n_e)])
    # w1 → gate_proj, w2 → down_proj, w3 → up_proj
    mlx_layer.mlp.switch_mlp.gate_proj.weight = _to_mx(w1_stack)
    mlx_layer.mlp.switch_mlp.down_proj.weight = _to_mx(w2_stack)
    mlx_layer.mlp.switch_mlp.up_proj.weight = _to_mx(w3_stack)

    # mHC
    for attr in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
                 "hc_attn_scale", "hc_ffn_scale"):
        copy_param(attr, getattr(torch_blk, attr))

    return mlx_layer


def run(source_dir: Path, layer_id: int):
    print(f"[torch] indexing + loading layer {layer_id}...")
    src_idx = ShardIndex(source_dir)
    cfg_json = json.loads((source_dir / "config.json").read_text())
    torch_cfg = DSV4Config.from_config_json(cfg_json)
    torch_blk = load_block_from_shards(layer_id, torch_cfg, src_idx).eval()

    print(f"[mlx] constructing MLX layer from torch weights (no quant)...")
    mlx_cfg = MLXArgs(**{k: v for k, v in cfg_json.items()
                         if k in {f.name for f in MLXArgs.__dataclass_fields__.values()}})
    mlx_layer = _build_mlx_layer_from_torch(torch_blk, mlx_cfg)

    # Same input
    torch.manual_seed(0)
    np.random.seed(0)
    B, L = 1, 8
    x_np = np.random.randn(B, L, torch_cfg.hc_mult, torch_cfg.dim).astype(np.float32) * 0.02
    x_t = torch.from_numpy(x_np).to(torch.bfloat16)
    x_m = mx.array(x_np.astype(np.float16))
    ids_np = np.random.randint(0, torch_cfg.vocab_size, (B, L)).astype(np.int64)
    ids_t = torch.from_numpy(ids_np)
    ids_m = mx.array(ids_np.astype(np.int32))
    fc = precompute_freqs_cis(
        torch_cfg.rope_head_dim, L, torch_cfg.original_seq_len,
        torch_cfg.rope_theta, torch_cfg.rope_factor,
        torch_cfg.beta_fast, torch_cfg.beta_slow,
    )

    print()
    print("=== Full layer forward diff (identical bf16 weights) ===")
    with torch.inference_mode():
        y_t = torch_blk(x_t, fc, ids_t)
    y_m = mlx_layer(x_m, mask=None, cache=None, input_ids=ids_m)
    mx.eval(y_m)
    diff("layer_output", y_m, y_t, report_sample=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--layer", type=int, default=3)
    args = ap.parse_args()
    run(args.source, args.layer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
