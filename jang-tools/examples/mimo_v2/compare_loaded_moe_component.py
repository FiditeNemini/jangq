"""Compare loaded MLX MoE output against source-side pruned QDQ.

This isolates the runtime MoE boundary after the converter/loader weight check:
given the exact hidden state entering one loaded MLX layer's MoE, compare
MLX routing/output against the source checkpoint with the bundle's keep-map and
the same affine QDQ profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401
from jang_tools.mimo_v2.convert_jang import ExpertKeepMap
from source_profile_probe import ProbeProfile, QuantizedSourceRunner


def torch_from_mx(x: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.array(x.astype(mx.float32)))


def rel_stats(ref: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float, float]:
    ref = ref.float()
    actual = actual.float()
    diff = ref - actual
    rmse = torch.sqrt(torch.mean(diff * diff))
    rms = torch.sqrt(torch.mean(ref * ref)) + 1e-12
    last = diff[:, -1, :]
    last_ref = ref[:, -1, :]
    last_rel = torch.sqrt(torch.mean(last * last)) / (torch.sqrt(torch.mean(last_ref * last_ref)) + 1e-12)
    return float(rmse / rms), float(last_rel), float(diff.abs().max()), float(diff.abs().mean())


def route_overlap(mx_idx: torch.Tensor, torch_idx: torch.Tensor) -> tuple[float, int, int]:
    mx_flat = mx_idx.reshape(-1, mx_idx.shape[-1])
    torch_flat = torch_idx.reshape(-1, torch_idx.shape[-1])
    hits: list[int] = []
    exact = 0
    for a, b in zip(mx_flat, torch_flat):
        aset = {int(x) for x in a.tolist()}
        bset = {int(x) for x in b.tolist()}
        hit = len(aset & bset)
        hits.append(hit)
        exact += int(hit == mx_idx.shape[-1])
    return float(sum(hits) / max(len(hits), 1)), int(min(hits)), exact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--profile", default="444g64")
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with only the number.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    args = parser.parse_args()

    cfg = json.loads((args.bundle / "config.json").read_text())
    keep_layers = {
        int(layer): [int(x) for x in experts]
        for layer, experts in cfg["runtime"]["expert_keep_map"]["layers"].items()
    }
    keep_map = ExpertKeepMap(keep_layers)

    model, tokenizer = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], **template_kwargs)
    ids = tokenizer.encode(prompt)

    h = model.model.embed_tokens(mx.array([ids], dtype=mx.int32))
    for idx in range(args.layer):
        layer = model.model.layers[idx]
        mask = create_attention_mask(h, None, window_size=layer.self_attn.sliding_window)
        h = layer(h, mask=mask, cache=None)
        mx.eval(h)

    layer = model.model.layers[args.layer]
    mask = create_attention_mask(h, None, window_size=layer.self_attn.sliding_window)
    post_attn = h + layer.self_attn(layer.input_layernorm(h), mask=mask, cache=None)
    mlp_in = layer.post_attention_layernorm(post_attn)
    mlx_out = layer.mlp(mlp_in)
    mx_idx, mx_w = layer.mlp.gate(mlp_in.reshape(-1, mlp_in.shape[-1]))
    mx.eval(mlp_in, mlx_out, mx_idx, mx_w)

    profile = ProbeProfile.parse(args.profile)
    qdq = QuantizedSourceRunner(args.src, profile, keep_map)
    x_t = torch_from_mx(mlp_in)
    qdq_out = qdq.moe(args.layer, x_t)

    keep = torch.tensor(keep_map.indices_for_layer(args.layer), dtype=torch.long)
    gate_w = qdq.idx.read_passthrough(
        f"model.layers.{args.layer}.mlp.gate.weight",
        out_dtype=torch.float32,
    ).index_select(0, keep)
    bias = qdq.idx.read_passthrough(
        f"model.layers.{args.layer}.mlp.gate.e_score_correction_bias",
        out_dtype=torch.float32,
    ).index_select(0, keep)
    scores = torch.sigmoid(F.linear(x_t.reshape(-1, x_t.shape[-1]).float(), gate_w.float()))
    _, torch_idx = torch.topk(scores + bias.view(1, -1), k=qdq.top_k, dim=-1, sorted=False)
    torch_w = scores.gather(1, torch_idx)
    torch_w = torch_w / (torch_w.sum(dim=-1, keepdim=True) + 1e-20)

    mlx_out_t = torch_from_mx(mlx_out)
    mx_idx_t = torch_from_mx(mx_idx).to(torch.long)
    mx_w_t = torch_from_mx(mx_w)
    mean_hit, min_hit, exact = route_overlap(mx_idx_t, torch_idx)
    rel, last_rel, maxerr, mae = rel_stats(qdq_out, mlx_out_t)
    w_rel, _, w_max, w_mae = rel_stats(torch_w[:, None, :], mx_w_t[:, None, :])

    print(f"bundle={args.bundle}")
    print(f"profile={profile.name} layer={args.layer} prompt_tokens={len(ids)}")
    print(f"mlp_in_shape={tuple(x_t.shape)}")
    print(
        f"route_overlap mean={mean_hit:.3f}/{qdq.top_k} min={min_hit} "
        f"exact={exact}/{torch_idx.shape[0]}"
    )
    print(f"route_weight rel={w_rel:.8f} maxerr={w_max:.8f} mae={w_mae:.8f}")
    print(f"moe_output rel={rel:.8f} last_rel={last_rel:.8f} maxerr={maxerr:.8f} mae={mae:.8f}")
    print(f"mlx_topk_first={mx_idx_t[0].tolist()}")
    print(f"torch_topk_first={torch_idx[0].tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
