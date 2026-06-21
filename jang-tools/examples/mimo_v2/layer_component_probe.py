"""Component-level source-vs-QDQ probe for MiMo-V2.5 decoder layers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from layer_diff_probe import SourceRunner, rmsnorm
from source_profile_probe import ProbeProfile, QuantizedSourceRunner, final_logits, torch_rel_stats, top_tokens


def format_stats(label: str, src: torch.Tensor, qdq: torch.Tensor) -> str:
    rel, last_rel, maxerr = torch_rel_stats(src, qdq)
    src_rms = torch.sqrt(torch.mean(src.float() * src.float())).item()
    q_rms = torch.sqrt(torch.mean(qdq.float() * qdq.float())).item()
    return (
        f"{label:18s} rel={rel:.6f} last_rel={last_rel:.6f} "
        f"max={maxerr:.6f} src_rms={src_rms:.6f} q_rms={q_rms:.6f}"
    )


def layer_components(runner: SourceRunner, layer_idx: int, h: torch.Tensor) -> dict[str, torch.Tensor]:
    ln1 = runner.idx.read_passthrough(
        f"model.layers.{layer_idx}.input_layernorm.weight",
        out_dtype=torch.float32,
    )
    attn_in = rmsnorm(h, ln1, runner.eps)
    attn_out = runner.attention(layer_idx, attn_in)
    post_attn = h + attn_out
    ln2 = runner.idx.read_passthrough(
        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        out_dtype=torch.float32,
    )
    mlp_in = rmsnorm(post_attn, ln2, runner.eps)
    if int(runner.cfg["moe_layer_freq"][layer_idx]):
        mlp_out = runner.moe(layer_idx, mlp_in)
    else:
        mlp_out = runner.dense_mlp(layer_idx, mlp_in)
    output = post_attn + mlp_out
    return {
        "input": h,
        "attn_in": attn_in,
        "attn_out": attn_out,
        "post_attn": post_attn,
        "mlp_in": mlp_in,
        "mlp_out": mlp_out,
        "output": output,
    }


def router_topk(runner: SourceRunner, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
    bsz, seq_len, hidden = x.shape
    xf = x.reshape(-1, hidden)
    gate_w = runner.idx.read_passthrough(f"model.layers.{layer_idx}.mlp.gate.weight", out_dtype=torch.float32)
    bias = runner.idx.read_passthrough(
        f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias",
        out_dtype=torch.float32,
    )
    scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
    _, topk_idx = torch.topk(scores + bias.view(1, -1), k=runner.top_k, dim=-1, sorted=True)
    return topk_idx.view(bsz, seq_len, runner.top_k)


def route_overlap(src_idx: torch.Tensor, q_idx: torch.Tensor) -> str:
    src_flat = src_idx.reshape(-1, src_idx.shape[-1])
    q_flat = q_idx.reshape(-1, q_idx.shape[-1])
    overlaps = []
    exact = 0
    for src_row, q_row in zip(src_flat, q_flat):
        src_set = {int(x) for x in src_row.tolist()}
        q_set = {int(x) for x in q_row.tolist()}
        hit = len(src_set & q_set)
        overlaps.append(hit)
        exact += int(hit == src_idx.shape[-1])
    overlap = torch.tensor(overlaps, dtype=torch.float32)
    total = src_flat.shape[0]
    return (
        f"route_overlap     mean={float(overlap.mean()):.3f}/{src_idx.shape[-1]} "
        f"min={int(overlap.min())} exact={exact}/{total}"
    )


def prompt_ids(tokenizer, text: str, thinking: str) -> list[int]:
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], **template_kwargs)
    return tokenizer.encode(prompt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--layers", default="44,45,46,47")
    parser.add_argument("--prompt", default="Name the capital city of France.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if not layers:
        raise ValueError("--layers must contain at least one layer index")
    max_layer = max(layers)

    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    ids = prompt_ids(tokenizer, args.prompt, args.thinking)
    profile = ProbeProfile.parse(args.profile)
    src = SourceRunner(args.src)
    qdq = QuantizedSourceRunner(args.src, profile)
    h_src = src.embed(ids)
    h_q = qdq.embed(ids)

    print(f"profile={profile.name} tokens={len(ids)} layers={layers}")
    print(format_stats("embed", h_src, h_q))

    for layer_idx in range(max_layer + 1):
        comp_src = layer_components(src, layer_idx, h_src)
        comp_q = layer_components(qdq, layer_idx, h_q)
        if layer_idx in layers:
            print(f"\nlayer {layer_idx:02d}")
            for key in ("input", "attn_in", "attn_out", "post_attn", "mlp_in", "mlp_out", "output"):
                print(format_stats(key, comp_src[key], comp_q[key]))
            if int(src.cfg["moe_layer_freq"][layer_idx]):
                src_route = router_topk(src, layer_idx, comp_src["mlp_in"])
                q_route = router_topk(src, layer_idx, comp_q["mlp_in"])
                print(route_overlap(src_route, q_route))
                src_weights_q_input = src.moe(layer_idx, comp_q["mlp_in"])
                q_weights_src_input = qdq.moe(layer_idx, comp_src["mlp_in"])
            else:
                src_weights_q_input = src.dense_mlp(layer_idx, comp_q["mlp_in"])
                q_weights_src_input = qdq.dense_mlp(layer_idx, comp_src["mlp_in"])
            print(format_stats("mlp_srcW_qX", comp_src["mlp_out"], src_weights_q_input))
            print(format_stats("mlp_qW_srcX", comp_src["mlp_out"], q_weights_src_input))
        h_src = comp_src["output"]
        h_q = comp_q["output"]

    if max_layer >= src.cfg["num_hidden_layers"] - 1:
        logits_src = final_logits(src, h_src)
        logits_q = final_logits(qdq, h_q)
        print()
        print(format_stats("final_logits", logits_src, logits_q))
        print("source_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_src):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
        print("quant_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_q):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
