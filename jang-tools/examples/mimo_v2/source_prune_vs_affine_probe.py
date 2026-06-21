"""Source-side MiMo prune-only vs prune+affine probe.

This separates two variables for sub-105GB routed-expert experiments:

1. Pruning/slicing routed experts and router rows to a keep-map.
2. Applying low-bit affine QDQ to the kept weights.

It uses the streaming source checkpoint loader, so it does not build or load a
full MLX bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from jang_tools.mimo_v2.convert_jang import ExpertKeepMap, load_expert_keep_map
from layer_diff_probe import SourceRunner
from source_profile_probe import ProbeProfile, QuantizedSourceRunner, final_logits, top_tokens, torch_rel_stats


class PrunedSourceRunner(SourceRunner):
    def __init__(self, src: Path, expert_keep_map: ExpertKeepMap):
        super().__init__(src)
        self.expert_keep_map = expert_keep_map

    def moe(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        keep = self.expert_keep_map.indices_for_layer(layer_idx)
        keep_t = torch.tensor(keep, dtype=torch.long)
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        gate_w = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.weight",
            out_dtype=torch.float32,
        ).index_select(0, keep_t)
        bias = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias",
            out_dtype=torch.float32,
        ).index_select(0, keep_t)
        scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
        _, topk_local = torch.topk(scores + bias.view(1, -1), k=self.top_k, dim=-1, sorted=False)
        topk_w = scores.gather(1, topk_local)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(xf)
        for local_expert_idx in torch.unique(topk_local).tolist():
            slots = topk_local == int(local_expert_idx)
            token_idx, slot_idx = torch.where(slots)
            if token_idx.numel() == 0:
                continue
            expert_idx = keep[int(local_expert_idx)]
            expert_x = xf[token_idx]
            gate = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
            up = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
            down = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
            expert_y = F.linear(F.silu(F.linear(expert_x, gate)) * F.linear(expert_x, up), down)
            out.index_add_(0, token_idx, expert_y * topk_w[token_idx, slot_idx].unsqueeze(-1))
        return out.view(bsz, seq_len, hidden)


def prompt_ids(tokenizer, text: str, thinking: str) -> list[int]:
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], **template_kwargs)
    return tokenizer.encode(prompt)


def print_stats(label: str, ref: torch.Tensor, actual: torch.Tensor) -> None:
    rel, last_rel, maxerr = torch_rel_stats(ref, actual)
    print(f"{label} rel={rel:.6f} last_rel={last_rel:.6f} max={maxerr:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--expert-keep-map", type=Path, required=True)
    parser.add_argument("--keep-experts", type=int, required=True)
    parser.add_argument("--affine-profile", default="444g64")
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with only the number.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    parser.add_argument("--report-layers", default="0,1,2,3,4,8,16,24,32,40,47")
    parser.add_argument("--max-layer", type=int, default=47)
    args = parser.parse_args()

    keep_map = load_expert_keep_map(args.expert_keep_map, keep_experts=args.keep_experts)
    report_layers = {int(x) for x in args.report_layers.split(",") if x.strip()}

    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    ids = prompt_ids(tokenizer, args.prompt, args.thinking)

    base = SourceRunner(args.src)
    pruned = PrunedSourceRunner(args.src, keep_map)
    affine = QuantizedSourceRunner(args.src, ProbeProfile.parse(args.affine_profile), expert_keep_map=keep_map)

    h_base = base.embed(ids)
    h_pruned = pruned.embed(ids)
    h_affine = affine.embed(ids)
    print(f"tokens={len(ids)} keep_experts={keep_map.keep_experts} affine_profile={ProbeProfile.parse(args.affine_profile).name}")
    print_stats("embed prune_only", h_base, h_pruned)
    print_stats("embed prune_affine", h_base, h_affine)

    for layer_idx in range(min(base.cfg["num_hidden_layers"], args.max_layer + 1)):
        h_base = base.layer(layer_idx, h_base)
        h_pruned = pruned.layer(layer_idx, h_pruned)
        h_affine = affine.layer(layer_idx, h_affine)
        if layer_idx in report_layers:
            print(f"layer {layer_idx:02d}")
            print_stats("  prune_only ", h_base, h_pruned)
            print_stats("  prune_affine", h_base, h_affine)

    if args.max_layer >= base.cfg["num_hidden_layers"] - 1:
        logits_base = final_logits(base, h_base)
        logits_pruned = final_logits(pruned, h_pruned)
        logits_affine = final_logits(affine, h_affine)
        print("final_logits")
        print_stats("  prune_only ", logits_base, logits_pruned)
        print_stats("  prune_affine", logits_base, logits_affine)
        print("base_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_base):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
        print("prune_only_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_pruned):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
        print("prune_affine_top:")
        for token_id, text, value in top_tokens(tokenizer, logits_affine):
            print(f"  {token_id:6d} {text!r} {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
