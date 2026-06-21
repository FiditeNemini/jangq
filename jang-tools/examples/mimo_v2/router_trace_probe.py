"""Collect MiMo V2.x source-side router traces for activation-guided profiles."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from jang_tools.mimo_v2.router_trace import RouteAccumulator

from layer_diff_probe import SourceRunner, rmsnorm


class TracingSourceRunner(SourceRunner):
    def __init__(self, src: Path, accumulator: RouteAccumulator):
        super().__init__(src)
        self.accumulator = accumulator

    def moe(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, hidden = x.shape
        xf = x.reshape(-1, hidden)
        gate_w = self.idx.read_passthrough(f"model.layers.{layer_idx}.mlp.gate.weight", out_dtype=torch.float32)
        bias = self.idx.read_passthrough(
            f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias",
            out_dtype=torch.float32,
        )
        scores = torch.sigmoid(F.linear(xf.float(), gate_w.float()))
        _, topk_idx = torch.topk(scores + bias.view(1, -1), k=self.top_k, dim=-1, sorted=False)
        topk_w = scores.gather(1, topk_idx)
        topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)
        self.accumulator.add(layer=layer_idx, indices=topk_idx, weights=topk_w)

        out = torch.zeros_like(xf)
        for expert_idx in torch.unique(topk_idx).tolist():
            slots = topk_idx == int(expert_idx)
            token_idx, slot_idx = torch.where(slots)
            if token_idx.numel() == 0:
                continue
            expert_x = xf[token_idx]
            gate = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
            up = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
            down = self.cached_tensor(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")
            expert_y = F.linear(F.silu(F.linear(expert_x, gate)) * F.linear(expert_x, up), down)
            out.index_add_(0, token_idx, expert_y * topk_w[token_idx, slot_idx].unsqueeze(-1))
        return out.view(bsz, seq_len, hidden)


def _prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompt_file:
        prompts.extend(
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if prompts:
        return prompts
    return [
        "What is 2 + 2? Answer in one short sentence.",
        "Name the capital city of France.",
        "Reply with exactly three comma-separated colors.",
        "Remember the word cerulean. What word did I ask you to remember?",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    parser.add_argument("--max-layer", type=int, default=47)
    parser.add_argument("--progress-every", type=int, default=1,
                        help="Print progress every N layers; 0 disables layer progress.")
    args = parser.parse_args()

    prompts = _prompts(args)
    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    acc = RouteAccumulator(num_experts=int(json.loads((args.src / "config.json").read_text())["n_routed_experts"]))
    runner = TracingSourceRunner(args.src, acc)
    total_layers = min(runner.cfg["num_hidden_layers"], args.max_layer + 1)
    print(
        f"trace_start prompts={len(prompts)} layers={total_layers} thinking={args.thinking}",
        flush=True,
    )

    t0 = time.monotonic()
    for prompt_idx, raw_prompt in enumerate(prompts, start=1):
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if args.thinking == "on":
            template_kwargs["enable_thinking"] = True
        elif args.thinking == "off":
            template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}], **template_kwargs)
        ids = tokenizer.encode(prompt)
        print(
            f"prompt_start {prompt_idx}/{len(prompts)} tokens={len(ids)} text={raw_prompt[:80]!r}",
            flush=True,
        )
        h = runner.embed(ids)
        for layer_idx in range(total_layers):
            layer_t0 = time.monotonic()
            h = runner.layer(layer_idx, h)
            if args.progress_every and (
                layer_idx == 0
                or layer_idx == total_layers - 1
                or (layer_idx + 1) % args.progress_every == 0
            ):
                elapsed = time.monotonic() - t0
                print(
                    f"prompt={prompt_idx}/{len(prompts)} layer={layer_idx:02d}/{total_layers - 1:02d} "
                    f"layer_sec={time.monotonic() - layer_t0:.2f} elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    trace = acc.to_trace(
        source=str(args.src),
        prompts=prompts,
        metadata={"thinking": args.thinking, "max_layer": args.max_layer},
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(args.out, flush=True)
    for layer_s, layer in sorted(trace["layers"].items(), key=lambda item: int(item[0])):
        top = layer["prob_mass_top"][:8]
        print(
            f"layer {int(layer_s):02d} tokens={layer['token_count']} "
            f"observed={layer['observed_experts']} top={top}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
