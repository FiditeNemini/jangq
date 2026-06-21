"""Greedy multi-token source-side continuation probe.

Decodes K tokens greedily (full re-forward per step, no cache) on up to three
source paths: full-precision base, prune-only, prune+affine. Separates
model-intrinsic behavior from pruning damage from affine quant damage for
pathologies that only appear during multi-token decode (e.g. repeated-token
loops inside think mode).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from transformers import AutoTokenizer

from jang_tools.mimo_v2.convert_jang import load_expert_keep_map
from layer_diff_probe import SourceRunner
from source_profile_probe import ProbeProfile, QuantizedSourceRunner, final_logits
from source_prune_vs_affine_probe import PrunedSourceRunner


def forward_logits(runner, ids: list[int]) -> torch.Tensor:
    h = runner.embed(ids)
    num_layers = int(runner.cfg["num_hidden_layers"])
    for layer_idx in range(num_layers):
        h = runner.layer(layer_idx, h)
    return final_logits(runner, h)


def greedy(runner, tokenizer, ids: list[int], steps: int, label: str) -> list[int]:
    cur = list(ids)
    out: list[int] = []
    for step in range(steps):
        t0 = time.monotonic()
        logits = forward_logits(runner, cur)
        nxt = int(torch.argmax(logits[0, -1, :]).item())
        out.append(nxt)
        cur.append(nxt)
        print(
            f"[{label}] step {step + 1}/{steps} token={nxt} "
            f"{tokenizer.decode([nxt])!r} ({time.monotonic() - t0:.0f}s)",
            flush=True,
        )
        if nxt == tokenizer.eos_token_id:
            break
    print(f"[{label}] TEXT: {tokenizer.decode(out)!r}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="on")
    parser.add_argument("--force-prefix", default="", help="Text appended after the chat template (e.g. '<think>').")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--paths", default="base,prune,affine")
    parser.add_argument("--expert-keep-map", type=Path)
    parser.add_argument("--keep-experts", type=int, default=0)
    parser.add_argument("--affine-profile", default="444g128")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.src, trust_remote_code=True)
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], **template_kwargs)
    if args.force_prefix:
        prompt = prompt + args.force_prefix
    ids = tokenizer.encode(prompt)
    print(f"prompt_tokens={len(ids)} thinking={args.thinking} prefix={args.force_prefix!r}", flush=True)

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    keep_map = None
    if any(p in ("prune", "affine") for p in paths):
        if not args.expert_keep_map or not args.keep_experts:
            raise SystemExit("prune/affine paths need --expert-keep-map and --keep-experts")
        keep_map = load_expert_keep_map(args.expert_keep_map, keep_experts=args.keep_experts)

    for path in paths:
        if path == "base":
            runner = SourceRunner(args.src)
        elif path == "prune":
            runner = PrunedSourceRunner(args.src, keep_map)
        elif path == "affine":
            profile = ProbeProfile.parse(args.affine_profile)
            runner = QuantizedSourceRunner(args.src, profile, expert_keep_map=keep_map)
        else:
            raise SystemExit(f"unknown path {path}")
        greedy(runner, tokenizer, ids, args.steps, path)
        del runner
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
