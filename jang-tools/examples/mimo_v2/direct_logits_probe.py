"""Direct one-forward logits probe for MiMo bundles."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--prompt", default="What is 2 + 2? Answer with only the number.")
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="off")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    model, tokenizer = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if args.thinking == "on":
        template_kwargs["enable_thinking"] = True
    elif args.thinking == "off":
        template_kwargs["enable_thinking"] = False
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], **template_kwargs)
    ids = tokenizer.encode(prompt)
    logits = model(mx.array([ids], dtype=mx.int32), cache=None)
    mx.eval(logits)
    tail = logits[0, -1, :]
    top_idx = mx.argsort(tail)[-args.top_k:][::-1]
    print(f"bundle={args.bundle}")
    print(f"prompt_tokens={len(ids)}")
    for token_id in top_idx.tolist():
        print(f"{int(token_id):8d} {tokenizer.decode([int(token_id)])!r} {float(tail[int(token_id)].item()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
