"""02 — Thinking-mode chat for Laguna (GLM-thinking-v5 family).

Laguna's chat template uses `<think>...</think>` tags when
`enable_thinking=True`. This script:
  1. Sends a coding question with thinking on
  2. Runs greedy decode
  3. Splits reasoning from content via partition('</think>')
  4. Verifies no leak

Run: python3 02_thinking.py [bundle_path]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ"

QUESTION = (
    "I have a Python list of integers. Write a one-liner that returns the "
    "sum of the squares of the even numbers in the list. Think briefly first, "
    "then give the one-liner."
)

LEAK_TAGS = ("<think>", "</think>", "<thought>", "</thought>")


def split_reasoning(text: str) -> tuple[str, str]:
    if "</think>" in text:
        r, _, c = text.partition("</think>")
        return r.lstrip("<think>").strip(), c.strip()
    return "", text


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    import mlx.core as mx
    mx.set_memory_limit(200 * 1024**3)

    from jang_tools.laguna.runtime import load
    from transformers import AutoTokenizer

    print(f"=== Loading Laguna from {bundle.name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
    t0 = time.time()
    model, cfg, fmt = load(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s (format={fmt})\n", flush=True)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ids = tok.encode(prompt)
    print(f"USER: {QUESTION[:100]}…\n")

    out = list(ids)
    x = mx.array([ids], dtype=mx.uint32)
    t1 = time.time()
    logits, caches = model(x, caches=None)
    for _ in range(512):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        if nxt == tok.eos_token_id: break
        x = mx.array([[nxt]], dtype=mx.uint32)
        logits, caches = model(x, caches=caches)
    dt = time.time() - t1
    new_text = tok.decode(out[len(ids):])
    n_new = len(out) - len(ids)
    print(f"  {n_new} tokens in {dt:.2f}s ({n_new/dt:.1f} tok/s)\n")

    reasoning, content = split_reasoning(new_text)
    if reasoning:
        head = reasoning[:200].replace("\n", " ")
        print(f"REASONING ({len(reasoning)} chars): {head}…\n")
    print(f"CONTENT: {content[:300]!r}\n")

    leaks = [t for t in LEAK_TAGS if t in content]
    if leaks:
        print(f"FAIL — content leaks {leaks}"); sys.exit(3)
    print("PASS — reasoning/content split clean")


if __name__ == "__main__":
    main()
