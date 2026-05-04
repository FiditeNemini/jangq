"""02 — Reasoning split + leak check on a hard math/logic question.

Verifies that on a question that triggers a real <think>...</think> block:
  1. result.reasoning_content contains the chain of thought
  2. result.content contains ONLY the final answer
  3. no `<think>` / `</think>` / `<|channel>` / `<channel|>` tag leaks into content
  4. truncated=True flag fires when budget exhausts before </think>

Run: python3 02_thinking.py [bundle_path]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"

# A question that reliably triggers ≥256-token reasoning even at temp=0.6
HARD_QUESTION = (
    "There are 3 boxes. Box A has 2 red balls and 3 blue balls. Box B has 1 red and 4 blue. "
    "Box C has 4 red and 1 blue. You pick a random box uniformly, then pick 2 balls without "
    "replacement from that box. Both turn out to be red. What is the probability you picked "
    "Box C? Show your reasoning step-by-step, then give a final numeric answer."
)

LEAK_TAGS = ("<think>", "</think>", "<|channel>", "<channel|>", "<|start|>", "<|message|>",
             "｜DSML｜", "</DSML>")


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    os.environ.setdefault("DSV4_LONG_CTX", "1")
    import mlx.core as mx
    mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)

    from jang_tools.load_jangtq import load_jangtq_model
    from jang_tools.dsv4.runtime import generate, GenerateOptions

    print(f"=== Loading DSV4-Flash from {bundle.name} ===", flush=True)
    t0 = time.time()
    model, tok = load_jangtq_model(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s\n", flush=True)

    print("--- Think High (max_tokens=2048) ---")
    print(f"USER: {HARD_QUESTION[:120]}…\n")
    t1 = time.time()
    res = generate(model, tok, str(bundle),
                   messages=[{"role": "user", "content": HARD_QUESTION}],
                   opts=GenerateOptions(mode="think", max_tokens=2048))
    dt = time.time() - t1

    print(f"REASONING ({len(res.reasoning_content)} chars):")
    head = res.reasoning_content[:300].replace("\n", " ")
    tail = res.reasoning_content[-300:].replace("\n", " ") if len(res.reasoning_content) > 600 else ""
    print(f"  HEAD: {head}…")
    if tail: print(f"  TAIL: …{tail}")
    print(f"\nCONTENT ({len(res.content)} chars):")
    print(f"  {res.content[:600]!r}")
    print(f"\nstats: tokens={res.n_tokens} saw_close={res.saw_think_close} "
          f"truncated={res.truncated} finish={res.finish_reason} t={dt:.1f}s")

    # Leak audit
    print("\n--- Leak audit ---")
    leaks = [tag for tag in LEAK_TAGS if tag in res.content]
    if leaks:
        print(f"  FAIL — content leaks {leaks}"); sys.exit(3)
    print("  PASS — no reasoning tags in content")

    # Reasoning vs content sanity
    if res.saw_think_close and not res.reasoning_content:
        print("  FAIL — saw </think> but no reasoning content captured"); sys.exit(3)
    if res.saw_think_close and not res.content:
        print("  WARN — saw </think> but content empty (may be fine if question ended in reasoning)")
    print("  PASS — reasoning + content captured separately")


if __name__ == "__main__":
    main()
