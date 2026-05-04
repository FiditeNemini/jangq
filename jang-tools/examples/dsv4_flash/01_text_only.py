"""01 — Text-only chat across all 3 DSV4-Flash modes (chat / think / think_max).

Demonstrates the canonical runtime entry point and the three reasoning
modes the model was trained for. Also prints the parsed reasoning vs
content split so you can verify no leak (no `<think>` in `.content`).

Modes:
  • chat       → enable_thinking=False, fast, no reasoning
  • think      → enable_thinking=True, reasoning_effort=None  (Think High)
  • think_max  → enable_thinking=True, reasoning_effort='max' (Think Max)

Run: python3 01_text_only.py [bundle_path]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"

PROMPTS = [
    ("chat",      "What is the capital of France? Answer in one word."),
    ("think",     "What is 17 + 28?"),
    ("think_max", "If a train leaves Paris at 9am at 80km/h and another leaves Lyon (450 km away) at 10am at 100km/h heading toward each other, when do they meet?"),
]


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

    for mode, prompt in PROMPTS:
        # think modes: bigger budget, sampled (T=0.6) — required for COT.
        # chat: greedy is fine.
        opts = GenerateOptions(mode=mode, max_tokens={"chat":64,"think":1024,"think_max":4096}[mode])
        print(f"--- mode={mode} ---")
        print(f"USER: {prompt}")
        t1 = time.time()
        res = generate(model, tok, str(bundle),
                       messages=[{"role": "user", "content": prompt}],
                       opts=opts)
        dt = time.time() - t1
        if res.error:
            print(f"ERROR: {res.error}"); continue
        if res.reasoning_content:
            preview = res.reasoning_content[:200].replace("\n", " ")
            ellipsis = "…" if len(res.reasoning_content) > 200 else ""
            print(f"REASONING ({len(res.reasoning_content)} chars): {preview}{ellipsis}")
        print(f"CONTENT:   {res.content[:300]!r}")
        print(f"  tokens={res.n_tokens}  saw_close={res.saw_think_close}  truncated={res.truncated}  finish={res.finish_reason}  t={dt:.1f}s\n")

        # Leak check
        for tag in ("<think>", "</think>", "｜DSML｜"):
            assert tag not in res.content, f"LEAK: {tag!r} in content!"


if __name__ == "__main__":
    main()
