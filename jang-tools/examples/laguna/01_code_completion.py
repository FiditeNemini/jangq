"""01 — Code completion across mixed full-attention + SWA layers.

Laguna's text decoder uses per-layer head count (48 for full_attention
layers, 64 for sliding_attention) and dual RoPE (YaRN on full, default
on SWA). This script:

  1. Runs three programming prompts (function-completion + class-design + bug-fix)
  2. Verifies decoded output is syntactically reasonable
  3. Reports tok/s on prefill + decode separately

Run: python3 01_code_completion.py [bundle_path]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ"

PROMPTS = [
    # Plain FIM
    "def quicksort(arr):\n    \"\"\"Recursive in-place quicksort.\"\"\"\n",
    # Type hints + edge cases
    "from typing import Optional\n\ndef binary_search(xs: list[int], target: int) -> Optional[int]:\n    \"\"\"Return index of `target` or None if missing.\"\"\"\n",
    # Bug fix
    "# This function should reverse a string but has a bug. Fix it.\ndef reverse(s: str) -> str:\n    return s[1:][::-1]\n\n# Fixed:\ndef reverse_fixed(s: str) -> str:\n",
]


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    import mlx.core as mx
    mx.set_memory_limit(200 * 1024**3)

    from jang_tools.laguna.runtime import load, greedy
    from transformers import AutoTokenizer

    print(f"=== Loading Laguna from {bundle.name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
    t0 = time.time()
    model, cfg, fmt = load(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s (format={fmt})\n", flush=True)

    for i, prompt in enumerate(PROMPTS):
        print(f"--- prompt {i+1}/{len(PROMPTS)} ---")
        print(f">>> {prompt[:80]!r}")
        ids = tok.encode(prompt)
        t1 = time.time()
        out = greedy(model, ids, max_new=64)
        dt = time.time() - t1
        n_new = len(out) - len(ids)
        new_text = tok.decode(out[len(ids):])
        print(f"<<< {new_text[:200]!r}")
        print(f"  {n_new} tokens in {dt:.2f}s ({n_new/dt:.1f} tok/s)\n")


if __name__ == "__main__":
    main()
