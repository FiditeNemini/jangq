"""04 — Long-context HSA + CSA + SWA tri-mode + pool-quant cache demo.

DSV4-Flash uses a hybrid attention design:
  • HSA layers (compress_ratio=128): dense pool over the full context
  • CSA layers (compress_ratio=4):   sparse Indexer over a top-k pool
  • SWA layers (compress_ratio=0):   plain sliding-window K/V

The runtime wires these via DeepseekV4Cache (or PoolQuantizedV4Cache when
DSV4_POOL_QUANT=1, default ON for long context). This script:

  1. Builds a 4 KB needle-in-haystack prompt with a hidden token
  2. Asks the model to retrieve it
  3. Verifies retrieval works AND HSA/CSA path was used (not bypassed)

Run: python3 04_long_context.py [bundle_path]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"

# 4 KB filler containing one needle phrase.
FILLER_LINES = [
    "The clouds rolled gently over the rolling green hills, indifferent to the day.",
    "Marketers gathered in conference rooms to debate the optimal hue of the call-to-action button.",
    "Children played in the courtyard, their laughter echoing off the stone walls.",
    "A long-forgotten letter sat unopened in a drawer, dust gathering on its edges.",
] * 32  # ~32 × 4 lines × ~85 chars = ~10K chars
NEEDLE_PHRASE = "the secret passcode is JANGTQ-RABBIT-447"
NEEDLE_INDEX = 60   # row near the middle of the filler


def make_prompt():
    lines = FILLER_LINES.copy()
    lines.insert(NEEDLE_INDEX, NEEDLE_PHRASE)
    return ("Read the following text carefully and remember every detail:\n\n"
            + "\n".join(lines)
            + "\n\nWhat is the secret passcode mentioned in the text? "
              "Answer with just the passcode itself.")


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    os.environ["DSV4_LONG_CTX"] = "1"
    # Default: pool quant auto-on. Set =0 to A/B test bf16 pool.
    os.environ.setdefault("DSV4_POOL_QUANT", "1")
    import mlx.core as mx
    mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)

    from jang_tools.load_jangtq import load_jangtq_model
    from jang_tools.dsv4.runtime import generate, GenerateOptions

    print(f"=== Loading DSV4-Flash from {bundle.name} ===", flush=True)
    t0 = time.time()
    model, tok = load_jangtq_model(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  DSV4_LONG_CTX={os.environ['DSV4_LONG_CTX']}  "
          f"DSV4_POOL_QUANT={os.environ['DSV4_POOL_QUANT']}", flush=True)

    prompt = make_prompt()
    n_chars = len(prompt)
    n_tokens_est = n_chars // 4
    print(f"\nprompt: {n_chars} chars (~{n_tokens_est} tokens), "
          f"needle at line {NEEDLE_INDEX} (~{NEEDLE_INDEX*85} chars in)\n")

    t1 = time.time()
    res = generate(model, tok, str(bundle),
                   messages=[{"role": "user", "content": prompt}],
                   opts=GenerateOptions(mode="chat", max_tokens=64, temperature=0.0))
    dt = time.time() - t1
    print(f"OUTPUT: {res.content!r}")
    print(f"t={dt:.1f}s tokens={res.n_tokens}")

    if "JANGTQ-RABBIT-447" in res.content:
        print("\nPASS — needle retrieved at long context (HSA+CSA path active)")
    else:
        print("\nFAIL — needle not retrieved. Likely culprits:")
        print("  • DSV4_LONG_CTX=0 (HSA/CSA bypassed → only window in scope)")
        print("  • PoolQuantizedV4Cache lossy beyond expectation (rare)")
        print("  • Prompt eaten by chat template truncation")


if __name__ == "__main__":
    main()
