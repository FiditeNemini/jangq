"""01 — Chat-template multi-turn for Mistral 3.5 text decoder.

Demonstrates the chat template (Mistral instruct format) with three
single-turn prompts. No `<think>` tags — Mistral 3.5 doesn't have
trained reasoning mode at the time of this example.

Run: python3 01_chat.py [bundle_path]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/Mistral-Medium-3.5-128B-JANGTQ"

PROMPTS = [
    "Briefly explain the difference between TCP and UDP.",
    "Translate 'Good morning, how are you?' to French.",
    "Write a one-paragraph summary of the Pythagorean theorem.",
]


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    import mlx.core as mx
    mx.set_memory_limit(200 * 1024**3)

    from jang_tools.mistral3.runtime import load
    from transformers import AutoTokenizer

    print(f"=== Loading Mistral 3.5 from {bundle.name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
    t0 = time.time()
    model, cfg, fmt = load(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s (format={fmt})\n", flush=True)

    for i, prompt in enumerate(PROMPTS):
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(text)
        print(f"--- prompt {i+1}/{len(PROMPTS)} ---")
        print(f"USER: {prompt}")
        out = list(ids)
        x = mx.array([ids], dtype=mx.uint32)
        t1 = time.time()
        logits, caches = model(x, caches=None)
        for _ in range(96):
            nxt = int(mx.argmax(logits[0, -1]).item())
            out.append(nxt)
            if nxt == tok.eos_token_id: break
            x = mx.array([[nxt]], dtype=mx.uint32)
            logits, caches = model(x, caches=caches)
        dt = time.time() - t1
        n_new = len(out) - len(ids)
        new_text = tok.decode(out[len(ids):])
        print(f"ASSISTANT: {new_text}")
        print(f"  {n_new} tokens in {dt:.2f}s ({n_new/dt:.1f} tok/s)\n")


if __name__ == "__main__":
    main()
