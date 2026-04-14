#!/usr/bin/env python3
"""
Plan 5 validation: load Gemma-4-26B-A4B-it-JANG_4M two ways and confirm
greedy decode produces identical token sequences.

  Path A (baseline):  jang_tools.loader.load_jang_vlm_model on the source
                      directory /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M
  Path B (bundle):    jang_tools.jangspec.bundle_loader.load_jang_model_from_bundle
                      on /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec

Usage:
    python3 jang-tools/scripts/validate_bundle_gemma4.py

Environment:
    GEMMA_BUNDLE         override bundle path
    GEMMA_SOURCE         override source JANG_4M dir
    GEMMA_PROMPT         override prompt (default: "The capital of France is")
    GEMMA_MAX_TOKENS     override max generated tokens (default: 16)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_SOURCE = "/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M"
DEFAULT_BUNDLE = "/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec"
DEFAULT_PROMPT = "The capital of France is"
DEFAULT_MAX = 16


def _generate_greedy(model, tokenizer, prompt: str, max_tokens: int) -> list[int]:
    """Run greedy decode and return the generated token IDs (no sampling)."""
    import mlx.core as mx

    input_ids = tokenizer.encode(prompt, return_tensors=None)
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    tokens = list(input_ids)
    cache = None

    for _ in range(max_tokens):
        x = mx.array([tokens[-1]] if cache is not None else tokens)
        x = x.reshape(1, -1)
        logits = model(x, cache=cache)
        if hasattr(logits, "logits"):
            logits = logits.logits
        next_token = int(mx.argmax(logits[0, -1, :]).item())
        tokens.append(next_token)
        # mlx-lm models construct cache on first call; subsequent calls
        # need the same cache instance. Pull it from the model if present.
        if cache is None:
            try:
                from mlx_lm.models.cache import make_prompt_cache
                cache = make_prompt_cache(model)
                # Replay the prompt through the new cache to seed it.
                _ = model(mx.array([tokens[:-1]]), cache=cache)
            except Exception:
                cache = None  # fall through and re-encode each step

    # Return only the newly generated tokens.
    return tokens[len(input_ids):]


def main() -> int:
    source = Path(os.environ.get("GEMMA_SOURCE", DEFAULT_SOURCE))
    bundle = Path(os.environ.get("GEMMA_BUNDLE", DEFAULT_BUNDLE))
    prompt = os.environ.get("GEMMA_PROMPT", DEFAULT_PROMPT)
    max_tokens = int(os.environ.get("GEMMA_MAX_TOKENS", str(DEFAULT_MAX)))

    if not source.exists():
        print(f"  source missing: {source}")
        return 2
    if not (bundle / "jangspec.json").exists():
        print(f"  bundle missing: {bundle}")
        print(f"  build with: jang spec build {source} --out {bundle}")
        return 2

    print("=" * 64)
    print(f"  Plan 5 — Gemma-4-26B-A4B bundle validation")
    print(f"  prompt:     {prompt!r}")
    print(f"  max_tokens: {max_tokens}")
    print("=" * 64)

    # --- Path A: source directory ---
    print("\n[A] Loading source via jang_tools.loader ...")
    t0 = time.time()
    from jang_tools.loader import load_jang_vlm_model
    model_a, tok_a = load_jang_vlm_model(str(source))
    print(f"    loaded in {time.time() - t0:.1f}s")

    print("    generating ...")
    t0 = time.time()
    tokens_a = _generate_greedy(model_a, tok_a, prompt, max_tokens)
    print(f"    generated {len(tokens_a)} tokens in {time.time() - t0:.1f}s")
    decoded_a = tok_a.decode(tokens_a)
    print(f"    text:   {decoded_a!r}")
    print(f"    tokens: {tokens_a}")

    # Free Path A weights aggressively before loading Path B — the model
    # is ~16 GB resident and we don't want both in RAM at once.
    del model_a
    import gc
    gc.collect()

    # --- Path B: .jangspec bundle ---
    print("\n[B] Loading bundle via jangspec.bundle_loader ...")
    t0 = time.time()
    from jang_tools.jangspec.bundle_loader import load_jang_model_from_bundle
    model_b, tok_b = load_jang_model_from_bundle(bundle)
    print(f"    loaded in {time.time() - t0:.1f}s")

    print("    generating ...")
    t0 = time.time()
    tokens_b = _generate_greedy(model_b, tok_b, prompt, max_tokens)
    print(f"    generated {len(tokens_b)} tokens in {time.time() - t0:.1f}s")
    decoded_b = tok_b.decode(tokens_b)
    print(f"    text:   {decoded_b!r}")
    print(f"    tokens: {tokens_b}")

    # --- Compare ---
    print("\n" + "=" * 64)
    if tokens_a == tokens_b:
        print("  TOKEN-LEVEL MATCH — bundle is functionally equivalent to source")
        return 0
    else:
        print("  MISMATCH — bundle path produces different tokens")
        print(f"     source tokens: {tokens_a}")
        print(f"     bundle tokens: {tokens_b}")
        # First divergence
        for i, (a, b) in enumerate(zip(tokens_a, tokens_b)):
            if a != b:
                print(f"     first diff at index {i}: source={a} bundle={b}")
                break
        return 1


if __name__ == "__main__":
    sys.exit(main())
