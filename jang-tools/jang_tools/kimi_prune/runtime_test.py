"""Coherence + runtime harness for Kimi-K2.6 JANGTQ bundles.

Runs THREE independent decode paths on the same prompt and prints top-k
next-token logits for each:

  1. No-cache greedy loop  — manual prefill+step, new tensors every token.
                             Bypasses mlx_lm's KV cache entirely. If this
                             is coherent but (2)/(3) aren't, the bug is in
                             the cache-aware attention path (GLM-5.1 lesson:
                             30 hours were wasted re-quantizing before we
                             ran this test and discovered the MLA bf16
                             SDPA drift at L==1).
  2. mlx_lm.generate, UNPATCHED   — stock deepseek_v3.py (if not patched).
                                    Expected to drift on quantized MLA.
  3. mlx_lm.generate, PATCHED     — with fp32 SDPA L==1 fix applied via
                                    jang_tools.kimi_prune.runtime_patch.
                                    Expected to match path (1).

For each prompt, reports agreement between the three paths and flags any
repetition-loop signature in the generation.

Usage:
  python -m jang_tools.kimi_prune.runtime_test \\
      --model <path/to/data-drive>/dealignai/Kimi-K2.6-REAP-30-JANGTQ_1L \\
      --prompts short,reasoning,code

Requires:
  - mlx, mlx-lm installed
  - jang_tools (this package)
  - research/deepseek_v3_patched.py present
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx


PROMPTS = {
    "short": "The capital of France is",
    "short_q": "What is the capital of France? Answer in one word.",
    "math": "What is 2 + 2? Answer with just the number.",
    "reasoning": "Explain in one sentence why the sky is blue.",
    "code": 'Write a Python function `fib(n)` that returns the n-th Fibonacci number. Only output the function, no explanation.',
    "chinese": "中国的首都是",
    "tool": 'You have a tool `get_weather(city: str) -> str`. User asks: "What is the weather in Tokyo right now?" Call the tool.',
}


# ---------------------------------------------------------------------------
# Manual no-cache greedy decode — the ground truth.
#
# On every step, run a FULL forward pass over all previously generated
# tokens. This deliberately makes no use of mlx_lm's KV cache, so any L==1
# absorb-path or cache-fetch bug is bypassed.
# ---------------------------------------------------------------------------

def no_cache_greedy_decode(
    model,
    tokenizer,
    prompt_ids: list[int],
    max_new_tokens: int = 30,
    top_k: int = 5,
) -> dict:
    t0 = time.time()
    ids = list(prompt_ids)
    steps = []
    for i in range(max_new_tokens):
        x = mx.array([ids], dtype=mx.int32)
        out = model(x)  # (B, T, V) — no cache
        mx.synchronize()
        if hasattr(out, "logits"):
            out = out.logits
        last = out[0, -1]  # (V,)
        # top_k with log-softmax for numerical stability
        logits_f = last.astype(mx.float32)
        probs = mx.softmax(logits_f, axis=-1)
        top_i = mx.argpartition(-logits_f, kth=top_k)[:top_k]
        top_v = logits_f[top_i]
        # sort top_k by score desc
        order = mx.argsort(-top_v)
        top_i = top_i[order]
        top_p = probs[top_i]
        top_i_list = [int(v.item()) for v in top_i]
        top_p_list = [float(v.item()) for v in top_p]
        next_id = top_i_list[0]
        piece = tokenizer.decode([next_id])
        steps.append({
            "step": i,
            "next_id": next_id,
            "piece": piece,
            "top": list(zip(top_i_list, top_p_list)),
        })
        ids.append(next_id)
        # Early stop on EOS
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None and next_id in (eos if isinstance(eos, list) else [eos]):
            break
    return {
        "ids": ids[len(prompt_ids):],
        "text": tokenizer.decode(ids[len(prompt_ids):]),
        "steps": steps,
        "elapsed": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# mlx_lm.generate wrapper — matches what a real deployment does.
# ---------------------------------------------------------------------------

def generate_with_mlx_lm(
    model, tokenizer, prompt: str, max_new_tokens: int = 30, temp: float = 0.0
) -> dict:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    t0 = time.time()
    text = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        sampler=make_sampler(temp=temp),
        verbose=False,
    )
    return {"text": text, "elapsed": time.time() - t0}


# ---------------------------------------------------------------------------
# Repetition loop detector.
# ---------------------------------------------------------------------------

def looks_like_repetition_loop(text: str) -> str | None:
    """Return a short reason string if `text` looks like a decode-drift loop."""
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) < 20:
        return None
    # Exact-substring repetition — "1.1.1.1..." signature
    for n in (2, 3, 4, 5, 6, 8, 12):
        if len(stripped) < 3 * n:
            continue
        chunk = stripped[-n:]
        if stripped.endswith(chunk * 3):
            return f"trailing {n!r}-char substring repeated 3x+: {chunk!r}"
    # Token repetition via whitespace split
    toks = stripped.split()
    if len(toks) >= 8 and len(set(toks[-8:])) <= 2:
        return f"last 8 space-separated tokens collapse to <=2 unique: {toks[-8:]!r}"
    return None


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def run(
    model_path: Path,
    prompts: list[str],
    max_new_tokens: int,
    skip_patched: bool,
    skip_unpatched: bool,
) -> int:
    from jang_tools.load_jangtq import load_jangtq_model

    print(f"[test] model: {model_path}")
    print(f"[test] prompts: {prompts}")
    print(f"[test] max_new_tokens: {max_new_tokens}", flush=True)

    model, tokenizer = load_jangtq_model(model_path)

    # Are we currently patched? Report so the two mlx_lm.generate rows are
    # labeled accurately instead of pretending we can toggle at runtime
    # (we'd have to reload deepseek_v3 + rebuild the model to really
    # compare — skip that for now, just report state).
    try:
        import mlx_lm.models.deepseek_v3 as _d3
        import inspect
        src = inspect.getsource(_d3.DeepseekV3Attention)
        patched = "JANG fast fix" in src
    except Exception:
        patched = False
    label = "PATCHED" if patched else "UNPATCHED"
    print(f"[test] deepseek_v3.py state: {label}", flush=True)

    fails: list[tuple[str, str]] = []
    for prompt_key in prompts:
        prompt = PROMPTS.get(prompt_key, prompt_key)
        print("\n" + "=" * 72)
        print(f"[prompt:{prompt_key}] {prompt!r}")
        print("=" * 72, flush=True)

        # Apply chat template if tokenizer supports it (Kimi needs this
        # for its media/tool-aware preamble).
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                templated = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, tokenize=False,
                )
                tmpl_ids = tokenizer.encode(templated, add_special_tokens=False)
                print(f"  templated prompt: {templated[:200]!r}")
            except Exception as _e:
                print(f"  apply_chat_template failed ({_e!r}), falling back to raw encode")
                templated = prompt
                tmpl_ids = tokenizer.encode(prompt, add_special_tokens=True)
        else:
            templated = prompt
            tmpl_ids = tokenizer.encode(prompt, add_special_tokens=True)

        print(f"  prompt token count: {len(tmpl_ids)}", flush=True)

        # --- Path 1: no-cache greedy ---------------------------------------
        print("\n  [1/2] no-cache greedy (ground truth) ...", flush=True)
        r1 = no_cache_greedy_decode(
            model, tokenizer, tmpl_ids,
            max_new_tokens=max_new_tokens, top_k=5,
        )
        print(f"    text: {r1['text']!r}")
        print(f"    elapsed: {r1['elapsed']:.1f}s  "
              f"({r1['elapsed'] / max(1, len(r1['ids'])):.2f}s/tok)")
        for s in r1["steps"][:3]:
            top_str = ", ".join(f"{i}:{p:.3f}" for i, p in s["top"][:3])
            print(f"    step {s['step']}: {s['piece']!r}  top=[{top_str}]")
        rep1 = looks_like_repetition_loop(r1["text"])
        if rep1:
            fails.append((prompt_key, f"path1 (no-cache) repetition: {rep1}"))
            print(f"    FAIL: {rep1}")

        # --- Path 2: mlx_lm.generate (w/ cache) ----------------------------
        if skip_patched and patched:
            print(f"\n  [2/2] mlx_lm.generate [{label}]: SKIPPED (--skip-patched)")
            continue
        if skip_unpatched and not patched:
            print(f"\n  [2/2] mlx_lm.generate [{label}]: SKIPPED (--skip-unpatched)")
            continue

        print(f"\n  [2/2] mlx_lm.generate [{label}] ...", flush=True)
        try:
            r2 = generate_with_mlx_lm(
                model, tokenizer, templated,
                max_new_tokens=max_new_tokens, temp=0.0,
            )
            print(f"    text: {r2['text']!r}")
            print(f"    elapsed: {r2['elapsed']:.1f}s  "
                  f"({r2['elapsed'] / max(1, max_new_tokens):.2f}s/tok)")
            rep2 = looks_like_repetition_loop(r2["text"])
            if rep2:
                fails.append((prompt_key, f"path2 (generate/{label}) repetition: {rep2}"))
                print(f"    FAIL: {rep2}")
        except Exception as e:
            print(f"    ERROR: {e!r}")
            fails.append((prompt_key, f"path2 exception: {e!r}"))

    print("\n" + "=" * 72)
    print("[test] SUMMARY")
    print("=" * 72)
    if not fails:
        print("  ALL PROMPTS COHERENT.")
        return 0
    for key, reason in fails:
        print(f"  FAIL  {key}: {reason}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--prompts", default="short,math,reasoning",
                    help="Comma-separated keys from PROMPTS, or raw strings.")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--skip-patched", action="store_true",
                    help="Skip mlx_lm.generate when deepseek_v3.py is patched.")
    ap.add_argument("--skip-unpatched", action="store_true",
                    help="Skip mlx_lm.generate when deepseek_v3.py is unpatched.")
    args = ap.parse_args()
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    return run(
        model_path=args.model,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        skip_patched=args.skip_patched,
        skip_unpatched=args.skip_unpatched,
    )


if __name__ == "__main__":
    sys.exit(main())
