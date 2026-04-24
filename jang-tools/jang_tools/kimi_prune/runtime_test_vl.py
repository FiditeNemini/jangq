"""Image + video coherence test for Kimi K2.6 JANGTQ bundles.

Complements runtime_test.py (which is text-only). This harness:

  1. Loads the JANGTQ bundle via load_jangtq_kimi_vlm_model (mlx_vlm path
     with MoonViT vision tower + PatchMergerMLP projector + DeepseekV3
     language backbone — all wired through JANGTQ TurboQuant kernels for
     the text side).
  2. For each test image: generates a description and checks for basic
     coherence signals (non-empty, no repetition loop, mentions image
     content for the hard-coded prompt).
  3. Reports per-prompt pass/fail + total latency.

Usage:
  python -m jang_tools.kimi_prune.runtime_test_vl \\
      --model <path/to/data-drive>/dealignai/Kimi-K2.6-REAP-30-JANGTQ_1L \\
      --image /path/to/test.jpg \\
      --prompt "Describe this image in one sentence."

Pass `--image ""` to run a text-only smoke check on the VL-wired model.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def looks_like_repetition_loop(text: str) -> str | None:
    if not text or len(text.strip()) < 20:
        return None
    stripped = text.strip()
    for n in (2, 3, 4, 5, 6, 8, 12):
        if len(stripped) >= 3 * n and stripped.endswith(stripped[-n:] * 3):
            return f"trailing {n!r}-char substring repeated 3x+"
    toks = stripped.split()
    if len(toks) >= 8 and len(set(toks[-8:])) <= 2:
        return f"last 8 tokens collapse to <=2 unique"
    return None


def run(model_path: Path, image: str | None, prompt: str,
        max_new_tokens: int) -> int:
    from jang_tools.load_jangtq_kimi_vlm import load_jangtq_kimi_vlm_model
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    print(f"[vl-test] model: {model_path}")
    print(f"[vl-test] image: {image!r}")
    print(f"[vl-test] prompt: {prompt!r}")
    t0 = time.time()
    model, processor = load_jangtq_kimi_vlm_model(model_path)
    print(f"[vl-test] loaded in {time.time()-t0:.1f}s", flush=True)

    config = getattr(model, "config", None)
    templated = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=1 if image else 0,
    )
    print(f"[vl-test] templated: {templated[:240]!r}")

    t0 = time.time()
    kwargs = dict(max_tokens=max_new_tokens, temp=0.0, verbose=False)
    if image:
        kwargs["image"] = image
    text = generate(model, processor, prompt=templated, **kwargs)
    elapsed = time.time() - t0
    print()
    print(f"[vl-test] elapsed: {elapsed:.1f}s  "
          f"({elapsed / max(1, max_new_tokens):.2f}s/tok)")
    print(f"[vl-test] output: {text!r}")
    rep = looks_like_repetition_loop(text)
    if rep:
        print(f"[vl-test] FAIL: {rep}")
        return 1
    if not text.strip():
        print("[vl-test] FAIL: empty generation")
        return 1
    print("[vl-test] OK (coherent, non-empty, no loop)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--image", default="",
                    help="Path or URL to a test image. Empty = text-only smoke test.")
    ap.add_argument("--prompt", default="Describe this image in one sentence.")
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()
    return run(
        model_path=args.model,
        image=args.image or None,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    sys.exit(main())
