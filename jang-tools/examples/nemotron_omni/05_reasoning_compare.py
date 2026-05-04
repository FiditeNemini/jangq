"""05: Reasoning ON vs OFF comparison.

Same prompt, both modes — shows tok/s + answer length difference.
Reasoning ON wraps thinking in `<think>...</think>` then gives the
answer. Reasoning OFF skips the think block — faster, more direct.

Run: python3 05_reasoning_compare.py [bundle_path] ["custom prompt"]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def detect_loader(bundle: Path) -> str:
    jc_path = bundle / "jang_config.json"
    if jc_path.exists():
        return json.loads(jc_path.read_text()).get("weight_format", "mlx")
    return "mlx"


def load_bundle(bundle: Path):
    if detect_loader(bundle) == "mxtq":
        from jang_tools.load_jangtq import load_jangtq_model
        return load_jangtq_model(str(bundle))
    from mlx_lm import load
    return load(str(bundle))


def time_chat(model, tokenizer, user: str, *, enable_thinking: bool,
              max_tokens: int, temp: float) -> tuple[str, float, int]:
    from mlx_lm import generate
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    # Warmup
    generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    t0 = time.time()
    out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                   verbose=False)
    dt = time.time() - t0
    n_tok = len(tokenizer.encode(out)) - len(tokenizer.encode(prompt))
    return out, dt, n_tok


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path.home() / ".mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4"
    prompt = sys.argv[2] if len(sys.argv) > 2 else \
        "Solve and explain: if a train leaves Boston at 9am traveling 60mph " \
        "and another leaves NYC at 10am traveling 80mph, when do they meet? " \
        "Boston-NYC distance is 215 miles."

    print(f"\n=== {bundle.name} ===\n")
    model, tokenizer = load_bundle(bundle)

    print("=== Reasoning OFF ===")
    out, dt, n = time_chat(model, tokenizer, prompt,
                           enable_thinking=False, max_tokens=200, temp=0.6)
    print(f"  {dt:.2f}s, ~{n} tokens, ~{n/dt:.1f} tok/s")
    print(f"  {out!r}\n")

    print("=== Reasoning ON ===")
    out, dt, n = time_chat(model, tokenizer, prompt,
                           enable_thinking=True, max_tokens=400, temp=0.6)
    print(f"  {dt:.2f}s, ~{n} tokens, ~{n/dt:.1f} tok/s")
    print(f"  {out!r}\n")


if __name__ == "__main__":
    main()
