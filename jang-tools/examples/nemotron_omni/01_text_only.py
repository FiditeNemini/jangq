"""01: Text-only chat across all 3 quant levels (MXFP4 / JANGTQ4 / JANGTQ2).

Demonstrates the fast text path:
  • MXFP4   → mlx_lm.load (stock)
  • JANGTQ4 → jang_tools.load_jangtq.load_jangtq_model
  • JANGTQ2 → jang_tools.load_jangtq.load_jangtq_model

Both reasoning ON (default) and OFF are shown.

Run: python3 01_text_only.py [bundle_path]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def detect_loader(bundle: Path) -> str:
    """Decide which loader to use from jang_config.json."""
    jc_path = bundle / "jang_config.json"
    if jc_path.exists():
        jc = json.loads(jc_path.read_text())
        return jc.get("weight_format", "mlx")
    return "mlx"


def load_bundle(bundle: Path):
    wf = detect_loader(bundle)
    if wf == "mxtq":
        from jang_tools.load_jangtq import load_jangtq_model
        return load_jangtq_model(str(bundle))
    from mlx_lm import load
    return load(str(bundle))


def chat(model, tokenizer, user: str, *, enable_thinking: bool = True,
         max_tokens: int = 80, temperature: float = 0.6) -> str:
    """Single-turn chat. Reasoning toggle via `enable_thinking`."""
    from mlx_lm import generate
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                    verbose=False)


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path.home() / ".mlxstudio/models/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-MXFP4"

    print(f"\n=== Loading {bundle.name} ===", flush=True)
    t0 = time.time()
    model, tokenizer = load_bundle(bundle)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # Reasoning ON (default for the Reasoning SKU)
    print("\n=== Reasoning ON ===")
    t0 = time.time()
    out = chat(model, tokenizer, "What is 17 + 28? Just the number.",
               enable_thinking=True, max_tokens=80, temperature=0.0)
    print(f"  ({time.time()-t0:.1f}s) {out}", flush=True)

    # Reasoning OFF — faster, less verbose
    print("\n=== Reasoning OFF ===")
    t0 = time.time()
    out = chat(model, tokenizer, "What is 17 + 28? Just the number.",
               enable_thinking=False, max_tokens=20, temperature=0.0)
    print(f"  ({time.time()-t0:.1f}s) {out}", flush=True)


if __name__ == "__main__":
    main()
