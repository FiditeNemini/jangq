"""00 — Verify a Laguna-XS.2 (or Laguna-M.1) bundle loads + emits coherent code.

Laguna is poolside's agentic-coding MoE — model_type=laguna. The Python
runtime auto-detects bundle format (bf16 / JANG affine / JANGTQ / MXFP4)
and dispatches to the right loader.

Run: python3 00_verify.py [bundle_path]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ"


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    print(f"=== Laguna verify: {bundle.name} ===", flush=True)
    cfg = json.loads((bundle / "config.json").read_text())
    jcfg_path = bundle / "jang_config.json"
    jcfg = json.loads(jcfg_path.read_text()) if jcfg_path.exists() else {}
    print(f"  model_type:           {cfg.get('model_type')}")
    print(f"  weight_format:        {cfg.get('weight_format')}")
    print(f"  num_hidden_layers:    {cfg.get('num_hidden_layers')}")
    print(f"  num_experts:          {cfg.get('num_experts')}")
    print(f"  layer_types[:6]:      {cfg.get('layer_types', [])[:6]}")
    print(f"  per-layer head count: {cfg.get('num_attention_heads_per_layer', [])[:6]}")
    print(f"  rope_parameters keys: {list(cfg.get('rope_parameters', {}).keys())}")
    print(f"  jangtq sidecar:       {(bundle / 'jangtq_runtime.safetensors').exists()}")

    import mlx.core as mx
    mx.set_memory_limit(int(sys.argv[2]) * 1024**3 if len(sys.argv) > 2 else 200 * 1024**3)

    from jang_tools.laguna.runtime import load, greedy
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
    print("  tokenizer chat_template:", "set" if getattr(tok, "chat_template", None) else "MISSING")

    t0 = time.time()
    model, mcfg, fmt = load(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s (format={fmt})", flush=True)

    prompt = "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"
    ids = tok.encode(prompt)

    t1 = time.time()
    out = greedy(model, ids, max_new=32)
    dt = time.time() - t1
    n_new = len(out) - len(ids)
    print(f"\n  generated {n_new} tokens in {dt:.2f}s ({n_new/dt:.1f} tok/s)", flush=True)
    decoded = tok.decode(out)
    print(f"\n=== Output ===\n{decoded}")
    if "return" in decoded[len(prompt):].lower() or "fibonacci" in decoded[len(prompt):].lower():
        print("\n=== PASS — coherent code ===")
    else:
        print("\n=== WARN — output doesn't look like Fibonacci code ===")


if __name__ == "__main__":
    main()
