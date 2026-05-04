"""00 — Verify a Mistral-Medium-3.5 (mistral3 + ministral3 + pixtral) bundle.

Mistral 3.5 is a wrapper class around `ministral3` text decoder + `pixtral`
vision tower + projector. JANGTQ bundles MXTQ-quant the text decoder and
keep vision_tower / multi_modal_projector / lm_head as fp16 passthrough.

This script verifies the text-only decode path works (vision tower stripped
at load time pending VL fold-in wiring).

Run: python3 00_verify.py [bundle_path]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/Mistral-Medium-3.5-128B-JANGTQ"


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    print(f"=== Mistral 3.5 verify: {bundle.name} ===", flush=True)
    cfg = json.loads((bundle / "config.json").read_text())
    print(f"  model_type:           {cfg.get('model_type')}")
    print(f"  weight_format:        {cfg.get('weight_format')}")
    tcfg = cfg.get("text_config", {})
    vcfg = cfg.get("vision_config", {})
    print(f"  text_config:          layers={tcfg.get('num_hidden_layers')}, "
          f"hidden={tcfg.get('hidden_size')}")
    print(f"  vision_config:        layers={vcfg.get('num_hidden_layers')}, "
          f"hidden={vcfg.get('hidden_size')}")
    print(f"  modules_to_not_convert: {cfg.get('modules_to_not_convert', [])[:5]}")
    print(f"  jangtq sidecar:       {(bundle / 'jangtq_runtime.safetensors').exists()}")

    import mlx.core as mx
    mx.set_memory_limit(200 * 1024**3)

    from jang_tools.mistral3.runtime import load
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
    print("  tokenizer chat_template:", "set" if getattr(tok, "chat_template", None) else "MISSING")

    t0 = time.time()
    model, mcfg, fmt = load(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s (format={fmt})", flush=True)

    prompt = "The capital of France is"
    ids = tok.encode(prompt)

    out = list(ids)
    x = mx.array([ids], dtype=mx.uint32)
    logits, caches = model(x, caches=None)
    for _ in range(8):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        x = mx.array([[nxt]], dtype=mx.uint32)
        logits, caches = model(x, caches=caches)
    text = tok.decode(out)
    print(f"\n=== Output ===\n  {text!r}")
    if "Paris" in text or "paris" in text.lower():
        print("\n=== PASS — coherent answer ===")
    else:
        print("\n=== WARN — answer doesn't contain 'Paris' ===")


if __name__ == "__main__":
    main()
