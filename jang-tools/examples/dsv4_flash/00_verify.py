"""00 — Verify a DSV4-Flash bundle loads and emits coherent first tokens.

Loads the bundle, prints the metadata that matters for runtime correctness
(weight_format, mxtq_bits, chat_template presence, EOS token, jang_config
chat block), and runs a 16-token greedy generation on a fixed prompt.
Anything coherent at the end means the bundle is hydrated correctly and
the runtime is wired (HSA + CSA + SWA, pool quant cache, chat template,
EOS) — not just that load() didn't crash.

Run: python3 00_verify.py [bundle_path]
Default bundle: ~/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"[verify] bundle not found: {bundle}")
        sys.exit(2)

    print(f"=== DSV4-Flash verify: {bundle.name} ===", flush=True)
    cfg = json.loads((bundle / "config.json").read_text())
    jcfg_path = bundle / "jang_config.json"
    jcfg = json.loads(jcfg_path.read_text()) if jcfg_path.exists() else {}
    print(f"  weight_format:    {cfg.get('weight_format')}")
    print(f"  mxtq_bits:        {cfg.get('mxtq_bits')}")
    print(f"  routed_expert:    {cfg.get('routed_expert_bits')}")
    print(f"  jang profile:     {jcfg.get('profile')}")
    print(f"  jangtq sidecar:   {(bundle / 'jangtq_runtime.safetensors').exists()}")
    chat_cfg = jcfg.get("chat", {})
    print(f"  reasoning support:{chat_cfg.get('reasoning', {}).get('supported')}")
    print(f"  reasoning modes:  {chat_cfg.get('reasoning', {}).get('modes')}")
    print(f"  tool parser:      {chat_cfg.get('tool_calling', {}).get('parser')}")
    print(f"  EOS:              {chat_cfg.get('eos_token_id')} ({chat_cfg.get('eos_token')!r})")

    import os
    os.environ.setdefault("DSV4_LONG_CTX", "1")
    print(f"  DSV4_LONG_CTX:    {os.environ['DSV4_LONG_CTX']}")
    print(f"  DSV4_POOL_QUANT:  {os.environ.get('DSV4_POOL_QUANT', '<auto-on>')}")

    import mlx.core as mx
    mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)
    from jang_tools.load_jangtq import load_jangtq_model
    from jang_tools.dsv4.runtime import generate, GenerateOptions

    t0 = time.time()
    model, tok = load_jangtq_model(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    res = generate(
        model, tok, str(bundle),
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        opts=GenerateOptions(mode="chat", max_tokens=24, temperature=0.0),
    )
    if res.error:
        print(f"  ERROR: {res.error}")
        sys.exit(3)
    print("\n=== Output ===")
    print(f"  content: {res.content!r}")
    print(f"  tokens:  {res.n_tokens}")
    print(f"  finish:  {res.finish_reason}")

    if "Paris" in res.content or "paris" in res.content:
        print("\n=== PASS — coherent answer ===")
    else:
        print("\n=== WARN — answer doesn't contain 'Paris'; check runtime ===")


if __name__ == "__main__":
    main()
