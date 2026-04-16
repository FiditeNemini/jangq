"""Smoke test the JANGTQ Python loader on a freshly-converted artifact.

Usage:
    python3 /tmp/test_qwen36_python.py /path/to/Qwen3.6-35B-A3B-JANGTQ_2L
"""
import sys
import time

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

model_path = sys.argv[1]
print(f"=== JANGTQ Python loader smoke ===", flush=True)
print(f"  model: {model_path}", flush=True)

t0 = time.time()
from jang_tools.load_jangtq import load_jangtq_model

model, tokenizer = load_jangtq_model(model_path)
load_dt = time.time() - t0
print(f"\n  load wall: {load_dt:.1f}s", flush=True)

# Use mlx_lm.generate for a quick coherence test
from mlx_lm import generate

prompts = [
    "What is 2+2?",
    "Name three colors.",
]

for p in prompts:
    t0 = time.time()
    out = generate(model, tokenizer, p, max_tokens=32, verbose=False)
    dt = time.time() - t0
    n_tok = len(tokenizer.encode(out)) if isinstance(out, str) else 32
    rate = n_tok / max(dt, 1e-6)
    print(f"\n  prompt: {p!r}", flush=True)
    print(f"  resp  : {out[:200]!r}", flush=True)
    print(f"  wall  : {dt:.2f}s  ~{rate:.2f} tok/s", flush=True)

print("\n  pipeline: OK", flush=True)
