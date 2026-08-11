"""Structural + numerical validation of a Gemma 4 MXFP bundle.

- checks the on-disk key layout (sanitized), quant-artifact dtypes
- dequantizes a sample of quantized tensors and compares to the source bf16
  weights (relative Frobenius error) to prove the MX codec/group_size/mode are
  correct and not producing garbage.
"""
import json, sys
from pathlib import Path
import mlx.core as mx
from safetensors import safe_open

SRC = Path(sys.argv[1]).expanduser()          # source bf16 dir
BUN = Path(sys.argv[2]).expanduser()          # MXFP bundle dir
bits = int(sys.argv[3]) if len(sys.argv) > 3 else 4
mode = f"mxfp{bits}"
GS = 32

idx = json.loads((BUN / "model.safetensors.index.json").read_text())
wmap = idx["weight_map"]
shards = sorted(set(wmap.values()))

# load bundle tensors lazily per shard
bun = {}
for sh in shards:
    with safe_open(str(BUN / sh), framework="mlx") as f:
        for k in f.keys():
            bun[k] = f.get_tensor(k)

keys = set(bun.keys())
quant_bases = sorted({k[:-len(".weight")] for k in keys if k.endswith(".weight") and (k[:-len(".weight")] + ".scales") in keys})
passthrough = sorted({k for k in keys if k.endswith(".weight") and (k[:-len(".weight")] + ".scales") not in keys}
                     | {k for k in keys if not k.endswith((".weight", ".scales", ".biases"))})
print(f"bundle tensors: {len(keys)}")
print(f"quantized linears (have .scales): {len(quant_bases)}")
print(f"fp16 passthrough tensors:         {len(passthrough)}")

# dtype sanity
sample_q = quant_bases[0]
print(f"  sample quant weight dtype: {bun[sample_q + '.weight'].dtype}, scales: {bun[sample_q + '.scales'].dtype}")

# k_eq_v: full-attn layers must have NO v_proj
full_layers = [5,11,17,23,29,35,41,47]
missing_v = [L for L in full_layers if f"language_model.model.layers.{L}.self_attn.v_proj" not in quant_bases]
present_v_full = [L for L in full_layers if f"language_model.model.layers.{L}.self_attn.v_proj" in quant_bases]
print(f"  full-attn layers without v_proj (expect all 8): {missing_v}")
assert not present_v_full, f"unexpected v_proj on full layers {present_v_full}"
# sliding layers must HAVE v_proj
slide_missing = [L for L in range(48) if L not in full_layers and f"language_model.model.layers.{L}.self_attn.v_proj" not in quant_bases]
print(f"  sliding layers missing v_proj (expect none): {slide_missing}")
assert not slide_missing

print("  loading source weights (mmap)...")
SRCW = mx.load(str(SRC / "model.safetensors"))

def src_tensor(sanitized_key):
    """map sanitized bundle key back to source HF name and load bf16->f32."""
    hf = sanitized_key
    if hf.startswith("language_model.model."):
        hf = "model.language_model." + hf[len("language_model.model."):]
    elif hf.startswith(("embed_vision.", "embed_audio.", "vision_embedder.")):
        hf = "model." + hf
    t = SRCW.get(hf, SRCW.get(hf + ".weight"))
    return t.astype(mx.float32)

def rel_err(a, b):
    return float(mx.linalg.norm(a - b) / (mx.linalg.norm(b) + 1e-8))

# sample across the network: embed, early/mid/late attn + mlp
samples = [
    "language_model.model.embed_tokens",
    "language_model.model.layers.0.self_attn.q_proj",
    "language_model.model.layers.0.mlp.gate_proj",
    "language_model.model.layers.5.self_attn.k_proj",   # full-attn layer
    "language_model.model.layers.23.self_attn.o_proj",  # full-attn layer
    "language_model.model.layers.24.mlp.down_proj",
    "language_model.model.layers.47.self_attn.q_proj",  # last (full)
    "language_model.model.layers.47.mlp.up_proj",
]
print("\n  dequant round-trip relative error vs source bf16:")
worst = 0.0
for base in samples:
    if base not in quant_bases:
        print(f"    {base}: MISSING"); continue
    w = bun[base + ".weight"]; s = bun[base + ".scales"]
    b = bun.get(base + ".biases")
    if b is not None:
        deq = mx.dequantize(w, s, b, group_size=GS, bits=bits, mode=mode)
    else:
        deq = mx.dequantize(w, s, group_size=GS, bits=bits, mode=mode)
    src = src_tensor(base)
    e = rel_err(deq.astype(mx.float32), src)
    worst = max(worst, e)
    print(f"    {base:55s} shape={tuple(deq.shape)} relerr={e:.4f}")

# passthrough fidelity (norms must be bit-identical-ish fp16, NO +1 applied)
print("\n  passthrough check (norm should equal source, NOT source+1):")
for base in ["language_model.model.norm.weight",
             "language_model.model.layers.0.input_layernorm.weight"]:
    bv = bun[base].astype(mx.float32)
    sv = src_tensor(base[:-len(".weight")] if base.endswith(".weight") else base) if base.endswith(".weight") else None
    # src_tensor expects no .weight strip for params; load directly
    hf = base
    if hf.startswith("language_model.model."):
        hf = "model.language_model." + hf[len("language_model.model."):]
    sv = SRCW[hf].astype(mx.float32)
    e_asis = rel_err(bv, sv)
    e_plus1 = rel_err(bv, sv + 1.0)
    print(f"    {base:55s} relerr_as_is={e_asis:.5f}  relerr_if_plus1={e_plus1:.5f}")

print(f"\n  WORST quant relerr across samples: {worst:.4f}")
print("  PASS" if worst < 0.15 else "  WARN: high error")
