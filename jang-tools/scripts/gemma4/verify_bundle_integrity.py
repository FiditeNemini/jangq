"""Verify a gemma4 bundle faithfully carries multimodal/tokenizer/audio data.

Checks the bundle-side things that could cause the runtime VL/audio bugs:
  - all 11 multimodal embedder tensors present, correct shape, no inf/nan
  - fp16 passthrough did NOT overflow vs the bf16 source (range safety)
  - tokenizer / processor_config / generation_config / chat_template faithful
"""
import json, sys, hashlib
from pathlib import Path
import mlx.core as mx

SRC = Path(sys.argv[1]).expanduser()
BUN = Path(sys.argv[2]).expanduser()
FP16_MAX = 65504.0

print("loading source (mmap)...")
SRCW = mx.load(str(SRC / "model.safetensors"))
idx = json.loads((BUN / "model.safetensors.index.json").read_text())
bun = {}
for sh in sorted(set(idx["weight_map"].values())):
    bun.update(mx.load(str(BUN / sh)))

def src_key(k):
    if k.startswith("language_model.model."):
        return "model.language_model." + k[len("language_model.model."):]
    if k.startswith(("embed_vision.", "embed_audio.", "vision_embedder.")):
        return "model." + k
    return k

MM = ("vision_embedder", "embed_vision", "embed_audio")
mm_keys = sorted(k for k in bun if any(m in k for m in MM))
print(f"\n=== MULTIMODAL tensors in bundle: {len(mm_keys)} (expect 11) ===")
print(f"{'key':52s} {'shape':18s} {'dtype':8s} {'src|max|':>10s} {'bun|max|':>10s} inf/nan  relerr")
ok = True
for k in mm_keys:
    bv = bun[k]
    sk = src_key(k)
    sv = SRCW.get(sk)
    src_max = float(mx.max(mx.abs(sv.astype(mx.float32)))) if sv is not None else float("nan")
    bf = bv.astype(mx.float32)
    n_inf = int(mx.sum(mx.isinf(bf))); n_nan = int(mx.sum(mx.isnan(bf)))
    bun_max = float(mx.max(mx.abs(bf)))
    relerr = float(mx.linalg.norm(bf - sv.astype(mx.float32)) / (mx.linalg.norm(sv.astype(mx.float32)) + 1e-8)) if sv is not None else float("nan")
    flag = ""
    if sv is None: flag += " MISSING-SRC"
    if n_inf or n_nan: flag += " !!INF/NAN!!"; ok = False
    if src_max > FP16_MAX: flag += " !!FP16-OVERFLOW-RISK!!"
    if bun_max == 0: flag += " !!ALL-ZERO!!"; ok = False
    print(f"{k:52s} {str(tuple(bv.shape)):18s} {str(bv.dtype).split('.')[-1]:8s} {src_max:10.2f} {bun_max:10.2f} {n_inf+n_nan:>6d}  {relerr:.4f}{flag}")

# norm passthrough overflow scan (all fp16 non-quantized 1-D)
print("\n=== passthrough fp16 range scan (norms + scalars) ===")
worst_norm = 0.0; over = 0
for k, v in bun.items():
    if k.endswith((".scales", ".biases")): continue
    if any(m in k for m in MM): continue
    if k.endswith(".weight") and (k[:-7] + ".scales") in bun: continue  # quantized
    sv = SRCW.get(src_key(k))
    if sv is None: continue
    m = float(mx.max(mx.abs(sv.astype(mx.float32))))
    worst_norm = max(worst_norm, m)
    if m > FP16_MAX: over += 1; print(f"  OVERFLOW {k}: src|max|={m}")
print(f"  worst |max| among passthrough = {worst_norm:.2f} (fp16 max {FP16_MAX})  overflow_tensors={over}")
if over: ok = False

# sidecar faithfulness vs source
print("\n=== sidecar faithfulness vs source ===")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12] if Path(p).exists() else "MISSING"
for f in ("tokenizer.json", "processor_config.json", "generation_config.json"):
    s, b = sha(SRC / f), sha(BUN / f)
    same = "IDENTICAL" if s == b else "DIFFERENT"
    print(f"  {f:28s} src={s} bun={b}  {same}")
    if s != b and s != "MISSING": ok = False

# chat template: source jinja vs folded-in tokenizer_config.chat_template
src_tpl = (SRC / "chat_template.jinja").read_text() if (SRC / "chat_template.jinja").exists() else None
bun_tc = json.loads((BUN / "tokenizer_config.json").read_text())
bun_tpl = bun_tc.get("chat_template")
bun_jinja = (BUN / "chat_template.jinja").read_text() if (BUN / "chat_template.jinja").exists() else None
print(f"  chat_template.jinja present in bundle: {bun_jinja is not None} ; matches source: {bun_jinja == src_tpl}")
print(f"  tokenizer_config.chat_template folded == source jinja: {bun_tpl == src_tpl}")
if bun_jinja != src_tpl: ok = False

# special tokens + suppress tokens
cfg = json.loads((BUN / "config.json").read_text())
scfg = json.loads((SRC / "config.json").read_text())
print("\n=== token id integrity (bundle vs source config) ===")
for key in ("image_token_id","audio_token_id","video_token_id","boi_token_id","boa_token_id","eoi_token_id","eoa_token_index"):
    print(f"  {key:18s} bundle={cfg.get(key)} source={scfg.get(key)} {'OK' if cfg.get(key)==scfg.get(key) else 'MISMATCH'}")
    if cfg.get(key) != scfg.get(key): ok = False
gen = json.loads((BUN / "generation_config.json").read_text())
print(f"  generation suppress_tokens={gen.get('suppress_tokens')} eos={gen.get('eos_token_id')}")

print("\n" + ("VERDICT: BUNDLE CLEAN ✅" if ok else "VERDICT: ISSUES FOUND ❌"))
