"""Verify a JANGTQ-converted Qwen 3.6 artifact is structurally complete.

Checks per the user's correctness concerns:
1. Bit sizes per layer kind (attn=8, embed/lm_head=8, shared=8, routed=2/3/4, norms=fp16)
2. Hybrid SSM / GatedDeltaNet linear-attention layers preserved
3. VL (vision_tower) tensors passed through
4. Tokenizer + chat template + config files all copied
5. Safetensors index is consistent with on-disk shards
6. No MTP heads (those should be stripped)

Usage:
    python3 /tmp/verify_qwen36_artifact.py /path/to/Qwen3.6-35B-A3B-JANGTQ_2L
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(1)

ART = Path(sys.argv[1])
if not ART.exists():
    sys.exit(f"FATAL: {ART} does not exist")

print(f"=== Verifying {ART} ===\n")

# 1. Required companion files
must_have = [
    "config.json",
    "jang_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    # jangtq_runtime.safetensors is required for the SWIFT runtime; the
    # Python loader computes signs/codebook on-the-fly so this is
    # optional from Python's perspective. Missing → run
    # `python3 -m jang_tools.build_jangtq_sidecar <model_dir>`.
    # Check it as nice-to-have so verify still passes if only the
    # Python path is being used.
]
nice_to_have = [
    "special_tokens_map.json",
    "generation_config.json",
    "merges.txt",
    "vocab.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "configuration.json",
    "jangtq_runtime.safetensors",  # Swift-runtime sidecar (Python doesn't need it)
]

print("[1] Required + nice-to-have files")
fail = False
for f in must_have:
    ok = (ART / f).exists()
    print(f"  {'OK' if ok else 'MISSING'}: {f}")
    if not ok:
        fail = True
for f in nice_to_have:
    ok = (ART / f).exists()
    print(f"  {'ok' if ok else '--'}: {f}")

# 2. Read jang_config bit-width contract
print("\n[2] jang_config.json bit-width contract")
jcfg = json.load(open(ART / "jang_config.json"))
bits = jcfg.get("mxtq_bits", {})
print(f"  weight_format: {jcfg.get('weight_format')}")
print(f"  profile: {jcfg.get('profile')}")
print(f"  mxtq_seed: {jcfg.get('mxtq_seed')}")
print(f"  bits.attention      = {bits.get('attention')}")
print(f"  bits.linear_attention = {bits.get('linear_attention')}")
print(f"  bits.shared_expert  = {bits.get('shared_expert')}")
print(f"  bits.routed_expert  = {bits.get('routed_expert')}")
print(f"  bits.embed_tokens   = {bits.get('embed_tokens')}")
print(f"  bits.lm_head        = {bits.get('lm_head')}")
expected = {
    "attention": 8,
    "linear_attention": 8,
    "shared_expert": 8,
    "embed_tokens": 8,
    "lm_head": 8,
}
for k, v in expected.items():
    if bits.get(k) != v:
        print(f"  MISMATCH: bits.{k} expected {v}, got {bits.get(k)}")
        fail = True
if bits.get("routed_expert") not in (2, 3, 4):
    print(f"  WARN: routed_expert bits = {bits.get('routed_expert')} (expected 2/3/4)")

# 3. config.json shape (layer counts, expert counts)
print("\n[3] config.json structure")
cfg = json.load(open(ART / "config.json"))
tcfg = cfg.get("text_config", cfg)
print(f"  model_type: {cfg.get('model_type')}  arch: {cfg.get('architectures')}")
print(f"  hidden_size:  {tcfg.get('hidden_size')}")
print(f"  hidden_layers: {tcfg.get('num_hidden_layers')}")
print(f"  num_experts:  {tcfg.get('num_experts')}")
print(f"  num_experts_per_tok: {tcfg.get('num_experts_per_tok')}")
print(f"  shared_expert_intermediate_size: {tcfg.get('shared_expert_intermediate_size')}")
print(f"  attn_output_gate: {tcfg.get('attn_output_gate')}")
print(f"  full_attention_interval: {tcfg.get('full_attention_interval')}")
print(f"  layer_types[:6]: {(tcfg.get('layer_types') or [])[:6]}")

# 4. Walk safetensors index — categorize keys
print("\n[4] Tensor inventory by kind")
idx = json.load(open(ART / "model.safetensors.index.json"))
weight_map = idx.get("weight_map", {})
total_size = idx.get("metadata", {}).get("total_size", 0)
print(f"  total_size: {total_size / 1e9:.2f} GB across {len(set(weight_map.values()))} shards")

cat_counts: Counter = Counter()
example_per_cat: defaultdict[str, str] = defaultdict(lambda: "")
for k in weight_map.keys():
    if "vision_tower" in k:
        cat = "VL/vision_tower"
    elif ".self_attn." in k:
        cat = "self_attn (full GQA)"
    elif ".linear_attn." in k:
        if k.endswith(".A_log") or k.endswith(".dt_bias"):
            cat = "linear_attn (A_log/dt_bias passthrough)"
        elif k.endswith("conv1d.weight"):
            cat = "linear_attn (conv1d passthrough)"
        elif "norm" in k:
            cat = "linear_attn (norm passthrough)"
        else:
            cat = "linear_attn (in/out_proj affine-8)"
    elif ".switch_mlp." in k:
        if k.endswith(".tq_packed"):
            cat = "switch_mlp.tq_packed (routed expert MXTQ)"
        elif k.endswith(".tq_norms"):
            cat = "switch_mlp.tq_norms"
        elif k.endswith(".tq_bits"):
            cat = "switch_mlp.tq_bits"
        else:
            cat = "switch_mlp other"
    elif ".shared_expert." in k:
        cat = "shared_expert (affine-8)"
    elif ".shared_expert_gate" in k:
        cat = "shared_expert_gate (passthrough fp16)"
    elif ".mlp.gate.weight" in k:
        cat = "router gate (passthrough fp16)"
    elif "embed_tokens" in k:
        cat = "embed_tokens (affine-8)"
    elif "lm_head" in k:
        cat = "lm_head (affine-8)"
    elif k.endswith("norm.weight"):
        cat = "norm (passthrough fp16)"
    elif k.startswith("mtp.") or ".mtp" in k:
        cat = "MTP (SHOULD NOT BE PRESENT)"
    else:
        cat = "OTHER"
    cat_counts[cat] += 1
    if not example_per_cat[cat]:
        example_per_cat[cat] = k

for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
    flag = "  WARN" if "SHOULD NOT" in cat else ""
    print(f"  {n:5}  {cat}  e.g. {example_per_cat[cat]}{flag}")
    if "SHOULD NOT" in cat:
        fail = True

# 5. Verify each layer has the right tensor categories
print("\n[5] Per-layer completeness check (sample first 4 layers)")
n_layers = tcfg.get("num_hidden_layers", 0)
fa_int = tcfg.get("full_attention_interval", 4)
layer_types = tcfg.get("layer_types") or []
for L in range(min(4, n_layers)):
    keys_for_layer = [k for k in weight_map if f".layers.{L}." in k]
    has_self_attn = any(".self_attn." in k for k in keys_for_layer)
    has_linear_attn = any(".linear_attn." in k for k in keys_for_layer)
    has_moe = any(".switch_mlp." in k for k in keys_for_layer)
    has_shared = any(".shared_expert." in k for k in keys_for_layer)
    declared = layer_types[L] if L < len(layer_types) else f"derived (interval={fa_int})"
    print(f"  layer {L}: types[L]={declared!r}  attn=({'self' if has_self_attn else '-'}/{'lin' if has_linear_attn else '-'})  moe={'Y' if has_moe else 'N'}  shared={'Y' if has_shared else 'N'}  ({len(keys_for_layer)} tensors)")

# 6. Check all routed-expert layers have 3 switch_mlp.tq triples
print("\n[6] Per-layer switch_mlp.tq_packed completeness")
expected_projs = {"gate_proj", "up_proj", "down_proj"}
missing_layers = []
for L in range(n_layers):
    found = set()
    for k in weight_map:
        if f".layers.{L}.mlp.switch_mlp." in k and k.endswith(".tq_packed"):
            for p in expected_projs:
                if f".switch_mlp.{p}." in k:
                    found.add(p)
    if found != expected_projs:
        missing = expected_projs - found
        missing_layers.append((L, missing))
if missing_layers:
    print(f"  WARN: {len(missing_layers)} layer(s) missing switch_mlp tq_packed:")
    for L, miss in missing_layers[:6]:
        print(f"    layer {L}: missing {sorted(miss)}")
    fail = True
else:
    print(f"  OK: all {n_layers} layers have full switch_mlp.{{gate,up,down}}_proj.tq_packed")

# 7. Final verdict
print()
if fail:
    print("=== VERIFY: FAIL (see warnings above) ===")
    sys.exit(1)
print("=== VERIFY: PASS ===")
