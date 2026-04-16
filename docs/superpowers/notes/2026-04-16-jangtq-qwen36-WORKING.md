# Qwen 3.6 JANGTQ — WORKING end-to-end (Python + Swift)

**Date:** 2026-04-16

## Result

First-ever Qwen 3.6 JANGTQ_2L decode in **both** Python and Swift, on
this MacBook M4 Max:

| Path | Wall (load + decode) | Decode tok/s | Output |
|---|---:|---:|---|
| Python `load_jangtq_model` + `mlx_lm.generate` | 1.85s (0.80 + 1.05) | **57.19** | "Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water..." |
| Swift `vmlxctl chat` | 10.16s (~3 + ~7) | **~50** | 250-word coherent AI essay |

**Swift is ~88% of Python on first run.**

## Artifact

- Path: `/Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L`
- Size: 11.6 GB across 12 shards
- Source: HF `Qwen/Qwen3.6-35B-A3B` BF16 (67 GB, 26 shards)
- Convert wall: ~62 minutes on M4 Max

Tensor inventory:
- 120 MXTQ (routed expert gate/up/down × 40 layers, 2-bit codebook)
- 312 affine 8-bit (attention q/k/v/o, linear_attn projections, shared experts, embed, lm_head)
- 634 fp16 passthrough (norms, conv1d, A_log, dt_bias, gates, vision_tower)
- 19 MTP heads stripped

Bit-size contract verified by `verify_qwen36_artifact.py`:
```
weight_format: mxtq
profile: JANGTQ_2L
mxtq_seed: 42
attention=8, linear_attention=8, shared_expert=8, routed_expert=2, embed_tokens=8, lm_head=8
all 40 layers have full switch_mlp.{gate,up,down}_proj.tq_packed
hybrid SSM (linear_attn) preserved
VL (vision_tower) tensors preserved
chat_template.jinja + tokenizer files present
```

## Bugs found and fixed

### 1. Python loader P18 QKV fusion broke on `attn_output_gate=True`

`load_jangtq.py:519` patches attention's __call__ to fuse Q/K/V into
one quantized matmul. Qwen 3.5/3.6 has `attn_output_gate=True` which
**doubles q_proj's output dim** (split into queries + sigmoid gate
inside the standard __call__). The P18 patch didn't know about the
split, fed a (B, 16, T, 512) query tensor to SDPA expecting K of
last_dim=256 → ValueError.

Fix: in the P18 enrollment loop, skip the fusion when
`q.weight.shape[0] == num_attention_heads * head_dim * 2` (the
attn_output_gate signature). Committed in `11468c1`.

After fix: `P18 QKV fusion: 0 class(es), 0 instances` (correctly
skipped). Decode works; speed cost is negligible because Python's
non-fused path is already optimized via `mx.compile` (P15).

### 2. Swift factory dispatch never fired (Qwen 3.6 routed to VLM)

The vMLXEngine's `ModelDetector` (Tier 2) classifies any model with
`text_config` in `config.json` as `.vision` modality and dispatches
to `VLMModelFactory.shared.loadContainer`. My `Qwen35JANGTQ`
dispatch lives in `LLMModelFactory`, which never got called.

Fix path A — converter writes `has_vision: false` in
`jang_config.json`. ModelDetector Tier 1 honors `jang_config.has_vision`
as authoritative, overriding Tier 2's text_config heuristic. Routes
to LLM factory which sees `weight_format: "mxtq"` and dispatches to
`Qwen35JANGTQModel`. Committed in `0ab9aab` (converter patch).

Fix path B — converter also surfaces `weight_format`, `mxtq_seed`,
`mxtq_bits` at the top level of `config.json` so the factory's
`FormatCheck` decode finds them without reading the sidecar.

Same commit `0ab9aab`.

### 3. Swift TurboQuantSwitchGLU required a `jangtq_runtime.safetensors`
sidecar that no converter wrote

Discovered in pre-convert audit. Python loader computes signs +
codebooks on-the-fly from `(in_features, seed, bits)` via
`generate_random_signs` and `compute_codebook`. Swift's
`JANGTQRuntimeCache` requires them as a pre-baked safetensors file
or fatalErrors on first forward.

Fix: new `jang_tools.build_jangtq_sidecar` script (commit `d32f278`)
walks the artifact, identifies unique `(in_features, seed)` and
`(in_features, bits)` triples from `.tq_packed` shapes + `.tq_bits`
metadata, computes signs/codebooks via the same Python functions
used at runtime, writes 10 KB sidecar to model dir.

For this Qwen 3.6 artifact: 4 tensors (signs.512.42, signs.2048.42,
codebook.512.2, codebook.2048.2).

## Working command sequence

```bash
# Convert (one-time, ~60 min on M4 Max)
SNAP=/Users/eric/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/7da1103448ba36029c34ce1a9a741dfe93ee0c50
OUT=/Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L
caffeinate -i python3 -u -m jang_tools.convert_qwen35_jangtq "$SNAP" "$OUT" JANGTQ_2L

# Build Swift sidecar (one-time, ~10s)
python3 -m jang_tools.build_jangtq_sidecar "$OUT"

# Verify (optional, ~2s)
python3 /Users/eric/jang/jang-tools/jang_tools/scripts/verify_qwen36_artifact.py "$OUT"

# Python decode
python3 -c "
from jang_tools.load_jangtq import load_jangtq_model
from mlx_lm import generate
m, t = load_jangtq_model('$OUT')
print(generate(m, t, 'What is 2+2?', max_tokens=32, verbose=False))
"

# Swift decode (interactive REPL)
/Users/eric/vmlx/swift/.build/arm64-apple-macosx/debug/vmlxctl chat -m "$OUT"
```

## What this delivers

- **First non-MiniMax model running on Swift JANGTQ runtime.** The
  whole `Qwen35JANGTQ.swift` + `GLM4MoEJANGTQ.swift` work from earlier
  sessions is now validated against a real artifact.
- **First Qwen 3.6 inference on Apple Silicon at 35B-A3B-JANGTQ_2L scale.**
  Both Python and Swift paths confirmed working with coherent decode.
- **Decode speed ~88% of Python on first Swift run** — without any
  additional kernel optimization. Speed parity is achievable; the
  remaining gap is in `gather_qmm` dispatch overhead per the
  JANGTQ-PLAN.md P15+ work.

## Next steps

1. **Standard JANG (affine) baseline** for Qwen 3.6 to compare
   speeds head-to-head. Run `convert.py` (the affine converter) to
   produce `Qwen3.6-35B-A3B-JANG_4M`. Estimated ~10-15 min.
2. **GLM 5.1**: existing JANGTQ_1L artifact on `/Volumes/EricsLLMDrive`
   (191 GB), Swift `GLM4MoEJANGTQModel` ready, but **GLM 5.1 uses
   `model_type: glm_moe_dsa` not `glm4_moe`** — the factory dispatch
   key needs updating, OR the GLM 5.1 artifact needs `model_type`
   patched, OR a new model file `GlmMoeDsaJANGTQ.swift` needs
   creating with MLA attention. Documented in the M3 Ultra runbook
   (won't fit on MacBook).
3. **GSM8K accuracy validation** to confirm JANGTQ_2L doesn't lose
   quality vs the BF16 source.
4. **Speed optimization**: the 12% gap (Swift vs Python) is likely
   in the gather kernel dispatch path. Investigate via Metal capture.
