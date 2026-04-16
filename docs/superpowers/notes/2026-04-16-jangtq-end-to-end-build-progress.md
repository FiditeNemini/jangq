# JANGTQ end-to-end build for Qwen 3.6 + GLM 5.1 — progress checkpoint

**Date:** 2026-04-16
**Goal:** Get JANGTQ + JANG working for Qwen 3.6 and GLM 5.1, both
Python and Swift, decode speed = baseline.

## Done in this session (committed)

### Swift JANGTQ runtime extension (commits `d8fc72a`, `c22c293`, etc.)

| File | Status |
|---|---|
| `Sources/vMLXLLM/Models/Qwen35JANGTQ.swift` | NEW, ~520 LOC, builds clean |
| `Sources/vMLXLLM/Models/GLM4MoEJANGTQ.swift` | NEW, ~370 LOC, builds clean |
| `Sources/vMLXLLM/LLMModelFactory.swift` | extended dispatch on `weight_format == "mxtq"` for `qwen3_5_moe` and `glm4_moe` |
| `vmlxctl` rebuilt | 121 MB binary at `.build/arm64-apple-macosx/debug/vmlxctl` |

The Swift Qwen35JANGTQModel reuses internal `Qwen35GatedDeltaNet`,
`Qwen35Attention`, `Qwen3NextMLP` from the affine path — no
duplication. Only the MoE block is replaced with one using
`TurboQuantSwitchGLU`. The full model hierarchy maps to the converter's
output keys (verified by inspection — `language_model.model.layers.L.mlp.switch_mlp.{gate,up,down}_proj.{tq_packed,tq_norms}`
matches the @ModuleInfo path exactly).

### Python build_jangtq_sidecar.py (commit `d32f278`)

Critical gap closed: the Swift `JANGTQRuntimeCache` requires a
`jangtq_runtime.safetensors` sidecar with signs + codebook arrays.
The Python loader computes these on-the-fly from `(in_features, seed,
bits)`, so the existing converter never wrote one. New script reads
the JANGTQ artifact, identifies every unique `(in_features, seed)`
and `(in_features, bits)` triple, generates signs + codebooks via
the same `generate_random_signs` and `compute_codebook` Python
functions used at runtime, and writes the sidecar.

```bash
python3 -m jang_tools.build_jangtq_sidecar /Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L
```

Without this sidecar: Swift's `TurboQuantSwitchLinear` `fatalError`s
on first forward.

### Test scripts (commit `750f35f`)

- `jang_tools/scripts/verify_qwen36_artifact.py` — structural verification
- `jang_tools/scripts/test_qwen36_python.py` — Python decode smoke
- `jang_tools/scripts/qwen36_pipeline.sh` — end-to-end pipeline

### Documentation (commit `d32f278`)

- `docs/superpowers/notes/2026-04-16-qwen36-jangtq-bitsize-audit.md` — full bit-size audit covering attn/embed/lm_head 8-bit, hybrid SSM (GatedDeltaNet) preservation, VL pass-through, chat template + tokenizer copying. **Addresses the user's "PROPER BIT SIZE ATTENTION EMBEDS / HYBRID SSM / GATEDDELTANET / VL LAYERS / CHAT TEMPLATE" concerns directly.**
- `docs/superpowers/runbooks/2026-04-16-glm51-jangtq-m3ultra-runbook.md` — executable runbook for GLM 5.1 on M3 Ultra (191 GB, won't fit on MacBook). Covers the `deepseek_v32` SDPA fp32 patch.

## In flight

### Convert: Qwen3.6-35B-A3B BF16 → JANGTQ_2L

Started 11:39 PDT, 4th attempt (under `caffeinate -i` via Bash
`run_in_background` for proper detachment). Three earlier runs died
silently from session-shell signal propagation under nohup.

Status as of 11:50:
- 43% (~450/1045 tensors), 10 min elapsed
- 2 shards written (~2 GB)
- ETA ~60 min more (system under memory pressure, 15 GB swap used)
- Output: `/Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L/`
- Log: `/tmp/jangtq-convert-logs/qwen36-v4.log`
- PID 1560

Why so slow: `tq_quantize_experts` runs a serial Python loop over 256
experts per stacked tensor (3 projections × 40 layers = 120 stacked
tensors). Each per-expert `tq_quantize_weight` does Hadamard rotation
+ Lloyd-Max codebook quantization at ~50ms. Total ≈ 256 × 0.05s × 80
heavy tensors ≈ 17 min of pure quantization compute. Plus disk I/O,
plus other tensors, plus memory-pressure-induced swapping.

## Hard blockers (M3 Ultra only)

- **GLM 5.1**: smallest existing JANGTQ artifact is 191 GB. Won't fit
  in 128 GB. Documented as M3 Ultra path in the GLM 5.1 runbook
  (`/Volumes/EricsLLMDrive/GLM-5.1-JANGTQ_1L/` already exists).
- **MiniMax M2.7-FP8 source**: 214 GB. Can't re-convert from MacBook.
  Existing `MiniMax-M2.7-JANG_3L/4M/6M` artifacts are on the external
  drive and tested at 44.3 tok/s on M3 Ultra (per memory).

## What this session unlocks

When the Qwen 3.6 convert finishes:

1. **Swift JANGTQ inference for Qwen 3.5/3.6 MoE family** — first
   non-MiniMax JANGTQ model that the vmlx-swift runtime can load and
   decode. Previously zero models could load via Qwen35JANGTQ; now
   one. Speed parity vs Python TBD on first runtime measurement.
2. **Swift JANGTQ inference for GLM 4 MoE / GLM 5.1** — code
   complete, awaits a JANGTQ artifact (which exists for GLM 5.1 on
   M3 Ultra; GLM 4 MoE conversion requires same converter generalization).
3. **Sidecar generator** — patches a real gap that would have blocked
   any Swift JANGTQ inference on any model going forward.

## Resume command after convert finishes

```bash
bash /Users/eric/jang/jang-tools/jang_tools/scripts/qwen36_pipeline.sh
```

This runs: structural verify → sidecar build → Python decode →
Swift decode → speed report.

## Known issues

1. **Multiprocessing semaphore leak on Python 3.14 + MLX shutdown** —
   noisy but harmless. 7 leaked semaphores total, well under the
   ~87 macOS limit.
2. **`/Users/eric/jang/models/` was deleted between checks earlier** —
   cause unknown (possibly an automated cleanup hook). Switched
   output to `/Users/eric/models/` which is verified-stable.
3. **convert process keeps dying under nohup but stable under
   `caffeinate -i` + Bash `run_in_background`** — Bash tool's
   detachment mechanism appears to be the right one for long-running
   convert jobs on this machine.

## Commits this session

- `d8fc72a` — Swift extension to GLM 5.1 + Qwen 3.6 (handoff notes)
- `d32f278` — Qwen 3.6 + GLM 5.1 audit + sidecar generator + M3 Ultra runbook
- `750f35f` — Qwen 3.6 verification + smoke test scripts
- (pending) — convert + smoke results
