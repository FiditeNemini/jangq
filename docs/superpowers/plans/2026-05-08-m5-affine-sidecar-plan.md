# M5 Native Affine Sidecar — Plan

**Author:** Jinho Jang (eric@jangq.ai)
**Date:** 2026-05-08
**Replaces:** `2026-05-08-jangtq-na-phase-a.md` (closed)
**Reference:** `research/experiments/jangtq-na/PHASE_A_FINDINGS.md`

## Why this plan exists

MLX 0.31.2's metallib already ships M5 Neural-Accelerator kernels for **affine** quantized matmul, both dense (`affine_qmm_*_nax_*`) and MoE-gather (`affine_gather_qmm_*_nax_*`). These are 64×64-tile, 2×2-simdgroup Steel-template kernels covering bits {2, 3, 4, 5, 6, 8} × group_size {32, 64, 128} × bf16/float. Any model that loads through `nn.QuantizedLinear` / `mx.quantized_matmul` / `mx.gather_qmm` on M5+macOS 26.2+ already hits NA tensor cores.

JANGTQ's codebook+Hadamard format does not load through these kernels — it goes through the project's own `TurboQuantLinear` / `gather_tq_matmul` Metal FMA kernels, which are not NA-accelerated. A custom codebook-NA kernel would need Steel-class engineering to be competitive (proved 2026-05-08, this branch).

The cheapest path to NA-accelerated inference on Eric's JANGTQ targets is therefore a **format question, not a kernel question**: re-quantize each target into a standard MLX-affine bundle and run it through MLX's existing NA kernels. The trade-off is quality (codebook is more compact at sub-2-bit) vs speed (affine NA wins on M5).

## Goal

Determine, with measured numbers, whether re-quantizing each currently-shipping JANGTQ target into MLX-affine format produces a bundle that is:

1. **Faster on M5 Max** than the JANGTQ_K equivalent (for prefill and decode), via MLX's NA-using kernels.
2. **Within acceptable quality bounds** of the JANGTQ_K equivalent (MMLU ±0.5 pp, 5-prompt coherence, no EOS-at-50 collapse).
3. **Within acceptable bundle-size bounds** vs JANGTQ_K (affine 2-bit is roughly the same size as JANGTQ 2-bit codebook indices + per-row norms; affine 4-bit is ~2× codebook 2-bit, may be problematic for 100B+ models).

Plan-level success: at least one target ships an `*-AFFINE-NAX` (working name) bundle that beats the JANGTQ_K equivalent on pp/s and tok/s on M5 Max, at quality parity, and at bundle size ≤ 1.5× JANGTQ_K.

Plan-level failure: no quantization recipe at acceptable quality reaches M5 Max NA's potential without exceeding bundle size budget. Outcome: keep JANGTQ_K format for size-constrained workloads, accept that NA speed wins live in MLX-native bundles.

## Targets in priority order

1. **MiniMax-M2.7** (228B / A15B MoE) — primary because it's the active JANGTQ_K reference. `affine_gather_qmm_*_nax_*` directly applicable. Bundle size at affine 2-bit: estimated ~50 GB, comparable to JANGTQ_K's 56.5 GB.
2. **Qwen3.6-27B dense** — fast iteration target. Already shipped in `JANG_4M` (affine 4-bit) format → already NA-accelerated. The work here is *measuring* the lift vs an unaffected baseline, not quantizing.
3. **Kimi-K2.6-Small** (61B/A6B JANGTQ) — second MoE; once MiniMax pattern is proven, Kimi reuses the converter.
4. **GLM-5.1** (357B JANGTQ) — large MoE, longer context (256K+); validates that the affine NA path doesn't break at large param counts.

Out of scope for this plan: DSV4-Flash (different attention shape, MLA + Compressor; needs its own architecture-specific validation), Mistral-Medium-3.5 (not currently a JANGTQ target, dense-only).

## Hardware policy

Affine bundles are **standard MLX-affine quant** — the same file format that any
`mlx_lm.convert`-produced bundle uses. They load and run anywhere MLX runs.
What's *M5-specific* is the speed: on M5+macOS 26.2+, MLX dispatches into the
`affine_qmm_*_nax_*` Steel-NAX kernels; on older silicon or older macOS, the
same `mx.quantized_matmul` call falls back to the non-NA affine path. We do
not need a hardware gate inside the bundle, and there is no separate
"M5-only" bundle artifact.

Documentation should advertise the bundle as "NA-accelerated on M5+macOS 26.2+;
runs as standard affine elsewhere." If a target's pp/s on non-M5 hardware
regresses badly vs JANGTQ_K (because JANGTQ's hand-tuned decode kernel was
faster on M3/M4), document that JANGTQ_K stays as the recommended bundle for
those machines — but it's a recommendation, not a hard runtime gate inside
the affine bundle.

## Plan structure

The work is genuinely a **measurement project** more than an engineering project. The MLX side already exists; what we don't know is the quality + size + speed tradeoff per target. Each phase is one target; phases are independent and can be reordered.

### Phase 0 — Baseline confirmation (Qwen3.6-27B JANG_4M)

This is the cheapest sanity check. Qwen3.6-27B already ships in JANG_4M format, which is MLX-affine 4-bit, which on M5+ is already NA-accelerated. Measure pp/s + tok/s + MMLU on M5 Max, compare to whatever the M3 Ultra reference numbers were (if any exist). This validates that the NA path actually delivers the speed advantage we expect on a real bundle-load + decode workflow before committing to converter work.

If Qwen3.6-27B-JANG_4M does NOT show meaningful speedup vs the existing reference, something is wrong with our model of how MLX-NA dispatches in practice and the rest of the plan needs reconsideration.

Deliverable: `research/experiments/m5-affine/qwen3.6-27b-jang_4m-baseline.json` — pp/s, tok/s, MMLU, peak GPU memory.

### Phase 1 — MiniMax-M2.7 affine conversion + bench

The primary target. Need to build a converter that takes MiniMax-M2.7-bf16 source weights and produces an MLX-native affine quantized bundle compatible with `vllm-mlx` (Python) and `vmlx-swift-lm` (Swift) loading paths.

Conversion options to test (fastest → most engineering):

**1a. Stock `mlx_lm.convert`**: the simplest path. `mlx-lm` ships a converter that quantizes any HF transformers model to MLX format. Try this first with `--bits 4 --group-size 64` (matches MLX's most-tuned NAX kernel). If it produces a working bundle, we're done with conversion. Bench against JANGTQ_K.

**1b. `mlx_lm.convert` at `--bits 3` or `--bits 2`**: smaller bundles, lower quality. Test which is the smallest size that holds MMLU within ±0.5 pp of JANGTQ_K. Sub-3-bit affine is known to be quality-fragile without calibration; expect to need AWQ.

**1c. AWQ-calibrated affine quantization**: if stock convert at `--bits 2` or `--bits 3` fails the quality bar, run an AWQ calibration pass to recover quality. `jang_tools` already has AWQ infrastructure (`awq.py`, `awq_capture.py`, `awq_scales.py`). Apply AWQ pre-quant scales, then `mlx_lm.convert` over the scaled weights.

**1d. Group-size sweep**: per-shape group sizes between {32, 64, 128} have different speed/quality tradeoffs. The MLX kernel ships specialized variants for each. Bench all three on the gate/up/down shapes; pick the best per-projection if needed.

Deliverables:
- `research/experiments/m5-affine/minimax-m2.7-affine-{2,3,4}bit-{32,64,128}.json` — pp/s, tok/s, MMLU, bundle size for each (bits, gs) variant.
- One winning bundle copied into `~/.mlxstudio/models/JANGQ-AI/MiniMax-M2.7-AFFINE-NAX/` (working dir) once a recipe meets the quality+size+speed bars.

### Phase 2 — Kimi-K2.6-Small affine conversion + bench

Reuse the MiniMax recipe. Validate it transfers to a different MoE family (Kimi K2 architecture is closer to DeepSeek than MiniMax). If quality drops more than for MiniMax at the same (bits, gs), the recipe needs per-family tuning. Same deliverables.

### Phase 3 — GLM-5.1 conversion + bench

Larger param count (357B), validates memory + load-time scaling. Same deliverables.

### Phase 4 — Decision: ship or don't, per target

For each target:

- If the affine bundle wins on pp/s AND tok/s AND quality is within ±0.5 pp MMLU AND bundle size ≤ 1.5× JANGTQ_K equivalent → publish to `JANGQ-AI/<MODEL>-AFFINE-NAX` on HF, mark as M5-only, document JANGTQ_K as the non-M5 alternative.
- If only some criteria are met → write per-target status note (e.g. "speed wins but quality regresses 0.8 pp, hold off on shipping pending AWQ tuning").
- If no recipe meets the bars → JANGTQ_K stays as the canonical format for that model, no NA path available without Steel-class custom kernel work.

Phase B (codebook NAX kernel) becomes interesting again only if Phase 4 returns "no recipe meets the bars" for a model whose users care about speed AND can't tolerate quality drop AND the bundle-size budget rules out affine.

## What this plan deliberately avoids

- **No new Metal kernels.** All NA work goes through MLX's existing affine kernels. Custom NA work for codebook formats is out of scope.
- **No new bundle format spec.** The output is a standard MLX quantized model directory, loadable by stock `mlx_lm.load()` / `vllm-mlx`'s loader / `vmlx-swift-lm`. No `jangtq_na.json`, no L1/L2 metadata, no custom converter machinery beyond what `mlx_lm.convert` + `jang_tools.awq` already provide.
- **No MoE routing histograms.** Affine `gather_qmm_nax` ships kernel
  variants for the standard bit-widths and group-sizes; per-expert batch
  shape variation is handled inside the gather kernel rather than at the
  Python level. The Phase A custom-NA-kernel routing-histogram concerns
  do not transfer to MLX's path. **However**: this assumption will only
  be confirmed on real workloads in Phases 1+ when an MoE target is
  benched. Phase 0 is dense-only (Qwen3.6-27B-JANG_4M), so MoE-gather
  behavior is not exercised yet.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| `mlx_lm.convert` doesn't support a target architecture | Medium for newer models (MiniMax M2 wrapper, Kimi K2 if novel) | Patch `mlx-lm` upstream is heavy; alternative is a thin per-architecture sanitizer in `jang_tools` that emits MLX-shaped weights from the HF source. We already do similar sanitization for VL models. |
| Affine 2-bit quality drop > 0.5 pp MMLU | Medium-high based on prior JANGTQ research | AWQ calibration. Worst case: settle for 3-bit or 4-bit affine, accept larger bundle. |
| Affine 4-bit bundle size > 1.5× JANGTQ_K → can't ship | Medium for 200B+ MoE | Mixed-bit per-projection recipe (gate/up at 4-bit, down at 2-bit) similar to JANGTQ_K's mixed strategy, calibrated per-tier. |
| Single-token decode tok/s regresses (NA kernels are prefill-tuned, may not help decode) | Medium | Phase 0 will surface this. If decode regresses, gate the affine bundle on long-prefill workloads only and keep JANGTQ_K as the canonical short-decode bundle. |
| JANGTQ_K already faster than MLX-affine on this hardware due to JANGTQ's hand-tuned decode kernels | Low but real | If true, the answer is "no general NA win available; JANGTQ_K wins on M5 Max for chat-length workloads." Document and move on. |

## Open questions to resolve in Phase 0

1. Does `mlx_lm.convert` produce a working bundle for the MiniMax M2 architecture, or does it choke on the custom `model_type: minimax_m2`? Same for Kimi K2 architecture.
2. What pp/s and tok/s does Qwen3.6-27B-JANG_4M actually achieve on this M5 Max? Is the NA advantage measurable at the bundle level, or has it always been there and we just never benchmarked?
3. Is `jang-tools` already producing some bundles in MLX-affine format (i.e. is JANG_4M already an NA-eligible bundle)? If yes, the conversion work for some targets may already be done.

These three questions get answered in Phase 0 before any converter work begins.

## Files this plan touches (all gitignored except plan/spec)

- `docs/superpowers/plans/2026-05-08-m5-affine-sidecar-plan.md` (this file, tracked)
- `research/experiments/m5-affine/*.json` (gitignored — bench results)
- `jang-tools/jang_tools/convert_*_affine_nax.py` per target (gitignored — converter scripts following existing converter conventions)
- `~/.mlxstudio/models/JANGQ-AI/<MODEL>-AFFINE-NAX/` (winning bundles, outside repo)

## Next concrete step

Run Phase 0 — measure Qwen3.6-27B-JANG_4M on M5 Max. If we already have a converted bundle on disk (per the survey, `Qwen3.6-27B-JANG_4M-CRACK` exists), bench that. Output goes into `research/experiments/m5-affine/qwen3.6-27b-jang_4m-baseline.json`.
