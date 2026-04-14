# Plan 7 — JANG MoE Decode Optimization

> Goal: make MiniMax-M2.7-JANG_2L (and JANG MoE models in general) decode faster than ~14 tok/s on M4 Max via vmlx. This plan lives on top of the fat-layout bundle fix from `docs/superpowers/notes/2026-04-14-minimax-jangspec-perf.md` — with that fix bundle and source paths are already at parity, so all speedups in this plan apply equally to both.

**Target:** beat the 14.35 tok/s source-directory baseline on MiniMax-M2.7-JANG_2L. Stretch goal: 20 tok/s on decode for this model class on M4 Max without quality loss.

**Non-goals:**
- Changing the JANG quantization format
- Retraining the model in any way
- Optimizing other model classes first — MiniMax is the benchmark
- Speculative decoding (Plan 8, separate effort)

## Baselines established

| Config | Load | RSS | Decode tok/s |
|---|---:|---:|---:|
| source dir → vmlx serve | 19 s | 55 GB | **14.35** |
| bundle (fat) → vmlx serve | 13 s | 56 GB | **14.23** |

Both are using the same model code, same Metal kernels, same KV cache, same mx.compile policy. Decode rate is now a pure function of the inference hot path, not the loader.

## Levers (priority order)

### L1 — Compile-shapeless fusion status check (hours)

vmlx-swift-lm's benchmark docs reference a `BENCH_COMPILE_DECODE=1` env flag and `compile(shapeless: true)` fusion islands that give 15–25% decode speedup on Qwen3.5-35B-A3B. **The question:** is this path actually wrapping the MiniMax decode hot body in the `serve` codepath, or only in the `RunBench` harness?

Steps:
1. Grep `vMLXEngine/` and `vMLXLMCommon/` for `compile(` wrappers and read what gates them
2. Look for `BENCH_COMPILE_DECODE` or equivalent env var reads
3. Look for `@Sendable func step(...)` / `TokenIterator.next()` fusion
4. Check MiniMax.swift / MiniMaxJANGTQ.swift model code for explicit `mx.compile` use
5. If fusion is off for serve: turn it on, rerun the 14.35 tok/s bench, measure delta
6. If fusion is on for serve: skip to L2

**Risk:** `compile(shapeless: true)` has historical crash correlation with JANG models on M2 Mac mini (see `HANDOFF-JANG-CRASH-813-814.md`) — but the crash was on Mac16,8 (M2), and we're on M4 Max. If enabling triggers issues, back out.

**Expected gain:** 15–25% if currently off, 0% if already on. Worth the investigation regardless.

### L2 — Cold expert pruning (days)

MiniMax's MoE uses top-8 routing across 256 experts. The softmax output over selected experts has a strong dominance tail — the top-2 experts typically account for ~60–70% of the routing weight, and the bottom few are near-zero contributions.

Proposal: after `topk`, drop experts whose gate weight is below a threshold (say 5% of the max selected weight). For top-8 with a typical distribution, this usually keeps 3–5 experts out of 8, halving the per-token MLP cost.

Quality impact: on Qwen-Next style MoE models, published results show <1% MMLU degradation at a 5% prune threshold. MiniMax is a newer architecture, needs a calibration run.

Implementation:
1. Add `enableColdExpertPruning: Float?` to the model config / GenerateParameters
2. In `MiniMaxModel` / `MiniMaxJANGTQModel` MoE dispatch, after router softmax, apply the threshold
3. Re-normalize the kept weights to sum to 1
4. Benchmark MMLU delta against source (use jang_tools' existing MMLU infra)
5. Ship the flag off-by-default; expose via CLI `--cold-expert-prune 0.05`

**Expected gain:** 20–40% decode speedup. Most direct win of any lever in this plan.

### L3 — JANGTQ P13+ Metal kernel extensions (weeks)

Per `Package.swift`, the existing stack is P1–P12 covering compile-G fusion, decode fusion, subgroup matrix instructions, MXTQ dequant optimization, and the TurboQuant codebook path. Extensions with the highest ROI for MiniMax:

- **2-bit-specific gather matmul kernel**: the current `gather_qmm` has a variable-bit-width path; a dedicated 2-bit variant can skip some dispatch overhead. MiniMax JANG_2L is 2-bit dominant.
- **Fused expert dispatch**: run 2 or more top-k experts in one kernel launch via structured gather, amortizing launch overhead. Works well for top-8 where launches otherwise sum to per-layer cost.
- **Activation caching**: SiLU of gate + up is idempotent per expert; cache across repeated expert selection within a long decode. Limited gain in normal decode, potentially meaningful with spec-dec K>1 verification.

**Expected gain:** 10–30% depending on which items ship. 2–4 weeks of focused Metal work.

### L4 — Speculative decoding with Metal draft (weeks)

This is Plan 8, tracked separately. Briefly: small dense JANG draft runs on Metal at 80+ tok/s, proposes K tokens, MiniMax target verifies K tokens in one forward pass, accept prefix with ~60% rate → effective ~30–50 tok/s on MiniMax target. ANE path is dead (see ANE spike in the perf notes) but Metal-draft path remains.

5–6 weeks scope. Out of scope for this plan.

## Sequence of work

**Week 0 (this session):**
1. L1 investigation: compile-shapeless status in the serve path. Either flip it on and rebench, or confirm it's already on.
2. Cleanup: default the bundle builder to fat-only, gate per-blob behind `--streaming`. Halves bundle disk size (63 GB saved on MiniMax).

**Week 1–2:**
1. L2: Cold expert pruning implementation + MMLU validation run.
2. Re-measure MiniMax decode. Record result.

**Week 3+:**
1. L3: JANGTQ P13+ kernel work. Scope selected items based on profiling.
2. Plan 8 spec-dec kicks off in parallel.

## Gates

- After L1: if decode ≥ 18 tok/s we're at target. If < 16 tok/s, continue.
- After L2: if decode ≥ 22 tok/s with <1% MMLU regression, consider it the default. Otherwise roll back and keep the threshold flag opt-in.
- After L3: the delta from each P13+ item should be isolated with a micro-bench before merging. The existing `vmlxctl serve` benchmark is the macro-bench.

## What NOT to do

- Re-quantize MiniMax into a different format (e.g., uniform 4-bit). That's a separate conversion, doesn't help the generic JANG path, and loses the mixed-precision quality advantage.
- Port to ANE. Dead end for this workload — see perf notes.
- Change the bundle format. Fat layout is the answer.
- Optimize loading further before decode. Loading is already competitive with source (31% faster to first token) and isn't the bottleneck.

## Success criteria

- MiniMax-M2.7-JANG_2L via vmlxctl serve reaches ≥ 18 tok/s sustained decode on 256-token generation (M4 Max, clean machine).
- Same path on Gemma-4-26B-A4B-it-JANG_4M reaches ≥ 45 tok/s (up from 39.4).
- Source-directory path hits the same numbers (since the optimizations apply equally).
- Zero regression on coherence — diff-tested against the baseline output via the same prompt.
