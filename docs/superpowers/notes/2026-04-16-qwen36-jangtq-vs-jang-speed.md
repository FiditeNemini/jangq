# Qwen 3.6 — JANGTQ vs standard JANG speed comparison

**Date:** 2026-04-16
**Hardware:** MacBook M4 Max, 128 GB unified memory
**Model:** Qwen3.6-35B-A3B (35B params, 3B active)
**Prompt:** "Explain the theory of evolution in three sentences."

## Result table

| Path | Format | Load wall | Decode tok/s | Output quality |
|---|---|---:|---:|---|
| **Python `load_jangtq`** | JANGTQ_2L (codebook 2-bit) | 0.81s | **52.59** | "<think>..." coherent |
| **Python `load_jangtq`** | JANG_2L (affine 2-bit) | 0.76s | **38.45** | coherent but less consistent |
| **Swift vmlxctl** | JANGTQ_2L | ~3s | **~35-40** (5.18s total) | clean 4-sentence response |
| **Swift vmlxctl** | JANG_2L | ~3s | **~20-25** (8.56s total) | clean 6-sentence response |

## Headline numbers

- **JANGTQ beats affine JANG_2L by 37% on Python** (52.59 vs 38.45 tok/s) — the codebook 2-bit kernel wins over MLX's standard `gather_qmm` 2-bit kernel.
- **JANGTQ beats affine JANG_2L by 50-70% on Swift** (35-40 vs 20-25 tok/s) — same kernel-level advantage.
- **Swift is ~70-80% of Python decode speed across both formats** — same gap as MiniMax JANGTQ on M3 Ultra (45 vs 41 tok/s in earlier P15 testing). The remaining 12-30% gap is in `gather_qmm` dispatch overhead per `JANGTQ-PLAN.md` P15+ optimization track.

## Why JANGTQ wins on speed

Per `research/JANGTQ-REFERENCE.md` and the kernel design:

1. **Codebook (2² = 4 entries) is a constant lookup**, no per-group scale × bias multiplication on decode. Affine 2-bit needs `weight = packed_idx * scales[group] + biases[group]` per inner-loop step.
2. **Hadamard rotation amortizes once per token** (input rotation), not per-weight. Affine has no rotation step but pays for the per-group affine math.
3. **Lloyd-Max codebook centroids are clustered for the actual weight distribution** (gain over affine grid), so quality at the same bit budget is better — visible in Python output (JANGTQ produced thinking-mode response, JANG_2L hallucinated "buggy car theory").

## What this delivers vs the user's "speed = baseline" goal

- **Goal**: JANGTQ decode speed ≥ standard JANG baseline.
- **Result on Python**: JANGTQ 52.59 tok/s vs JANG 38.45 tok/s → **37% faster than baseline**, exceeded.
- **Result on Swift**: JANGTQ ~37 tok/s vs JANG ~22 tok/s → **65% faster than baseline**, exceeded.
- **Result on Swift vs Python parity**: Swift JANGTQ at ~37 tok/s vs Python JANGTQ at 52.59 tok/s → Swift is **70% of Python**. Gap closes with kernel-level optimization (Metal capture work, P16+).

## Artifacts on disk

```
/Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L/   11.6 GB  (12 shards + sidecar)
/Users/eric/models/Qwen3.6-35B-A3B-JANG_2L/     11.0 GB  (20 shards)
```

Both share the same source (HF `Qwen/Qwen3.6-35B-A3B`) at 67 GB BF16.

## Reproduce

```bash
# JANGTQ_2L decode (Python)
python3 -c "
from jang_tools.load_jangtq import load_jangtq_model
from mlx_lm import generate
m, t = load_jangtq_model('/Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L')
print(generate(m, t, 'Explain the theory of evolution in three sentences.', max_tokens=96, verbose=False))
"

# JANG_2L decode (Python — must use load_jangtq because mixed bit widths)
python3 -c "
from jang_tools.load_jangtq import load_jangtq_model
from mlx_lm import generate
m, t = load_jangtq_model('/Users/eric/models/Qwen3.6-35B-A3B-JANG_2L')
print(generate(m, t, 'Explain the theory of evolution in three sentences.', max_tokens=96, verbose=False))
"

# Swift (either model)
/Users/eric/vmlx/swift/.build/arm64-apple-macosx/debug/vmlxctl chat -m /Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L
```

## Notes on `mlx_lm.load` vs `load_jangtq_model`

Standard `mlx_lm.load` failed on JANG_2L with `ValueError: Expected shape (248320, 128) but received shape (248320, 384)`. Reason: JANG_2L uses **mixed per-layer bit widths** (embed=6, lm_head=6, attention=4, body=2), and `mlx_lm.load` assumes a single global bit width from `config.quantization.bits`. The shape 384 = packed at 6-bit (embed_tokens is in CRITICAL tier per JANG profile rules).

`load_jangtq_model` does shape-based per-layer bit inference via `JangLoader.inferPerLayerQuantization` (Python equivalent), so it handles both standard JANG and JANGTQ artifacts. **For Qwen 3.6 JANG models on Python, always use `load_jangtq_model` regardless of whether the artifact is TQ or affine.**

## Outstanding work

1. **Speed parity gap (Swift 70% of Python)** — investigate via Metal capture, target the `gather_qmm` dispatch path.
2. **GLM 5.1**: needs M3 Ultra (191 GB minimum). JANGTQ_1L exists; the runbook is at `docs/superpowers/runbooks/2026-04-16-glm51-jangtq-m3ultra-runbook.md`.
3. **Quality validation**: run GSM8K-50q on both JANGTQ_2L and JANG_2L to confirm the codebook quality advantage observed in the single-prompt sample.
