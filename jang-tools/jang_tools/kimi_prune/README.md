# `jang_tools.kimi_prune`

Routing-aware expert pruning (REAP-style) for Kimi K2.6 and similar
DeepSeek-V3-architecture MoE models. Produces a smaller-but-equivalent
checkpoint that preserves task performance, then quantizes via JANGTQ.

## Pipeline

```
build_calib_v2  →  profile  →  score  →  prune  →  convert_kimi_jangtq  →  eval
```

1. `build_calib_v2.py` — assemble mixed-domain calibration corpus (default
   mix: 24% code / 20% agentic / 20% general / 10% academic_mc / 8% science /
   8% Chinese / 5% cyber / 3% systems / 2% long-context).

2. `profile.py` — stream calibration shards through model, capture per-layer:
   - routing frequency
   - weighted frequency (× expert weight)
   - co-activation matrix
   - output energy

3. `score.py` — combine signals into per-expert importance scores.

4. `prune.py` — drop low-importance experts, absorb-merge co-activated
   neighbors into the survivors, rewrite router weight rows accordingly,
   re-save FP8 compressed-tensors shards.

5. `convert_kimi_jangtq.py` — JANGTQ conversion of the pruned bundle.

6. `bench_humaneval*.py`, `bench_mmlu*.py`, `bench_text.py` — eval harnesses
   covering code, MC reasoning, generation.

## Quick example

```bash
python -m jang_tools.kimi_prune.build_calib_v2 \
  --out <path/to/calib.jsonl> --tokens 200_000

python -m jang_tools.kimi_prune.profile \
  --model <path/to/Kimi-K2.6> \
  --calib <path/to/calib.jsonl> \
  --out <path/to/profile.npz>

python -m jang_tools.kimi_prune.score \
  --profile <path/to/profile.npz> \
  --out <path/to/scores.npz>

python -m jang_tools.kimi_prune.prune \
  --model <path/to/Kimi-K2.6> \
  --scores <path/to/scores.npz> \
  --keep_pct 70 \
  --out <path/to/Kimi-K2.6-REAP-30>

python -m jang_tools.kimi_prune.convert_kimi_jangtq \
  --src <path/to/Kimi-K2.6-REAP-30> \
  --dst <path/to/Kimi-K2.6-REAP-30-JANGTQ2> \
  --profile 2
```

## Target models

Default tuned for Moonshot Kimi K2.6 (61 layers, 384 routed + 1 shared,
top-8 routing, MLA). Same pipeline applies to:
- DeepSeek-V3.x variants
- GLM-4.7
- MiniMax-M2+
- Qwen3-Coder-480B+

Each new model needs:
1. Profile capture pass (architecture-aware hooks)
2. Calibration mix tuned to model's training distribution
3. Quantization profile selection (2-bit usually OK for 64+ expert MoE)

See `research/JANGREAP-LESSONS.md` for the canonical bug/trap doc — read
before running on a new model.
