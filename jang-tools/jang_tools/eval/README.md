# JANG eval harness — MMLU two-pass

Runs MMLU on any JANG bundle (bf16 / JANGTQ / MXFP4 / affine) with two
passes per the project convention:

| pass | sampler | max_new | thinking | rationale |
|---|---|---|---|---|
| **no-reasoning** | greedy `temp=0` | 20 | OFF (`enable_thinking=False`, `<think>`/`</think>` blocked via logit processor) | direct answer; matches the GLM-5.1 / DSV4 fair-comparison pattern |
| **reasoning** | `temp=1.0 top_p=0.95` | 2048 | ON (`enable_thinking=True`) | lets the model think then extract the A/B/C/D after the closing tag |

Reasoning-toggle behavior follows each model's `chat_template.jinja` —
DSV4-family models honor `enable_thinking`, MiniMax always reasons (the
flag is harmless), Qwen3.6 ditto.

## Usage

```bash
# Single model, both passes, default 20 questions × 10 subjects = 200 each pass
python -m jang_tools.eval.mmlu \
    --src JANGQ-AI/Laguna-XS.2-JANGTQ \
    --mode both --out laguna_jangtq_mmlu.json

# Just direct-answer pass for a quick smoke
python -m jang_tools.eval.mmlu --src ... --mode no-reasoning --qps 5

# All 4 fresh bundles
bash scripts/eval_mmlu_all.sh                # default QPS=20
QPS=5 bash scripts/eval_mmlu_all.sh          # quick smoke
```

## Generation config rules baked in

- **Greedy + no rep penalty** for the direct-answer pass (avoids cache-decode
  drift; learned from the GLM-5.1 runtime audit).
- **`enable_thinking`** is the single switch — never strip `<think>` from
  templates by hand; let the chat template + the logit-block processor
  handle it.
- **Reasoning pass**: `temp=1.0 top_p=0.95` — matches DeepSeek's recommended
  pass@1 sample for chain-of-thought.
- The harness extracts `A/B/C/D` by scanning the response for the first
  letter token after the literal "ANSWER" (when present), else the first
  alphabetical letter — matches the `bench_humaneval`-style extractor.

## Notes by bundle format

- **MXFP4** loads via `mlx_lm.load()` for any arch in mainline mlx_lm; for
  custom arches (mistral3, laguna) it loads via the matching
  `jang_tools.<arch>.runtime.load()`.
- **JANGTQ** Python path uses `jang_tools.turboquant.codebook.compute_codebook`
  to derive the codebook from `(in_features, bits)` deterministically — no
  separate sidecar needed for the Python eval. (Swift requires the sidecar;
  see warning banners on older OsaurusAI repos.)

## Output JSON schema

```jsonc
{
  "src": "/path/to/bundle",
  "weight_format": "mxtq" | "mxfp4" | ...,
  "subjects": [...],
  "qps": 20,
  "passes": [
    {"mode": "no-reasoning", "correct": 173, "total": 200, "by_subject": {...}, "elapsed_s": 84.2},
    {"mode": "reasoning",    "correct": 184, "total": 200, "by_subject": {...}, "elapsed_s": 1234.5}
  ]
}
```
