# JANGTQ inference-time top-K override

Created 2026-05-10.

## What it is

A MoE inference-time fast-mode flag that lowers the per-token expert
count below the value the model was trained with. Mechanism: walk the
loaded model after JANGTQ hydration and override every router's
`top_k` attribute (and any sibling `num_experts_per_tok` attribute on
outer MoE containers) to a smaller K.

The mechanism is simple where the runtime exposes K as a mutable integer:
the converter ships every routed expert weight regardless of K, the router
still scores all experts every step, and only the `argpartition` / top-k
winner count changes. No re-conversion, no weight surgery, and no tensor
shape changes.

This is not a new bundle profile and it is not a replacement for quality
benchmarks. It is an explicit runtime flag.

## How to use

### Function call

```python
from jang_tools.hy3 import register_mlx_lm_hy3
from jang_tools.load_jangtq import load_jangtq_model
from jang_tools.topk_override import apply_topk_override

register_mlx_lm_hy3()
model, tokenizer = load_jangtq_model("/path/to/Hy3-preview-JANGTQ2")
n_patched = apply_topk_override(model, 4)  # cut top-K from 8 to 4
```

`apply_topk_override` returns the count of router/MoE attributes
patched. Pass `None` (default) for no-op. Top-1 MoE families (ZAYA)
have no `top_k` and silently skip.

Guardrails:

- K must be a positive integer.
- K may lower the observed trained/original K.
- K may restore the original K after a lower-K call.
- K may not increase above the original K. That is treated as a config
  error and raises `ValueError`.

### Env var

```sh
JANGTQ_TOPK_OVERRIDE=4 python -m jang_tools.examples.hy3.smoke_decode
```

`load_jangtq_model` reads `JANGTQ_TOPK_OVERRIDE` after hydration and
applies it automatically. Empty / unset / `0` is a no-op. Invalid values
raise instead of silently falling back to the trained K.

### Reverting

Call again with the original K, or unset the env var and reload.
The override is non-destructive — model weights and module shapes are
untouched.

## Runtime Coverage

The patcher is intentionally generic (`named_modules()` + integer attrs),
but support should be described by observed runtime coverage:

| Family | Status | Patched runtime field(s) | Notes |
|---|---|---|---|
| Hy3-preview | Smoke-tested on local `Hy3-preview-JANGTQ2` | `mlp.gate.top_k`, `mlp.num_experts_per_tok` | 79 sparse layers -> 158 attrs. |
| MiniMax M2.7 Small JANGTQ | Smoke-tested locally on 2026-05-10 | `num_experts_per_tok` | 62 sparse blocks patched; short-prompt K=4 smoke looked coherent. |
| dots1 | Mechanically same router class as Hy3 | `mlp.gate.top_k` | Needs bundle-specific smoke before public claims. |
| DeepSeek-V3 / V4 | Candidate | family-specific | Needs forward-pass test; DSV4 cache/MLA behavior is not Hy3-like. |
| Qwen3-MoE | Candidate | family-specific | Needs forward-pass test. |
| Bailing v2.5 / Ling | Candidate | family-specific | Needs forward-pass test; has separate MTP/cache concerns. |
| Laguna-XS.2 | Candidate | family-specific | Needs forward-pass test. |
| ZAYA | No-op expected | none | Top-1 router; override should patch 0 attrs. |

Do not describe the feature as benchmark-validated on a family until that
family has an actual load + generation test at the requested K.

## Hy3-preview measured A/B (M5 Max 128 GB, 2026-05-10)

Bundle: `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2`
(JANGTQ2, 79 GB, 80 transformer layers, 192 routed experts, trained
`num_experts_per_tok=8`).

Greedy decode, 47 tokens (or to EOS), prompts varied between
chat-template and plain text. Same prompts each K.

| K | "What is 2+2?" | "The capital of France is" | "Write a Python function..." | Avg tok/s |
|---|---|---|---|---:|
| 8 (default) | `4` | `Paris.\nThe capital of France is Paris.…` (loop) | spec for `get_nth_fibonacci` | 15.75 |
| 6 | `4` | `Paris.\nThe capital of France is Paris.…` (loop) | full Fibonacci recurrence | **16.41** (+4.2%) |
| 4 | `2 + 2 = 4` | `Paris.\nThe capital of Italy is Rome.\nThe capital of Spain is Madrid.…` | spec for `get_nth_fibonacci` | **17.40** (+10.5%) |

158 attributes patched at K=6 and K=4 (79 sparse layers × 2 attrs each:
`gate.top_k` and `mlp.num_experts_per_tok`).

### Observations

- **Coherence holds at K=4** on these short prompts. No junk-token
  regressions (no doubled letters, no early-EOS, no looping that wasn't
  already in K=8 output).
- **K=4 actually broke the K=8 looping** on the "Capital of France"
  base-text prompt by producing a more diverse continuation (other
  capitals). This is luck-of-the-routing, not an indication that K=4
  is generally better — but it's interesting.
- **Speedup is modest**, not 2×. Routed FFN is one slice of per-layer
  cost; attention (Q/K/V/O proj + RoPE + SDPA + KV cache append),
  shared expert, layer norms, and lm_head are unchanged. K=4 saves
  ~50% of the routed FFN dispatches; total decode time drops ~10%.
- Long-form quality (reasoning chains, code with rare patterns) was
  **not** measured. Always A/B on a real benchmark before shipping
  K<trained as a default.

## MiniMax-M2.7-Small Smoke (2026-05-10)

Four short prompts were run on `MiniMax-M2.7-Small-JANGTQ` at K=8 and
K=4. K=4 patched the runtime `num_experts_per_tok` field across the
sparse blocks and measured roughly `30.4 -> 34.1 tok/s` (+12.2%).

The surprising short-prompt result was that K=4 looked at least as coherent
as K=8 on those prompts, including one case where K=8 emitted invalid
byte-like junk while K=4 produced coherent text. Treat this as a smoke
finding only. It is not a proof that K=4 is generally higher quality than
the trained K, and it does not cover HumanEval, MMLU, long-context,
multi-turn, or batch behavior.

## Quality risk profile

The model was trained with the original K. Reducing K means:

1. **Each surviving expert's weight is larger** under route_norm
   renormalization (weights sum to 1 → per-expert weight ≈ 1/K up from
   1/K_trained). The model's layer outputs end up with similar total
   magnitude but routed through fewer specialty heads.
2. **Specialty experts get dropped more often.** If the trained
   distribution was top-8 with 6 strongly-active and 2 weakly-active
   experts, dropping to top-4 may keep all 4 strong ones (no loss) or
   may evict one strong + one weak (real loss). Distribution depends
   on prompt content.
3. **Failure mode is graceful** — typically degraded coherence on hard
   prompts, not full-junk garbage. K=4 on Hy3 produced no junk on our
   smoke prompts.

## When to use

- **Bench latency, not paying for batch throughput** — K-override is
  pure single-token decode speedup; batched throughput on H100/A100
  benefits more from other levers (continuous batching, paged KV).
- **Want a "fast mode" toggle** — ship with K=trained as default,
  expose an env var or CLI flag for users who want speed.
- **Already past the qualitative cliff** — if K=8 is producing junk
  on your prompts (post-quantization quality issue), K=4 won't help;
  fix the upstream issue first.

## When NOT to use

- **As silent default behavior.** Always make the override explicit so
  bug reports are easy to triage.
- **Without a quality A/B.** Hy3 holds at K=4 on simple prompts;
  HumanEval, MMLU, long reasoning chains have not been tested.
- **Below trained K/2.** No MoE family has been validated below half
  the trained K. Hy3 trained=8, so K=4 is the floor we'd consider
  shipping; K=2 is "we don't know."

## Implementation notes

`jang_tools/topk_override.py`:

```python
def apply_topk_override(model, k: int | None) -> int:
    if k is None: return 0
    # First pass records the original/trained K and refuses K > original.
    for path, mod in model.named_modules():
        if hasattr(mod, "top_k") and isinstance(mod.top_k, int):
            ...
        if hasattr(mod, "num_experts_per_tok") and isinstance(mod.num_experts_per_tok, int):
            ...
    return n
```

Hooked into `load_jangtq.py` immediately after `Hydration complete` so
the override applies before any forward pass.

## Future work

- **Per-layer K**. Some research suggests early layers tolerate lower
  K than late layers (or vice versa). Override could accept a list of
  K per layer.
- **Quality A/B suite**. Wire HumanEval+ / MMLU / long-context tests
  to compare K=trained, K=trained-2, K=trained/2 on every MoE family
  in the JANG catalog. Output a published quality-vs-speed table.
- **Runtime auto-tune**. If acceptance-rate signal is available
  (compare routed-K logits vs main-model verification), dynamically
  raise/lower K per token. Adjacent to MTP speculative decoding.
