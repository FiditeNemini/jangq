# JANGTQ inference-time top-K override

Created 2026-05-10.

## What it is

A universal MoE inference-time speedup that lowers the per-token expert
count below the value the model was trained with. Mechanism: walk the
loaded model after JANGTQ hydration and override every router's
`top_k` attribute (and any sibling `num_experts_per_tok` attribute on
outer MoE containers) to a smaller K.

Mechanism is universal: the converter ships every routed expert weight
regardless of K, the router still scores all 192/256/etc. experts every
step, and only the `argpartition` step that picks K winners is changed.
No re-conversion, no quality-impacting changes to weights, no per-layer
surgery — one integer counter per router.

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

### Env var

```sh
JANGTQ_TOPK_OVERRIDE=4 python -m jang_tools.examples.hy3.smoke_decode
```

`load_jangtq_model` reads `JANGTQ_TOPK_OVERRIDE` after hydration and
applies it automatically. Empty / unset / `0` is a no-op.

### Reverting

Call again with the original K, or unset the env var and reload.
The override is non-destructive — model weights and module shapes are
untouched.

## Universal across MoE families

Tested attribute paths exist in all of:

| Family | Router class | top_k path |
|---|---|---|
| Hy3-preview (this) | `Dots1TopkRouter` | `mlp.gate.top_k` |
| dots1 | `Dots1TopkRouter` | `mlp.gate.top_k` |
| DeepSeek-V3 / V4 | `DeepseekV3MoE.gate` | `mlp.gate.top_k` |
| Qwen3-MoE | `MoeBlock` | `mlp.top_k` |
| Bailing v2.5 / Ling | `BailingTopkRouter` | `mlp.gate.top_k` |
| Laguna-XS.2 | `LagunaRouter` | `mlp.router.top_k` |
| MiniMax M2.7 | `MiniMaxRouter` | `mlp.gate.top_k` |
| ZAYA | (top-1 only) | n/a — silent skip |

`apply_topk_override` does not hardcode any of these — it walks
`named_modules` and patches any module whose `top_k` is an int.

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
    n = 0
    for path, mod in model.named_modules():
        if hasattr(mod, "top_k") and isinstance(mod.top_k, int):
            mod.top_k = k; n += 1
        if hasattr(mod, "num_experts_per_tok") and isinstance(mod.num_experts_per_tok, int):
            mod.num_experts_per_tok = k; n += 1
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
