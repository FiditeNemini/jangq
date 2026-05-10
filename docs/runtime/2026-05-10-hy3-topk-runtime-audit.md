# Hy3 / Top-K Runtime Audit

Created 2026-05-10.

## Scope

Audit of recent Hy3 runtime, JANGTQ2 bundle, Top-K override, and MTP notes.
This document is for repo/runtime handoff. Public model cards must use shorter
runtime-status wording and must not imply benchmark validation.

## Confirmed Current State

| Area | Result |
|---|---|
| Hy3 source | `/Users/eric/models/Tencent/Hy3-preview`, 557 GB, 112 shards. |
| Built bundle | `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2`, 79 GB. |
| Unbuilt profiles | `Hy3-preview-JANGTQ_K` / `JANGTQK` do not exist locally as of this audit. |
| JANGTQ2 bit policy | Routed experts 2-bit MXTQ; attention/shared/dense/embed/lm_head/MTP matmuls 8-bit affine; norms/router/bias passthrough. |
| MTP | Hy3 MTP weights are preserved but JANG/vmlx decode is normal autoregressive decode (`preserved_disabled`). |
| Python runtime | `jang_tools.hy3` exists and wraps `mlx_lm.models.dots1` with Hy3 namespace/key remapping. |
| Swift runtime | Still needs implementation in `../vmlx-swift-lm`; do not claim Swift runtime support for Hy3 yet. |
| Top-K override | Mechanically verified on Hy3 skeleton: K=4 patches 158 attrs (`79 sparse layers * 2`). |
| MiniMax Top-K | MiniMax-M2.7-Small-JANGTQ was smoke-tested at K=4; useful smoke only, not benchmark proof. |

## Fixes Applied

1. `load_jangtq_model` now auto-registers `hy_v3` before MLX skeleton load,
   matching the Hy3 README/runtime contract.
2. `JANGTQ_TOPK_OVERRIDE` invalid values now fail loudly. The loader no longer
   silently swallows `ValueError` and continues at trained K.
3. `apply_topk_override` now records original/trained K per patched attribute
   and refuses to increase above it. Lowering K and restoring K are allowed.
4. Top-K documentation now says runtime coverage is smoke-tested for Hy3 and
   MiniMax small only; other families are candidates until a forward-pass test
   exists.

## Top-K Correctness Boundary

The override is a runtime flag, not a quantization profile:

```text
JANGTQ_TOPK_OVERRIDE=4
```

It changes the number of selected experts after the router scores all experts.
It does not remove experts from the bundle and does not change tensor shapes.

Validated facts:

- Hy3/dots1-style router uses `Dots1TopkRouter.top_k` at forward time.
- Hy3 outer MoE also carries `num_experts_per_tok`.
- A Hy3 skeleton built from the source config patches exactly 158 attributes.
- The MiniMax small smoke confirms MiniMax reads `num_experts_per_tok` at
  forward time and K=4 produced coherent short outputs.

Not validated:

- HumanEval, MMLU, long-context, multi-turn, batched serving, and K below
  trained/2.
- DeepSeek-V4, Qwen-MoE, Bailing/Ling, Laguna, or full MiniMax large bundles.

## MTP Boundary

Hy3 has `num_nextn_predict_layers=1`, but current JANG runtime strips/drops the
MTP layer during sanitize and decodes one token per forward pass. Future MTP
runtime work must keep speculative draft state separate from accepted base KV
and only commit accepted tokens.

Required public wording until that exists:

```text
MTP tensors preserved; speculative decoding disabled pending runtime implementation.
```

## 128 GB Profile Boundary

`JANGTQ2` is the correct 128 GB candidate. The measured local bundle is 79 GB
and the planning estimator puts 4K-context runtime use near 102 GB with a 12 GB
headroom reserve.

`JANGTQ_K` is still a quality candidate, but it has not been built locally and
the estimator puts it near 126.6 GB runtime use at 4K context. Do not call it
comfortable on 128 GB without a measured load proof.

## Verification Commands

```sh
uv run --project jang-tools pytest -q jang-tools/tests/test_topk_override.py
uv run --project jang-tools python -m py_compile \
  jang-tools/jang_tools/topk_override.py \
  jang-tools/jang_tools/load_jangtq.py \
  jang-tools/jang_tools/hy3/model.py \
  jang-tools/jang_tools/hy3/runtime.py
uv run --project jang-tools python jang-tools/examples/mtp/estimate_jangtq_fit.py \
  /Users/eric/models/Tencent/Hy3-preview --profile JANGTQ2
uv run --project jang-tools python jang-tools/examples/mtp/estimate_jangtq_fit.py \
  /Users/eric/models/Tencent/Hy3-preview --profile JANGTQ_K
```
