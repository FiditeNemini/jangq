# MTP Runtime Integration Plan

Created 2026-05-09.

Scope: Python `../vmlx`, Swift `../vmlx-swift-lm`, and JANG bundle metadata. Treat both runtime repos as read-only source material from this workspace unless explicitly approved.

## Current Status

| Surface | Current evidence | Hy3 MTP status |
|---|---|---|
| `jang-tools` | Converter can preserve MTP metadata and weights; no Hy3 local runtime yet. | `preserved_disabled` until a Python accept/reject loop exists. |
| `../vmlx` Python | Existing speculative paths and cache docs exist, but no verified Hy3 internal-MTP path was found in this pass. | Needs a small reference implementation before product wiring. |
| `../vmlx-swift-lm` Swift | Has speculative-decoding docs and multiple cache types; no `hy_v3` model dispatch or Hy3 MTP path yet. | Needs Hy3 model first, then MTP accept/reject on top. |
| MiniMax M2.7 local bundles | Good MoE/router/JANGTQ_K runtime analog; inspected configs show no active MTP. | Not an MTP test target unless another source config says otherwise. |
| Qwen/Gemma local bundles | Useful parser/cache regression targets; inspected configs did not expose MTP keys. | Keep as non-MTP regression baselines. |

## Required Runtime Contract

Every MTP-bearing model must expose one of these states:

```text
mtp_mode = none | preserved_disabled | enabled
```

- `none`: config and tensor census show no MTP.
- `preserved_disabled`: bundle contains MTP tensors, runtime decodes normally.
- `enabled`: runtime drafts with MTP and verifies drafted tokens before accepting them into base KV.

Do not use plain "MTP supported" wording.

## Cache Rules

Base KV cache contains accepted tokens only:

```text
base_kv[layer][accepted_position] = K/V for committed tokens
```

MTP draft state is temporary:

```text
draft_kv != base_kv
```

Acceptance rules:

- accepted draft token: recompute or safely commit through the base model path, then append to base KV
- rejected draft token: discard draft state
- interrupted or cancelled request: discard draft state
- prefix-cache reuse: cache key must include `mtp_mode`, model revision, quant profile, chat-template salt, and reasoning/tool-parser mode

Unsafe states to gate:

- `RotatingKVCache` after wrap, unless the runtime can rewind/truncate safely
- batched decode where one sequence accepts and another rejects unless slot-local draft state exists
- VLM image-token prefill unless draft cache is text-position aligned and media salt is included
- hybrid SSM/MLA/CCA caches until each companion state has explicit rollback semantics

## Python Bring-Up Order

1. Write a source/bundle inspector that reports `mtp_mode`, MTP tensor namespace, cache type, and quant profile without loading weights.
2. Add a minimal Hy3 reference wrapper that can run normal decode with MTP disabled.
3. Add one-token MTP draft behind a flag.
4. Add deterministic acceptance tests:
   - prompt with forced greedy token
   - accept path
   - reject path
   - cancellation path
   - two-turn cache continuation
5. Only then wire to server flags.

## Swift Bring-Up Order

1. Add `hy_v3` config/model dispatch with normal KV cache only.
2. Implement q/k RMSNorm GQA attention, sigmoid+bias top-k MoE, shared expert, dense layer 0.
3. Prove fresh-cache and continuation generation with MTP disabled.
4. Add `MTPDraftState` as a separate object from `[KVCache]`.
5. Implement one-token MTP accept/reject with rollback tests.
6. Add BatchEngine gating: enable only for slot-local caches that can discard draft state independently.

## Bundle Metadata

JANG bundles should carry:

```json
{
  "runtime": {
    "bundle_has_mtp": true,
    "mtp_layers": 1,
    "mtp_mode": "preserved_disabled"
  }
}
```

When enabled:

```json
{
  "runtime": {
    "bundle_has_mtp": true,
    "mtp_layers": 1,
    "mtp_mode": "enabled",
    "speculative_tokens": 1
  }
}
```

## First Model Order

1. Hy3: first true internal-MTP target.
2. MiniMax: first MoE/JANGTQ_K runtime regression target, but not MTP in current local configs.
3. Qwen and Gemma: parser/cache regressions and non-MTP controls.
4. Bailing/Ling and DeepSeek-V4: only after hybrid cache rollback rules are explicit.
