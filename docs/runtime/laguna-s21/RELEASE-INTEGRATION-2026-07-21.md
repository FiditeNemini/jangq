# Laguna S 2.1 JANG integration checkpoint — 2026-07-21

## Verdict

`SOURCE-TESTED / ENGINE-LIVE-PROOF-PENDING`

This record covers the clean `jang-tools` source integration only. It does
not claim that the Python/Electron vMLX release, every API protocol, or the
real model cache paths are release-verified. Those runtime gates remain in
the vMLX release matrix and must be proved with current-source API streams and
the live Electron application.

## Repository state

- Repository: `/Users/eric/jang-release-prep-20260721`
- Integration branch: `codex/laguna-release-sync-20260721`
- Base: `ca75f0c` (`origin/main` when this worktree was created)
- Source integration head: `dd220b8`
- Remote branch: `origin/codex/laguna-release-sync-20260721`
- Fast-forward relationship: `ca75f0c` is an ancestor of `994b811`
- Dirty `/Users/eric/jang` checkout was not reset or overwritten.

Integrated commits, oldest first:

1. `c6dfb47` — mixed per-module affine bits and per-head/per-element gating
2. `ddb5639` — per-layer SWA masks and `RotatingKVCache(keep=0)`
3. `39bf085` — BF16 activation stream and wired-limit handling
4. `19b021d` — Laguna S 2.1 converter/capability registration
5. `f8127a2` — AWQ capture/search correction
6. `d2b4541` — runtime notes, example, and Swift port checklist
7. `994b811` — 1-bit affine storage metadata and pack/unpack coverage
8. `f5d4077` — keep 1-bit storage out of semantic quantization/allocation
9. `dd220b8` — validate storage widths and correct Laguna AWQ provenance

The 1-bit work is required for affine 1-bit bundles such as Bonsai. It does
not convert JANGTQ/MXTQ Hadamard-codebook bundles into affine JANG, and it does
not change base MLX MXFP routing. Those formats remain distinct.

## Source tests

The complete clean-worktree suite was rerun after independent review and the
storage-width correction:

```text
573 passed, 37 skipped, 2 warnings in 12.87s
```

The final focused allocator/quantizer/format run completed before the full
suite:

```text
76 passed
```

The two warnings were test-environment warnings, not failed assertions. No
real model generation or vMLX UI/API result is inferred from these tests.

Independent review caught and blocked a cross-family regression before main:
temporarily adding 1 to global semantic `ALLOWED_BIT_WIDTHS` also inserted it
into the generic allocator downgrade path. A deterministic 2-bit budget could
therefore assign 1-bit to ordinary dense tensors. The corrected contract keeps
semantic allocation/quantization at `{2,3,4,5,6,8}`, exposes 1-bit only to the
packed-storage helpers, and makes generic tensor quantization reject semantic
1-bit. Regression coverage pins both expanded and compact allocators.
The writer now also rejects unsupported `storage_bits` metadata rather than
copying arbitrary integers into `config.json`.

The Python package build also completed from the clean worktree:

```text
Successfully built jang-2.5.31.tar.gz and jang-2.5.31-py3-none-any.whl
```

Setuptools emitted a future deprecation warning for the table form of
`project.license`; it did not fail this build and remains a follow-up before
the February 2027 enforcement date.

The published Laguna bundle sidecars report AWQ enabled, but the scale files
named by the historical notes were not present on the build Mac or external
drive during this audit. The conversion examples now require an explicit AWQ
path and label it as a placeholder. Exact published-bundle reproduction stays
blocked until those scales are recovered or regenerated and revalidated; the
docs no longer imply that omitting AWQ is equivalent.

## Real-bundle source facts inspected

Bundles on the original M5 external drive:

- `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_2L`
- `/Volumes/EricsLLMDrive/jangq-ai/Laguna-S-2.1-JANG_4M`
- `/Volumes/EricsLLMDrive/dealignai/Bonsai-27b-1bit-JANG-CRACK`

Laguna S 2.1 bundle facts:

- `model_type=laguna`
- 48 layers: 36 sliding-attention and 12 full-attention
- sliding window 512
- gating is per-head
- 527 per-module quantization overrides in each inspected bundle
- 2L override widths: 8-bit 241, 6-bit 145, 3-bit 47, 2-bit 94
- 4M override widths: 8-bit 385, 6-bit 1, 4-bit 141
- current Laguna configs do not use `storage_bits`

Bonsai affine 1-bit bundle facts:

- top-level affine quantization is 1-bit, group size 128
- 498 module rows explicitly carry `bits=1` and `storage_bits=1`
- profile is `JANG_AFFINE_1BIT`
- this is not JANGTQ/MXTQ

## Runtime gates still required before the vMLX release can cite this row

1. Load both Laguna 2L and 4M through the real Electron Start flow and verify
   bundle-derived reasoning, tool, sampling, cache, and quant controls.
2. Prove reasoning-on Auto, reasoning-off, and parser-separated
   `reasoning_content`/content deltas through Chat Completions, Responses,
   Anthropic, and Ollama.
3. Complete a real tool-result continuation and a multi-tool interleaved
   reasoning loop without inline think-tag leakage or empty final content.
4. Prove a coherent prompt longer than 512 tokens so the mixed full/SWA path
   is exercised, with truthful TTFT and decode-rate accounting.
5. Prove q4 TurboQuant applies only to the KV-bearing component selected by
   the engine policy, while rotating/native state follows its documented
   rederive path.
6. Prove cold, RAM-warm, post-eviction disk-L2, restart-from-disk, and partial
   prefix reuse. Include disk-only operation with paged RAM disabled.
7. Verify saved server and chat settings survive restart and that the UI,
   launched argv, capabilities/health, and effective runtime values agree.
8. Run the vMLX full suites, bundle the exact clean JANG source, and complete
   the signed/notarized Sequoia and Tahoe release gates.

## Release integration rule

The Python/Electron bundler should use the clean exact path:

```text
VMLINUX_JANG_TOOLS_SOURCE=/Users/eric/jang-release-prep-20260721/jang-tools
```

Do not bundle from the dirty `/Users/eric/jang` checkout. A future cleanup may
reconcile that checkout, but this checkpoint preserves its unrelated user and
agent work.
