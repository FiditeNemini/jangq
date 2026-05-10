# Codex Audit: JANG Repo + Runtime State

Timestamp: 2026-05-09 21:42 PDT.

Scope: audit Claude's latest JANG/JANGQ/jang-tools/vmlx/vmlx-swift-lm status before any push, upload, or public runtime claim.

## Bottom Line

Do not push, reset, merge, or upload yet.

The Hy3 converted bundle is structurally good, but the runtime ports are not complete. `../vmlx` has no Hy3 Python runtime implementation. `../vmlx-swift-lm` has a Hy3 recognition gate that intentionally throws, not a runnable Hy3 decoder. Public README/model-card text must say runtime pending and MTP `preserved_disabled`.

## Verified Good

- Local bundle exists:
  - `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2`
  - size: `79G`
  - files: 93 total, 85 safetensor shards
  - source download: `/Users/eric/models/Tencent/Hy3-preview`, `557G`
- Bundle integrity:
  - `config.json`, `jang_config.json`, `model.safetensors.index.json`, `jangtq_runtime.safetensors`, tokenizer/template files are present.
  - output index references 85 shards and none are missing.
- Capability verifier:
  - `uv run --project jang-tools python -m jang_tools.verify_capabilities /Users/eric/models/JANGQ/Hy3-preview-JANGTQ2`
  - result: `PASS`, `family=hy_v3`.
- Converter dry-run on complete source:
  - 47,138 source tensors
  - 45,504 routed expert tensors -> MXTQ 2-bit
  - 1,146 dense/attention/shared/embed/lm_head/MTP matmuls -> affine 8-bit
  - 488 norms/router/expert-bias tensors -> fp16 passthrough
- Hy3 example scripts:
  - Python runtime skeleton emits valid JSON.
  - Python parser sample emits valid JSON.
  - Swift runtime skeleton emits valid JSON.
  - Swift parser sample emits valid JSON.
- `../vmlx-swift-lm` library target:
  - `swift build --target MLXLLM` passes.

## Not Verified / Still Blocked

- `../vmlx-swift-lm` tests:
  - `swift test --filter Hy3RegistrationTests` does not complete because the package test build still fails on `Tests/MLXLMTests/BailingGLATests.swift:3:8: no such module 'XCTest'`.
  - Do not claim Hy3 Swift tests are green.
- Hy3 runtime:
  - `Libraries/MLXLLM/Models/Hy3.swift` is a configuration/recognition contract only.
  - `LLMModelFactory.dispatchHy3Unsupported` decodes the config and then throws `unsupportedModelType`.
  - This is acceptable as a guard, but it is not runtime support.
- Python `../vmlx` runtime:
  - No Hy3 Python engine implementation was found in `/Users/eric/vmlx`.
  - Existing matches are docs/queues/package-lock artifacts, not model code.
- MTP:
  - Hy3 MTP tensors are preserved in the bundle.
  - No JANG/vmlx/vmlx-swift-lm accept/reject speculative decode path is implemented for Hy3.
  - Required public wording: `MTP tensors preserved; speculative decoding disabled pending runtime implementation`.

## Repo Hygiene / Git State

- `/Users/eric/jang` is not clean from Codex's current view:
  - untracked source files:
    - `jang-tools/jang_tools/zaya/README.md`
    - `jang-tools/jang_tools/zaya/__init__.py`
    - `jang-tools/jang_tools/zaya/model.py`
    - `jang-tools/jang_tools/zaya/runtime.py`
  - ignored bytecode exists under `jang-tools/jang_tools/zaya/__pycache__/`.
- Branch/divergence:
  - current branch: `main`
  - `HEAD`: `de9ae04`
  - `origin/main`: `79caa61`
  - divergence: origin is 19 commits ahead, local is 6 commits ahead.
  - Do not force-push.
- Recommended GH path:
  - push Claude/Codex work to a feature branch after explicit approval, or first reconcile by rebasing/cherry-picking onto `origin/main` in a controlled pass.
  - Do not reset local state or merge `origin/main` casually while untracked ZAYA runtime files exist.

## Corrections To Claude's Latest Status

- "Local main is clean" is not true from Codex's current checkout because `jang-tools/jang_tools/zaya/` has untracked source files.
- The vmlx ZAYA port is present only as untracked `jang-tools` files in this checkout; it is not yet a committed/public contract.
- Hy3 cannot be "ported from vmlx" because `../vmlx` has no Hy3 runtime to copy. It must be authored from the Hy3 handoff/spec.
- The bigger vmlx TurboQuant internals must not be copied wholesale into public `jang-tools` without a deliberate public/private boundary review.

## Required Next Gates

1. Decide whether `jang-tools/jang_tools/zaya/` should be kept, tested, and committed, or removed from this branch.
2. Reconcile `main` with `origin/main` before any public push.
3. Render Hy3 model card with the runtime-status matrix:
   - converter/bundle: complete
   - Python vmlx runtime: pending
   - Swift vmlx-swift-lm runtime: recognition gate only
   - MTP decode: preserved-disabled
4. Do not upload `OsaurusAI/Hy3-preview-JANGTQ2` until Eric explicitly accepts a preview/runtime-pending upload posture or real runtime smoke proof exists.

## Commands That Passed In This Audit

```sh
uv run --project jang-tools python -m jang_tools.verify_capabilities /Users/eric/models/JANGQ/Hy3-preview-JANGTQ2
uv run --project jang-tools python -m jang_tools.convert_hy3_jangtq /Users/eric/models/Tencent/Hy3-preview /tmp/hy3-dry-run-audit JANGTQ2 --dry-run
uv run --project jang-tools python -m py_compile jang-tools/jang_tools/zaya/__init__.py jang-tools/jang_tools/zaya/model.py jang-tools/jang_tools/zaya/runtime.py
python3 jang-tools/examples/hy3/python_runtime/hy3_jangtq_runtime_skeleton.py /Users/eric/models/JANGQ/Hy3-preview-JANGTQ2
swift jang-tools/examples/hy3/swift_runtime/Hy3JANGTQRuntimeSkeleton.swift
swift build --target MLXLLM
```

## Command That Is Blocked

```sh
swift test --filter Hy3RegistrationTests
```

Blocked by pre-existing package test build failure:

```text
Tests/MLXLMTests/BailingGLATests.swift:3:8: error: no such module 'XCTest'
```
