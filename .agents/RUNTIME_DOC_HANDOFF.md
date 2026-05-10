# Runtime Documentation Handoff

Status timestamp: 2026-05-09 local.

Private coordination only. `.agents/` is gitignored.

## Source-Of-Truth Rule

- `/Users/eric/vmlx-swift-lm` is read-only source material for this task.
- Do not edit `/Users/eric/vmlx-swift-lm` from this repo session.
- If code/docs/tests are needed, copy the minimum relevant pieces into the active repo and record original path + commit/status.
- That repo is currently dirty and in-progress, so treat uncommitted files as evidence that must be rechecked before any public claim.

## Current vmlx-swift-lm Evidence

Observed branch state:

- Current logged head sequence includes:
  - `b9da180 feat(runtime): harden osaurus integration checkpoint`
  - `88fc352 feat(runtime): harden hybrid cache model gates`
  - `aafd292 feat(zaya): full ZayaModel + factory dispatch (replaces unsupported throw)`
  - `0d4da12 feat(zaya): wire ZayaCCACache through BatchEngine admission + decode dispatch`
  - `b15a278 feat(zaya): BatchZayaCCACache per-slot CCA gather/scatter for batched decode`
  - `1d3d1a2 feat(cache): TQDiskSerializer + CacheHelpers wire ZayaCCACache disk round-trip`
  - `ae8821f feat(zaya): add ZayaCCACache hybrid KV+conv_state+prev_hs cache type`

Text-only ZAYA runtime exists:

- `Libraries/MLXLLM/LLMModelFactory.swift` dispatches `model_type=zaya` to `dispatchZaya`.
- `Libraries/MLXLLM/Models/Zaya.swift` is the text ZAYA model implementation.
- `Libraries/MLXLMCommon/Cache/ZayaCCACache.swift` keeps KV plus `conv_state` plus `prev_hs` together. It explicitly treats restoring KV without CCA state as a false cache hit.
- `Libraries/MLXLMCommon/BatchEngine/BatchZayaCCACache.swift` handles per-slot CCA state for batching.
- ZAYA tests include config decode, smoke, CCA state round-trip, disk round-trip, batch isolation, RMSNorm, and cache-helper coverage.
- `Tests/MLXLMTests/ZayaSmokeJANGTQ2Tests.swift` asserts generated text-only ZAYA bundles disable production thinking but keep tools and `zaya_xml`.

Qwen2.5-VL vision/runtime pieces exist:

- `Libraries/MLXVLM/Models/Qwen25VL.swift`
- `VLMModelFactory.swift` registers `qwen2_5_vl`.
- `VLMProcessorTypeRegistry` registers `Qwen2_5_VLProcessor`.
- `MEDIA-MODEL-MATRIX.md` documents Qwen2/Qwen2.5 VL as image/video capable with dense `KVCacheSimple` topology.

ZAYA1-VL source and ready-dispatch status:

- `Zyphra/ZAYA1-VL-8B` uses `model_type=zaya1_vl`, not `zaya`.
- `/Users/eric/models/Zyphra/ZAYA1-VL-8B` is present locally and was used for bundle generation.
- `VLMModelFactory.swift` has no `zaya1_vl` registration in the inspected state.
- `VLMModelFactory.swift` still has a `zaya` unsupported VLM dispatch note, but text-only `zaya` is implemented in the LLM factory. Do not read that old VLM note as current text-ZAYA status.
- ZAYA1-VL will need a new bridge: ZAYA CCA decoder/cache semantics plus Qwen2.5-VL image processor/vision tower handling plus the VL LoRA/projection policy.

## Runtime Status Matrix To Keep Updated

| Bundle family | Source model_type | Runtime status | Cache status | Quant status | Osaurus note |
|---|---|---|---|---|---|
| ZAYA1-8B text | `zaya` | Implemented in vmlx-swift-lm text LLM path | `ZayaCCACache` + `BatchZayaCCACache`; prefix/paged cache must not restore KV without CCA state | JANGTQ2/JANGTQ4/MXFP4 bundles exist locally; verify metadata mismatch before upload | Requires vmlx pin carrying ZAYA runtime and non-thinking metadata policy |
| ZAYA1-VL-8B | `zaya1_vl` | Not implemented as ready vmlx dispatch yet | Must combine image/media salt with ZAYA CCA state; no false KV-only restore | MXFP4/JANGTQ2/JANGTQ4 bundles generated locally at `/Users/eric/models/Zyphra/ZAYA1-VL-8B-*` | Do not list as supported until `zaya1_vl` dispatch + image proof exists |
| Qwen2.5-VL | `qwen2_5_vl` | Implemented VLM path | Dense `KVCacheSimple`; media salt applies | Existing Qwen path only; not ZAYA decoder | Use as vision-tower/processor reference, not as full ZAYA1-VL runtime |
| Qwen3.5/3.6-VL/Holo3 | `qwen3_5`, `qwen3_5_moe` | Implemented existing path | Hybrid family match needed where SSM/gated state exists | Existing JANGTQ/MXFP4 policy by family | Keep Osaurus hybrid matcher/cache notes current |
| Nemotron Omni | `nemotron_h_omni` | Implemented omni path | Mixed Mamba/KV/nil layers; media salt should include media | Existing JANGTQ/MXFP4 doc pattern | Good pattern for media model matrix, not ZAYA-specific |

## Documentation Required Before Public Upload

For each ZAYA1-VL bundle or Osaurus catalog entry:

- Exact source repo + revision.
- Exact `model_type` and architecture.
- Runtime support state: implemented, copied but unverified, or missing.
- Cache topology: KV, CCA state, media salt, prefix/paged-cache restrictions.
- Quant tier: MXFP4, JANGTQ2, JANGTQ4. No JANGTQ3.
- Sidecar requirement: `jangtq_runtime.safetensors` for JANGTQ.
- Processor sidecars: tokenizer, chat template, preprocessor/image processor config.
- Proof state: structure-only, forward pass, image+text generation, or batch/cache proof.
- Osaurus compatibility: required vmlx pin/commit and host-side guardrail.

## Osaurus-Side Notes Needed

Create/update an Osaurus-facing runtime-status matrix before exposing ZAYA1-VL to users:

- Text-only ZAYA works only with a vmlx pin that includes the ZAYA text runtime and cache fixes.
- ZAYA1-VL should be hidden or marked pending until `zaya1_vl` dispatch exists and a real image+text proof passes.
- If ZAYA1-VL bundles are surfaced early for users, the catalog row must explicitly state `runtime status: pending runtime`, `missing proof: no zaya1_vl runtime dispatch`, and `required commit: runtime path + media salt + image handling`.
- Catalog metadata must not reuse `zaya` for `zaya1_vl` unless the runtime adapter intentionally maps it.
- For models not yet carrying the fix, document expected failure mode clearly: unsupported model type, missing runtime dispatch, or no image path.
- Cache claims must distinguish:
  - dense VL (`KVCacheSimple`)
  - SSM/gated hybrid
  - SWA hybrid
  - ZAYA CCA hybrid (`KV + conv_state + prev_hs`)
  - media salt isolation

## Open Runtime Questions

- Where is the current Zyphra `zaya1-vl` Transformers/vLLM fork for image+text proof?
- Does ZAYA1-VL share enough tensor naming with text ZAYA to reuse expert pre-stacking directly, or does the VL LoRA branch require separate merge/passthrough treatment?
- Should production `zaya1_vl` capabilities remain non-thinking, or does the final runtime/chat-template implementation alter that?
- Which Osaurus/vmlx commit will be the first public pin that claims ZAYA1-VL support?
- Are the current local VL bundle READMEs acceptable to keep or should we replace with a temporary generated Osaurus contract card?

## Current Proof Snapshot (2026-05-09)

- Conversion status: `ZAYA1-VL-8B-MXFP4`, `ZAYA1-VL-8B-JANGTQ2`, `ZAYA1-VL-8B-JANGTQ4` are present in `/Users/eric/models/Zyphra`.
- Gate status (low-RAM):
  - `verify_directory`: all six target bundles return `OK`.
  - `model_type`: `zaya1_vl` for all three VL bundles.
  - `supports_thinking`: `false` for all six bundles (matches current product policy).
  - sidecar: present for both VL + text JANGTQ bundles, absent for MXFP4.
- VL image-token identity:
  - `image_token_id=262147`
  - `vision_start_token_id=255999`
  - `vision_end_token_id=256000`
- Current release posture:
  - Text-only ZAYA bundles can proceed to upload gating once the coherence artifact requirement is handled.
  - ZAYA1-VL bundles remain `runtime-pending` until a Swift/Python `zaya1_vl` dispatch and at least one image+text proof passes.
