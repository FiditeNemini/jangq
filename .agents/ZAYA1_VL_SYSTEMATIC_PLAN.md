# ZAYA1-VL-8B Systematic Coordination Plan

Status timestamp: 2026-05-09 local.

This is a private agent coordination file. It is intentionally covered by `.gitignore`.

## Operating Policy

- Work one phase at a time. P1 blocks upload. P2-P5 are follow-up lanes unless Eric explicitly reorders.
- Keep RAM use conservative. Default to header/config/index scans, dry-runs, and file-count checks. Do not run full model loads, conversion, coherence generation, or multi-worker download without an explicit go-ahead.
- Use this local M5 Max MacBook only. Do not use retired Macs or remote machines unless Eric explicitly approves for the current task.
- Do not publish anything with `-VL` naming unless the source is verified as `Zyphra/ZAYA1-VL-8B` or a generated bundle derived from it.
- Do not publish ZAYA1 text-only bundles as VL bundles.
- Keep all token/auth handling in environment/keychain/CLI state. Do not write token values into repo files, wiki pages, `.agents`, commands, or responses.

## Verified Metadata Snapshot

Source repo: `Zyphra/ZAYA1-VL-8B`

Observed config facts:

- `architectures`: `["Zaya1VLForConditionalGeneration"]`
- `model_type`: `zaya1_vl` (text-only `ZAYA1-8B` is `zaya`, never co-mingled)
- `num_hidden_layers`: `40`
- `num_attention_heads`: `8`
- `num_key_value_heads`: `2`
- `hidden_size`: `2048`
- `rotary_base`: `1000000`
- `rope_pct`: `0.5`
- `max_position_embeddings`: `32768`
- `tie_word_embeddings`: `true`
- `image_token_id`: `262147`
- `vision_start_token_id`: `255999`
- `vision_end_token_id`: `256000`
- `vision_config.model_type`: `qwen2_5_vl`
- `vision_config.hidden_size`: `1280`
- `vision_config.out_hidden_size`: `2048`
- `vision_config.spatial_patch_size`: `14`
- `vision_config.temporal_patch_size`: `1`
- `preprocessor_config.image_processor_type`: `Qwen2VLImageProcessor`
- `preprocessor_config.processor_class`: `Zaya1VLProcessor`
- `model.safetensors.index.json.metadata.total_size`: `19444482624`
- Weight-map counts from the index: `5833` entries, `390` containing `vision`, `2960` containing `lora`, `5437` under `model.layers`, `3840` `local_experts`, `399` `router`.
- Hub file list has no custom `.py` modeling/processor files; the required Zyphra runtime/fork is external to the model repo and must be identified separately before coherence proof.

Important contrast:

- Existing generated local bundles under `/Users/eric/models/Zyphra/ZAYA1-8B-*` are text-only `model_type=zaya` bundles. They have no `vision_config`, no `image_token_index`, and no vision/image tensors in their indexes.
- Existing generated VL bundles at:
  - `/Users/eric/models/Zyphra/ZAYA1-VL-8B-MXFP4`
  - `/Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ2`
  - `/Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ4`
  are now present and pass local structure/capability gates.
- Existing `convert_zaya1_vl_*.py` are the VL-specific implementations; they are not interchangeable with `convert_zaya_*`.

## P1: ZAYA1-VL-8B Conversion And Upload

Goal: produce OsaurusAI-only MXFP4, JANGTQ2, and JANGTQ4 bundles derived from the real VL source, with coherence/runtime evidence.

Order:

1. Confirm source availability and storage path.
   - First search local paths and mounted drives.
   - If absent, plan a low-worker `hf download` into an ignored model path.
   - Do not start the 19.44 GB download while other heavy jobs are running.
2. Write a VL-specific converter rather than mutating the text converter blindly.
   - Reuse text path patterns: sidecar copy, `jang_config`, tokenizer/template handling, sidecar build, pre-stacked experts.
   - Add VL-specific handling for Qwen2.5-VL vision tower, image processor files, image tokens, and LoRA-gated vision weights.
   - Preserve exact tensor-name mapping from `model.safetensors.index.json`.
3. Keep precision floors explicit.
   - Router, norms, CCA/hybrid state, vision processor-sensitive tensors, and LoRA controls should stay passthrough until proven safe.
   - JANGTQ routed experts should be JANGTQ2 or JANGTQ4 only.
   - Do not use JANGTQ3 unless the packing/group-size/runtime constraint is solved and independently verified.
4. Verify bundle structure before runtime proof.
   - `config.json`, `jang_config.json`, `model.safetensors.index.json`
   - tokenizer/template files
   - preprocessor/image processor files
   - no missing shards
   - no text-only metadata leakage
   - JANGTQ bundles include `jangtq_runtime.safetensors`
5. Run coherence/runtime proof only after structure passes.
   - Use the smallest image+text prompt proof that exercises image preprocessing, vision tokens, language decode, and cache initialization.
   - Capture output to a durable report file.
   - Avoid long-context or batch stress until the simple proof passes.
6. Upload only after proof and publication hygiene pass.
   - Target: OsaurusAI only, unless Eric explicitly changes it.
   - No AI attribution.
   - No tokens in command logs or docs.

P1 blockers as of this note:

- `supports_thinking=False` policy is now aligned in converter/runtime metadata for all generated ZAYA1-VL bundles.
- `jang_tools.capabilities` no longer blocks verify for these generated bundles.
- A real VL coherence path is still required:
  - Swift `zaya1_vl` dispatch + media/cache proof path must be implemented and run.
  - The custom Zyphra runtime/fork for full VLM behavior still needs to be identified for proof.
- Runtime/source blocker resolved:
  - local source folder `/Users/eric/models/Zyphra/ZAYA1-VL-8B` is present and has been used for conversion.

## P2: VL Cross-Model Documentation

Goal: document reusable VL plumbing across Qwen3.5/3.6-VL, Holo3, ZAYA1-VL, and future families.

Minimum outputs:

- Private draft under `.agents` first.
- Public docs only after facts are source-verified.
- Wiki page for `zaya1_vl` architecture after P1 metadata and proof are stable.
- Runtime-status matrix for Osaurus-facing docs. See `.agents/RUNTIME_DOC_HANDOFF.md`.
- Cache-aware example checklist for both agents is in `.agents/RUNTIME_BUNDLE_EXAMPLES.md`.

Topics to cover:

- image token insertion
- processor/image processor sidecars
- vision tower tensor policy
- LoRA / projector / merge policy
- cache boundary between vision prefill and language decode
- quantization precision floors

## P3: Runtime Sync From vmlx

Goal: compare latest relevant runtime/converter/script changes from `/Users/eric/vmlx` and active Swift/runtime repos into `jang-tools` where appropriate.

Constraints:

- Treat `/Users/eric/vmlx-swift-lm` as read-only source material. Copy over only if needed; do not edit there from this session.
- Read-only compare first.
- Do not wholesale copy code.
- Prefer minimal, source-backed patches.
- Track exact source commit/path for any borrowed behavior.

## P4: Jang Repo Housekeeping

Goal: make the local and GitHub-facing repo clean without destroying ongoing work.

Current known state:

- Many pre-existing modified/deleted/untracked files.
- Some tracked binary fixtures are currently deleted.
- `.agents/` is ignored.
- Text-only generated bundles live outside repo under `/Users/eric/models/Zyphra`.

Process:

1. Inventory by category: publishable code, private research, generated outputs, stale fixkits, tracked binary deletions.
2. Do not delete or revert user/Claude work without explicit approval.
3. Make ignore changes only for clearly local/private/generated artifacts.
4. Before any commit, check author/committer identity and staged diff for secrets/private docs/AI attribution.
5. Keep `.agents` runbooks as private coordination artifacts until promoted into reviewed public docs.

## P5: Wiki Updates

Goal: durable cross-project memory without secrets.

Already written:

- `notes/2026-05-09-zaya1-bundle-audit-capability-mismatch-2026-05-09.md`

Needed after P1/P2:

- `ZAYA1-VL-8B` model page.
- `zaya1_vl` architecture page.
- VL conversion pattern page.
- Updated JANG/JANGTQ page if ZAYA1-VL establishes a reusable VL quantization rule.
