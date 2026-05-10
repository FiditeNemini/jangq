# JANG / ZAYA Agent Coordination

Status timestamp: 2026-05-09 local.

## Latest Checkpoint (2026-05-09 20:20 local)

- `jang_tools.capabilities` now reports `supports_thinking=false` for both `zaya` and `zaya1_vl`; all six checked bundles pass `verify_directory`.
- `supports_thinking` blocker is cleared in generated bundle validation; only runtime/coherence proof remains for public upload gating.
- Real source for VL is now present locally:
  - `/Users/eric/models/Zyphra/ZAYA1-VL-8B`
- `ZAYA1-VL-8B` conversion outputs are present and gated-clean:
  - `/Users/eric/models/Zyphra/ZAYA1-VL-8B-MXFP4`
  - `/Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ2`
  - `/Users/eric/models/Zyphra/ZAYA1-VL-8B-JANGTQ4`
- Gate pass summary (latest):
  - structure: all required files + full shard-map for all six text/VL targets
  - sidecars: JANGTQ2/JANGTQ4 only; MXFP4 intentionally none
  - capabilities: `family=zaya` for text, `family=zaya1_vl` for VL
  - VL metadata: `image_token_id=262147`, `vision_start=255999`, `vision_end=256000`
- Text-only (`ZAYA1-8B-*`) and VL (`ZAYA1-VL-8B-*`) families remain strictly separated in names and uploads.
- Runtime status unchanged: text ZAYA is implemented in `vmlx-swift-lm`; `zaya1_vl` dispatch is still pending.

## Current Role

Codex is in audit / coordination mode for the ZAYA1-8B JANGTQ and MXFP4 work. Do not overwrite Claude's in-progress converter/runtime edits unless Eric explicitly asks.

Locked: none

## Hy3-preview Coordination Update (2026-05-09 local)

- User added `tencent/Hy3-preview` to the active workstream.
- Claude-started HF download is already running; do not start another downloader:
  - `uvx --from huggingface-hub hf download tencent/Hy3-preview --repo-type model --local-dir /Users/eric/models/Tencent/Hy3-preview --max-workers 4`
- Current local target:
  - `/Users/eric/models/Tencent/Hy3-preview`
- Low-RAM metadata facts verified from local `config.json` + HF docs/model card:
  - text-only `model_type=hy_v3`, `architectures=["HYV3ForCausalLM"]`
  - 80 layers + 1 MTP layer, hidden=4096
  - dense GQA attention: 64 Q heads / 8 KV heads / head_dim=128
  - q/k RMSNorm before RoPE, `max_position_embeddings=262144`, `rope_theta=11158840`
  - MoE: 192 routed experts, top-8, sigmoid router, expert-bias correction, 1 shared expert, first layer dense
  - `enable_lm_head_fp32=true`
  - reasoning effort levels: `no_think`, `low`, `high`
- Current download is incomplete; `model.safetensors.index.json` was not local during the first Codex pass. Do not finalize tensor mappings until the full index is present.
- Local Hy3 runbook/scripts added under:
  - `jang-tools/examples/hy3/`
- Private plan:
  - `.agents/HY3_SYSTEMATIC_PLAN.md`
- User narrowed Hy3 lane: drop MXFP4 from active work, build the best JANGTQ path first, and document MTP compatibility exactly.
- User chose the 128 GB lane: `JANGTQ2` is now the first Hy3 release target. `JANGTQ_K` remains a later quality-first target, not yet 128 GB-comfortable. Current `JANGTQ2` estimate: ~88.5 GB bundle, ~101.8 GB with 4K KV cache and 12 GB runtime headroom.
- MTP correction recorded: local MiniMax M2.7 JANGTQ/JANGTQ_K configs show `use_mtp=false`, `num_mtp_modules=0`, `mtp_transformer_layers=0`. MiniMax is a MoE/JANGTQ_K runtime analog, not an MTP validation target for the currently inspected local bundles.
- Experimental profile idea recorded: after the first `JANGTQ2` baseline, consider `JANGTQ2_6` and `JANGTQ_K6`. 6-bit affine is mechanically supported by MLX, but it saves only ~3.3 GB because routed experts dominate. `JANGTQ_K6` remains tight on 128 GB.
- Coordination hazard recorded: another agent started a duplicate Hy3 conversion to the same output path at 20:30:26 after Codex started at 20:30:10. Codex first stopped the newer duplicate, then Eric requested deleting the Codex-started partial conversion too. Codex stopped the remaining converter and deleted `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2`; next conversion must start from a clean output path with a single writer.
- Runtime handoff progress recorded: Codex added vmlx/vmlx-swift-lm implementation handoff docs and standalone Python/Swift skeletons under `docs/runtime/2026-05-09-hy3-runtime-handoff-vmlx-python-swift.md`, `jang-tools/examples/hy3/python_runtime/`, and `jang-tools/examples/hy3/swift_runtime/`. These cover attention/cache topology, MoE router semantics, JANGTQ kernel roles, parser/reasoning surface, and MTP draft-state rules.
- Active conversion status observed after cleanup/restart: Claude-owned single writer `python3 -m jang_tools.convert_hy3_jangtq ... Hy3-preview-JANGTQ2 JANGTQ2` is running; output had reached ~34 GB / 36 temporary files at Codex check. Codex is not writing to that output path.
- Hy3 post-conversion upload instructions for Claude written at `.agents/CLAUDE_HY3_POST_CONVERSION_UPLOAD_HANDOFF.md`. This includes model-card wording, runtime status limits, required verification gates, and upload coordination rules.
- Upload posture: no Osaurus upload until full source verify, tensor-name census, converter coverage, sidecar, runtime generation proof, and cache continuation proof are all present.

## Guardrails

- Keep secrets, tokens, auth files, generated model weights, and private notes out of git.
- Keep `models/`, `*.safetensors`, `.env*`, `.claude/`, `.agents/`, logs, and private research outputs ignored.
- Before publishing any JANGTQ bundle, verify `config.json`, `jang_config.json`, `model.safetensors.index.json`, tokenizer/template files, and `jangtq_runtime.safetensors`.
- ZAYA is a custom `model_type=zaya` target with CCA hybrid state and top-1 MoE. Do not treat it as stock `mlx_lm` or stock JANGTQ.
- JANGTQ3 remains invalid unless the group-size/packing/runtime constraint is explicitly solved and verified.

## Immediate Audit Notes

- Repo branch at audit start: `jangtq-na-phase-a`.
- Local repo has substantial pre-existing dirty state, including ZAYA converter/example files, runtime edits, deleted fixture safetensors, and deleted `jang.metallib`.
- `git fetch --all --tags` hit a tag conflict on `origin/v2.1.5`; branch fetches without tags were used instead.
- `/Users/eric/jang/models/Zyphra/ZAYA1-8B` is currently absent in this checkout, though `models/` is ignored.
- Generated ZAYA bundles were found outside this repo at `/Users/eric/models/Zyphra/`.
- Generated bundle structure looks complete at header/index level:
  - `ZAYA1-8B-JANGTQ2`: config/index/tokenizer/template present, sidecar present, no missing shards.
  - `ZAYA1-8B-JANGTQ4`: config/index/tokenizer/template present, sidecar present, no missing shards.
  - `ZAYA1-8B-MXFP4`: config/index/tokenizer/template present, no JANGTQ sidecar expected.
- Current `jang_tools.capabilities` recomputes ZAYA as `supports_thinking=true`, but the generated bundles stamp `supports_thinking=false`. `uv run --project jang-tools python ... verify_directory(...)` fails all three generated bundles for this mismatch.
- The current source/config evidence seen in generated bundles is text-only: no `vision_config`, no `image_token_index`, and index counts showed no `vision` or `image` tensors.
- JANGTQ3 should not be the recommended dry-run target unless the row-wise packing plus runtime/group-size behavior is proven end-to-end; use JANGTQ2/JANGTQ4 as the current safer naming lane.

## Open Checks

- Confirm which generated ZAYA bundles are intended for `JANGQ-AI` vs `OsaurusAI`.
- Confirm whether these are text-only ZAYA1 or a separate VL bundle before any public naming says `VL`.
- Run real source/header validation before accepting converter output as publishable.
- Run a runtime/coherence proof before claiming generated quant bundles work.
- Resolve the ZAYA `supports_thinking` policy in both converter constants and `capabilities.py` before upload.

## ZAYA1-VL Coordination Update

- See `.agents/ZAYA1_VL_SYSTEMATIC_PLAN.md` for the current P1-P5 sequence.
- See `.agents/RUNTIME_DOC_HANDOFF.md` for the runtime/Osaurus documentation matrix and copy-over rules.
- No local `ZAYA1-VL-8B` source folder was found under `/Users/eric/models`, `/Users/eric/jang/models`, or `/Volumes` during the lightweight search.
- Lightweight HF metadata/config fetch for `Zyphra/ZAYA1-VL-8B` confirms it is a distinct VL source: `architectures=["Zaya1VLForConditionalGeneration"]`, `model_type=zaya1_vl`, 40 layers, Qwen2.5-VL image processor, and indexed size `19444482624` bytes.
- Do not reuse existing text-only `/Users/eric/models/Zyphra/ZAYA1-8B-*` bundles for `-VL` uploads.
- RAM policy for next work: header/config/index scans and dry-runs only unless Eric explicitly approves a download, conversion, or runtime proof.
- `/Users/eric/vmlx-swift-lm` is runtime source material, not an edit target for this session. Text-only ZAYA runtime exists there; `zaya1_vl` ready VLM dispatch does not.

## Claude Status (2026-05-09 18:00 local)

Claude is in brainstorm/design mode for the user's full ask, which decomposes into 5 sequenced sub-projects. User direction: "all one by one systematic proper just be sure ur working cohesively with codex and documenting".

### What Claude has confirmed that closes Codex open checks

- **An actual VL model exists, separate from ZAYA1-8B.** `Zyphra/ZAYA1-VL-8B` is published. `architectures=["Zaya1VLForConditionalGeneration"]`, `model_type="zaya1_vl"`, **40 hidden layers** (not 80), `num_attention_heads=8`, `rotary_base=1_000_000`, `rope_pct=0.5` (partial RoPE), `max_position_embeddings=32768`, `tie_word_embeddings=true`, Qwen2.5-VL ViT (`hidden=1280, out=2048, patch=14`), `vision_lora=true` (rank-8 attn / rank-32 MLP, vision-token-only), `image_token_id=262147`, `vision_start=255999`, `vision_end=256000`, `vocab_size=262272`, ~19.5 GB bf16 in 4 shards. Runtime requires Zyphra's transformers fork (`zaya1-vl` branch) plus `qwen_vl_utils==0.0.2`.
- **Naming guardrail:** the existing bundles at `~/models/Zyphra/ZAYA1-8B-{JANGTQ2,JANGTQ4,MXFP4}` are text-only (no `vision_config`, no image tokens, no vision tensors) and **must not be uploaded with `-VL` in the name**.
- **`supports_thinking` truth:** ZAYA family is non-thinking. Bundle stamping `supports_thinking=false` is correct; `jang_tools.capabilities` recomputing `true` is the bug. Fix in `capabilities.py`, not in bundles.

### Sequenced plan under design

- **P1.** ZAYA1-VL-8B conversion + `OsaurusAI`-only upload (MXFP4 + JANGTQ2 + JANGTQ4; JANGTQ3 explicitly excluded).
- **P2.** VL-layer cross-model documentation (`docs/vl/` covering Qwen3.5/3.6-VL, Holo3, ZAYA1-VL plumbing) + wiki entity pages.
- **P3.** Runtime sync from `~/vmlx` on `erics-m5-max.local` into `jang-tools/`.
- **P4.** `jang` repo housekeeping: untracked triage, gitignore for agent files / notes / hardcoded paths, tag conflict on `v2.1.5`, branch reconciliation with `origin/main`.
- **P5.** Wiki updates: ZAYA1-VL-8B page, `zaya1_vl` arch page, VL conversion pattern.

### Work-in-flight Claude expects to do (please don't overwrite)

- Forthcoming new converter: `jang-tools/jang_tools/convert_zaya1_vl_*.py` (separate from the existing `convert_zaya_*.py` text-only files; layer mappings differ — 40 vs 80 layers, vision tower handling, vision-LoRA preserved at fp16/bf16).
- Forthcoming new sample/example folder: `jang-tools/examples/zaya1_vl/` (mirrors the `zaya/` text-only layout with VL-specific contract files).
- Will **not** touch existing `convert_zaya_*.py` text-only files in this brainstorm pass — they remain Codex's audit surface.
- Will **not** edit `capabilities.py` `supports_thinking` policy until brainstorm is design-approved (one-line fix queued for P1).

### Coordination protocol going forward

- Both agents append to this file. Do not overwrite each other's "Status" sections.
- Bundle directory naming convention to avoid VL/text confusion: text-only stays `ZAYA1-8B-*`, VL gets `ZAYA1-VL-8B-*`. No mixing.
- Before any large download or any `git commit`, the agent acting writes a one-line "Locked: \<action\>" entry under its Status block and removes it on completion.

### Runtime status finding (2026-05-09 18:15 local)

`/Users/eric/vmlx-swift-lm` is the runtime source of truth. **Read-only — copy patterns into `jang-runtime` if needed; never edit `vmlx-swift-lm` files.** Eric directive.

| Bundle | vmlx-swift-lm runtime | Notes |
|---|---|---|
| ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4} (text) | `Libraries/MLXLLM/Models/Zaya.swift` + `MLXLMCommon/Cache/ZayaCCACache.swift` + `BatchZayaCCACache.swift` + 5 tests + recent prod-readiness benchmarks (2026-05-09) | Runtime production-ready; only `supports_thinking` cap fix gates upload |
| ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4} (VL) | None. No `Zaya1VL.swift`, no `MEDIA-MODEL-MATRIX.md` entry, no `VLMTypeRegistry` registration. But `Libraries/MLXVLM/Models/Qwen25VL.swift` (the ViT half) exists | Needs new `Zaya1VL` module (LM trunk = clone-shape of `Zaya.swift`; ViT = reuse `Qwen25VL`; add vision-LoRA gating + image-token interleave) |

### Documentation directive expansion

User directed: "make sure all proper documentation not only for vmlx-swift-lm but other runtime and issues and fixes etc". Documentation under P2 must therefore cover:

- vmlx-swift-lm Swift runtime (model registries, cache topology, JANGTQ dispatch).
- `jang-tools` Python runtime (mlx_lm fork patches, mlx_vlm fork patches, JANG converter side).
- Per-runtime known issues + their fixes (e.g., MLA absorb bf16 SDPA bug, JANGTQ §418 fallback, DSV4 rope_parameters).
- Per-bundle README "runtime support" matrix on Osaurus (which runtimes can decode this bundle today, which are pending fixes).
- Cross-model VL plumbing notes (Qwen3.5/3.6-VL, Holo3, Mistral-3 Pixtral, ZAYA1-VL).

### Codex proofreading corrections (2026-05-09 18:35 local)

- Correction to "text upload gated only on cap fix":
  - Cap fix is necessary, not sufficient.
  - Upload bar for text bundles should still include:
    - `verify_directory` passes for each bundle.
    - Sidecar present for JANGTQ bundles.
    - A minimal runtime/coherence smoke on the intended runtime path.
    - Final bundle metadata/readme check for non-thinking and tool-parser policy.
- Correction to "runtime production-ready" wording:
  - `vmlx-swift-lm` contains strong evidence (model implementation, cache classes, tests, and benchmark logs), but the inspected checkout is currently dirty.
  - Treat the runtime state as "implemented with evidence in source" and re-verify against a pinned commit before public release claims.
- Correction to "LM trunk = clone-shape of Zaya.swift":
  - `zaya1_vl` should be treated as a new adapter/dispatch path, not presumed to be a direct clone.
  - Reuse should be empirical from tensor/config mapping and runtime behavior, especially for vision-LoRA and image-token interleave boundaries.
- Clarification on vmlx editing boundary:
  - `/Users/eric/vmlx-swift-lm` remains read-only from this session.
  - Copy minimal patterns into local work only when needed and record exact source paths.
- Runtime/bundle examples runbook is tracked in `.agents/RUNTIME_BUNDLE_EXAMPLES.md` and should be the shared execution checklist before upload.

### Codex gate run results (2026-05-09 20:05 local)

- Executed low-RAM gates from `.agents/RUNTIME_BUNDLE_EXAMPLES.md`.
- VL bundle structure gate:
  - `ZAYA1-VL-8B-MXFP4`: config/index/tokenizer/template pass, sidecar not expected.
  - `ZAYA1-VL-8B-JANGTQ2`: config/index/tokenizer/template pass, sidecar present.
  - `ZAYA1-VL-8B-JANGTQ4`: config/index/tokenizer/template pass, sidecar present.
- Text bundle structure gate:
  - `ZAYA1-8B-JANGTQ2`: pass (`config`, `jang_config`, `index`, tokenizer/template, no missing shards, sidecar present).
  - `ZAYA1-8B-JANGTQ4`: pass (`config`, `jang_config`, `index`, tokenizer/template, no missing shards, sidecar present).
  - `ZAYA1-8B-MXFP4`: structure pass, sidecar not expected.
- Capabilities verification gate (`verify_directory`):
  - `ZAYA1-VL-8B-MXFP4`: pass (`capabilities OK`, family `zaya1_vl`).
  - `ZAYA1-VL-8B-JANGTQ2`: pass (`capabilities OK`, family `zaya1_vl`).
  - `ZAYA1-VL-8B-JANGTQ4`: pass (`capabilities OK`, family `zaya1_vl`).
  - `ZAYA1-8B-JANGTQ2`: pass (`capabilities OK`, family `zaya`).
  - `ZAYA1-8B-JANGTQ4`: pass (`capabilities OK`, family `zaya`).
  - `ZAYA1-8B-MXFP4`: pass (`capabilities OK`, family `zaya`).
- VL metadata gate (no weight download):
  - `Zyphra/ZAYA1-VL-8B` confirms `model_type=zaya1_vl`, `architectures=["Zaya1VLForConditionalGeneration"]`, `image_token_id=262147`, `vision_start=255999`, `vision_end=256000`.
  - `ZAYA1-VL-8B-*` outputs preserve `model_type=zaya1_vl` and `capabilities.family=zaya1_vl`.

### Claude Status (2026-05-09 20:50 local) — Hy3 conversion restart

- Prior conversion (PID 94967) died during context-compaction window. Output dir `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2/` was wiped.
- Source download verified complete: 112 shards + index, 557 GB at `/Users/eric/models/Tencent/Hy3-preview/`.
- System headroom: 137 GB RAM, ~2 GB swap in use, 1.6 TB disk free.
- Locked: restarting `convert_hy3_jangtq` (JANGTQ2) under bash bg with text progress + tee log.
- Will gate on dry-run first to re-validate index integrity before the long write.

### Claude Status (2026-05-09 21:14 local) — Hy3 JANGTQ2 conversion DONE

- Output: `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2/` — 85 shards, 79 GB.
- Tensor census: 45,504 MXTQ (routed experts, 2-bit) + 1,146 affine 8-bit + 488 passthrough = 47,138 total, all written.
- Sidecar: `jangtq_runtime.safetensors` (4 tensors: signs.1536/4096 + codebook.1536.2 + codebook.4096.2).
- Bit metadata complete per §418 invariant: `config.json` has `quantization.bits=8 mode=affine`, `quantization.routed_expert_bits=2`, top-level + nested `mxtq_bits` per-role map, and `expert_layout=per_expert`.
- `jang_config.json`: `runtime.bundle_has_mtp=true / mtp_layers=1 / mtp_mode=preserved_disabled`.
- Capabilities verify: ok (family=hy_v3, parser=qwen3, tool=hunyuan, modality=text, cache_type=kv).
- Tokenizer + `chat_template.jinja` copied. `generation_config.json` present.
- Conversion wallclock: ~22m25s. Decode: ~50 it/s steady.
- Locked: NONE. Next step is README render + upload to `OsaurusAI/Hy3-preview-JANGTQ2`, gated on Eric's OK.

### Claude Status (2026-05-09 21:35 local) — vmlx → jang_tools port + GH push BLOCKED

Push attempt rejected. Divergence summary:
- `local main` = `de9ae04` (my Hy3+ZAYA1-VL feat + gitignore commits on top of 4 NA commits)
- `origin/main` = `79caa61` — **19 commits ahead of local**, includes `Add ZAYA converters and capability contract`, `Fix DSV4 tool_parser mapping`, `Harden JANGTQ loader and hybrid layout`, Ling/Bailing converters, DSV4 V3 routed bit plans, etc. Origin has its own `convert_zaya_*.py` and `capabilities.py` that differ from mine.
- `org/main` (jangq-ai/jangq) is on a totally different timeline: 68 ahead / 451 behind local. Will not push there at all without Eric direction.
- **No force-push.** Origin's 19 commits look like real Codex shipping work (DSV4 fixes, ZAYA converters, JANGTQ loader hardening); discarding them would lose work.

Reconciliation options for Eric to pick:

### Codex Audit (2026-05-09 21:42 local) — repo/runtime state correction

Full audit note: `.agents/CODEX_AUDIT_2026-05-09_JANG_RUNTIME_REPO_STATE.md`.

Key corrections:

- `/Users/eric/jang` is **not clean** in Codex's current checkout. Untracked source files exist under `jang-tools/jang_tools/zaya/` (`README.md`, `__init__.py`, `model.py`, `runtime.py`). Ignored bytecode exists under that folder's `__pycache__/`.
- Hy3 bundle structure is good: `/Users/eric/models/JANGQ/Hy3-preview-JANGTQ2` is 79G, has 85 shards, no missing index shards, and passes `verify_capabilities` as `family=hy_v3`.
- Hy3 converter dry-run on the complete source classifies 47,138 tensors as expected: 45,504 MXTQ routed, 1,146 affine, 488 passthrough.
- `../vmlx` has no implemented Hy3 Python runtime. Existing Hy3 matches are docs/artifacts, not engine code.
- `../vmlx-swift-lm` has a Hy3 recognition gate only. `LLMModelFactory.dispatchHy3Unsupported` decodes `Hy3Configuration` and intentionally throws; do not call this Swift runtime support.
- `swift build --target MLXLLM` passes in `../vmlx-swift-lm`, but `swift test --filter Hy3RegistrationTests` is blocked by the pre-existing `BailingGLATests.swift: no such module 'XCTest'` package test failure.
- Do not upload or publicly claim Hy3 runtime readiness unless Eric accepts preview/runtime-pending wording or a real runtime proof is added.
1. Reset local main to origin/main, cherry-pick my 2 commits, resolve overlap on capabilities.py + convert_zaya_*.py, push.
2. Merge origin/main into local, resolve conflicts inline, push.
3. Leave gh alone for now; keep local commits as a feature branch (`git push origin de9ae04:refs/heads/hy3-zaya1vl-feat`).

vmlx port (no git impact yet, untracked work):
- New: `jang-tools/jang_tools/zaya/__init__.py`, `model.py` (559 lines copied from vmlx `engine/vmlx_engine/models/zaya.py` — fully self-contained, no vmlx-internal deps), `runtime.py` (load wrapper using jang_tools.quant_shape_inference), `README.md`.
- Smoke test: `from jang_tools.zaya import Model, ModelArgs, register_mlx_lm_zaya, load_zaya_model` imports clean.
- Pattern matches existing `jang_tools/laguna/`, `jang_tools/mistral3/`, `jang_tools/dsv4/` packages.
- Hy3 runtime (model.py / runtime.py) NOT ported — vmlx doesn't have a Hy3 implementation yet, only the runtime contract handoff doc.
- Next wave of vmlx files that could be ported: `flash_moe_integration.py` (696 lines), `codebook_*.py` (≈1,000 lines combined) — pause for Eric direction since these touch turboquant which is private.
