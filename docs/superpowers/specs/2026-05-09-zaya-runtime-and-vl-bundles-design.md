# ZAYA runtime in JANG + ZAYA1-VL-8B bundles + Osaurus upload — design

Status: DRAFT. Awaiting Eric review.
Created: 2026-05-09.


---

## 1. Goal

Land production-ready ZAYA family inference inside `/Users/eric/jang` and ship `OsaurusAI` HuggingFace bundles for both text and VL flavors. Specifically:

1. ZAYA text-only Swift + Python runtime lives in `jang-runtime` and `jang-tools`, copied/derived from `~/vmlx-swift-lm` and patterns from `~/vmlx`. Both Swift and Python paths exercise the full caching contract (CCA hybrid: KV + `conv_state` + `prev_hs`; disk round-trip; batched per-slot isolation; no false KV-only restore).
2. ZAYA1-VL Swift + Python runtime is built fresh in `jang-runtime` and `jang-tools` by combining the ZAYA decoder with the existing Qwen2.5-VL ViT plumbing, plus vision-LoRA gating, image-token interleave, and a `Zaya1VLCache` that adds an image-media salt to the existing CCA hybrid invariant.
3. Three text-only bundles already at `~/models/Zyphra/ZAYA1-8B-*` get a capability fix (`supports_thinking=False`), pass the verifier, and upload to `OsaurusAI` only.
4. Three new VL bundles `ZAYA1-VL-8B-{MXFP4, JANGTQ2, JANGTQ4}` get converted, structurally verified, runtime-proven (image+text decode, cache hit + miss, batch isolation), and uploaded to `OsaurusAI` only.
5. Documentation across runtimes (Swift in `vmlx-swift-lm` and `jang-runtime`; Python in `~/vmlx/engine/vmlx_engine` and `jang-tools`) plus per-bundle Osaurus README runtime-status matrices.
6. `jang` repo housekeeping: untracked triage, tightened `.gitignore`, tag conflict resolution, branch reconciliation with `origin/main`.
7. Wiki entries for `Zyphra/ZAYA1-VL-8B`, the `zaya1_vl` architecture, and the cross-model VL conversion pattern.

Non-goals:

- JANGTQ3 for ZAYA. Excluded until row-wise packing / group-size / runtime constraint is independently verified.
- `JANGQ-AI` upload mirror. Per Eric: `OsaurusAI` only for this batch.
- Editing files inside `~/vmlx-swift-lm` or `~/vmlx`. Both are read-only sources for this work.

---

## 2. Background facts (verified 2026-05-09)

### 2.1 ZAYA1-8B (text-only)

`Zyphra/ZAYA1-8B`. `architectures=["ZayaForCausalLM"]`, `model_type=zaya`. 80 hidden layers (alternating ATT + MoE-MLP), 16 attn heads, 2 KV heads, `head_dim=128`, 16 experts top-1 routed, `zaya_mlp_expansion=256`, `rope_theta=5_000_000`, `max_position_embeddings=131_072`. CCA attention has standard KV plus inner state (`conv_state`, `prev_hs`). Official vLLM disables prefix caching. Non-thinking in production policy.

### 2.2 ZAYA1-VL-8B (VL)

`Zyphra/ZAYA1-VL-8B`. `architectures=["Zaya1VLForConditionalGeneration"]`, `model_type=zaya1_vl`. **40 hidden layers**, 8 attn heads, 2 KV heads, `head_dim=128`, 16 experts top-1 routed, `rope_pct=0.5` (partial RoPE), `rotary_base=1_000_000`, `max_position_embeddings=32_768`, `tie_word_embeddings=true`. Vision tower is Qwen2.5-VL ViT (`hidden=1280`, `out=2048`, `spatial_patch=14`, `temporal_patch=1`). Vision-LoRA on the LM trunk: `rank-8` attn / `rank-32` MLP, gated to activate only on vision tokens. Special tokens: `image_token=262147`, `vision_start=255999`, `vision_end=256000`. `vocab_size=262_272`. `~19.5 GB` bf16 across 4 shards. Index has `5833` total tensors: `390 vision`, `2960 lora`, `5437 layers`, `3840 local_experts`, `399 router`. Source runtime requires `transformers @ git+https://github.com/Zyphra/transformers.git@zaya1-vl` + `qwen_vl_utils==0.0.2`.

### 2.3 Existing local state

| Path | State |
|---|---|
| `~/models/Zyphra/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}` | Built 2026-05-06. Header/index/sidecar checks pass. Verifier fails on `supports_thinking` mismatch. |
| `~/models/Zyphra/zaya_upload_manifest.json` | Lists `JANGQ-AI` and `OsaurusAI` repos for the text bundles. `JANGQ-AI` half is to be dropped per current direction. |
| `~/models/Zyphra/zaya_coherence_report.json` | All `passed=false` due to connection-refused. Treated as no evidence. |
| `~/models/Zyphra/ZAYA1-VL-8B/` | Does not exist. Needs `hf download`. |
| `jang-tools/jang_tools/convert_zaya_{common,jangtq,mxfp4}.py` | Untracked. Targets text-only ZAYA1-8B. Pattern source for the VL converter. |
| `jang-tools/examples/zaya/` | Untracked. Text-only contract files (`00_inspect_source.py` … `04_prepare_hf_uploads.py`, `ATTENTION_ARCHITECTURE.md`, `ZayaRuntimeContract.swift`). |
| `jang-runtime/Sources/JANG/` | No ZAYA Swift code today. New `Zaya/` and `Zaya1VL/` submodules to be added here. |

### 2.4 Runtime source-of-truth

| Source | Purpose | Rule |
|---|---|---|
| `~/vmlx-swift-lm` | Swift LLM/VLM runtime. **State: "implemented with evidence in source"** — `Libraries/MLXLLM/Models/Zaya.swift`, `MLXLMCommon/Cache/ZayaCCACache.swift`, `BatchEngine/BatchZayaCCACache.swift`, 5 tests, prod-readiness benchmarks 2026-05-09, plus Qwen2.5-VL ViT (`Libraries/MLXVLM/Models/Qwen25VL.swift`) all present. Inspected checkout is dirty; treat as evidence pending **pinned-commit re-verification** before any public release claim. No `Zaya1VL` module. | Read-only. Copy patterns into `jang-runtime`; record original commit + path of each copy in `PROVENANCE.md`. Never edit. |
| `~/vmlx/engine/vmlx_engine` | Python engine: `attention.py`, `cache/`, `disk_cache.py`, `memory_cache.py`, `mllm_cache.py`, `mllm_batch_generator.py`, `loaders/`, `metal/`. | Read-only. Pattern source for `jang-tools` Python ZAYA runtime. |

### 2.5 Coordination

Both agents append `### <Agent> Status` sections; never overwrite the other's section. Before any large download or `git commit`, the acting agent writes a one-line `Locked: <action>` entry under its Status block, removed on completion. audit agent's existing guardrails are inherited:

- ZAYA family is non-thinking in production. Bundles stamping `supports_thinking=false` are correct; `jang_tools.capabilities` recomputing `true` is the bug.
- JANGTQ3 excluded for ZAYA.
- `models/`, `*.safetensors`, `.env*`, `.claude/`, `.agents/`, logs ignored. Tokens never written into repo / wiki / `.agents` / commands / responses.
- RAM-conservative: dry-runs, header/config/index scans by default; full conversion / coherence / multi-worker downloads gated on explicit Eric go-ahead.

---

## 3. Architecture

### 3.1 Module layout

```
jang/
├── jang-runtime/Sources/JANG/
│   ├── Zaya/
│   │   ├── ZayaModel.swift              ← copy of vmlx-swift-lm Libraries/MLXLLM/Models/Zaya.swift
│   │   ├── ZayaCCACache.swift           ← copy of MLXLMCommon/Cache/ZayaCCACache.swift
│   │   ├── BatchZayaCCACache.swift      ← copy of MLXLMCommon/BatchEngine/BatchZayaCCACache.swift
│   │   └── PROVENANCE.md                ← original path + vmlx-swift-lm commit per file
│   └── Zaya1VL/
│       ├── Zaya1VLModel.swift           ← new adapter / dispatch path. Empirically derived from `zaya1_vl` config + tensor index; do NOT presume to be a direct clone of ZayaModel. Reuse from Zaya only where tensor-name and runtime behavior demonstrably match.
│       ├── Zaya1VLVisionTower.swift     ← new; thin wrapper around Qwen25VL ViT (call into qwen_25_vl runtime)
│       ├── Zaya1VLLoRAGate.swift        ← new; vision-token mask + LoRA matmul gate (ranks 8 attn / 32 MLP)
│       ├── Zaya1VLProcessor.swift       ← new; image preprocessor wiring (Qwen2VLImageProcessor pattern)
│       ├── Zaya1VLCache.swift           ← new; ZayaCCACache + image-media salt segment
│       └── PROVENANCE.md
├── jang-runtime/Tests/JANGTests/
│   ├── ZayaSmokeTests.swift             ← copy from vmlx-swift-lm Tests/MLXLMTests/ZayaSmokeJANGTQ2Tests.swift
│   ├── ZayaCCACacheRoundTripTests.swift ← copy
│   ├── BatchZayaCCAIsolationTests.swift ← copy
│   ├── Zaya1VLSmokeTests.swift          ← new; image+text smoke
│   ├── Zaya1VLCacheRoundTripTests.swift ← new; salt-aware disk round-trip
│   └── Zaya1VLBatchIsolationTests.swift ← new; per-slot isolation incl. image media
├── jang-tools/jang_tools/
│   ├── zaya/                            ← Python text runtime; mirrors vmlx_engine cache/attention patterns
│   │   ├── runtime.py
│   │   ├── cache.py                     ← ZayaCCACache (KV + conv_state + prev_hs)
│   │   └── batch.py                     ← BatchZayaCCACache (per-slot)
│   ├── zaya1_vl/                        ← Python VL runtime
│   │   ├── runtime.py
│   │   ├── vision_tower.py              ← Qwen2.5-VL ViT call wrapper
│   │   ├── lora_gate.py                 ← vision-token-only LoRA gating
│   │   ├── processor.py                 ← image processor
│   │   └── cache.py                     ← Zaya1VLCache (CCA + media salt)
│   ├── convert_zaya_common.py           ← UNCHANGED (audit surface)
│   ├── convert_zaya_jangtq.py           ← UNCHANGED
│   ├── convert_zaya_mxfp4.py            ← UNCHANGED
│   ├── convert_zaya1_vl_common.py       ← NEW; mirrors text path; adds vision_config, image tokens, vision-LoRA preservation, processor sidecar
│   ├── convert_zaya1_vl_jangtq.py       ← NEW; routed-expert pre-stacking, 4/2-bit MXTQ profile, JANGTQ3 explicitly errored
│   ├── convert_zaya1_vl_mxfp4.py        ← NEW
│   └── capabilities.py                  ← MODIFIED; `zaya` and `zaya1_vl` map to `supports_thinking=False`
├── jang-tools/examples/zaya/             ← UPDATED (existing untracked)
│   ├── 00_inspect_source.py             (existing)
│   ├── 01_python_vllm_smoke.py          (existing)
│   ├── 02_python_runtime_contract.py    (existing)
│   ├── 03_coherence_gate.py             (existing)
│   ├── 04_prepare_hf_uploads.py         (existing; reduce to OsaurusAI-only)
│   ├── 05_cache_roundtrip.py            ← NEW; CCA disk round-trip via the bundle
│   ├── 06_batch_isolation.py            ← NEW; multi-prompt batched decode, per-slot CCA state
│   └── README.md, ATTENTION_ARCHITECTURE.md, ZayaRuntimeContract.swift   (existing)
├── jang-tools/examples/zaya1_vl/         ← NEW
│   ├── 00_inspect_source.py
│   ├── 01_python_zyphra_smoke.py        (Zyphra fork bf16 baseline)
│   ├── 02_python_runtime_contract.py    (vision tower + LoRA gate + cache)
│   ├── 03_image_text_smoke.py
│   ├── 04_cache_roundtrip.py            (CCA + image media salt)
│   ├── 05_batch_isolation.py
│   ├── 06_prepare_hf_uploads.py         (OsaurusAI-only)
│   ├── README.md
│   ├── VL_LAYERS.md                     (cross-model VL plumbing notes)
│   └── Zaya1VLRuntimeContract.swift
└── docs/runtime/
    ├── 2026-05-09-zaya-runtime-status-matrix.md   ← runtime-status across vmlx-swift-lm, jang-runtime, jang-tools, mlx-vlm
    ├── 2026-05-09-vl-layers-cross-model.md        ← Qwen3.5/3.6-VL, Holo3, Mistral-3 Pixtral, ZAYA1-VL plumbing
    └── 2026-05-09-osaurus-bundle-readme-template.md  ← README template incl. runtime support matrix
```

### 3.2 Cache topology contract (P1 invariant — must hold across Swift + Python)

ZAYA family CCA hybrid:

- Per attention layer: standard `KV [B, 2, T, 128]` + CCA inner `conv_state [B, 1280, 2]` and `prev_hs [B, 2048]`.
- Disk round-trip serializes all three. Restoring `KV` without `conv_state` / `prev_hs` is a **false hit** and must be rejected.
- Prefix cache: disabled in first port. Re-enable only after `(prefix_hash, image_media_salt)` reuse tests pass.
- Paged KV: only standard KV is paged. Block restoration must check the `(KV, CCA-inner)` pair.
- Batched decode: per-slot CCA state via `BatchZayaCCACache` gather/scatter.
- Disk store: `TQDiskSerializer` integration for ZAYA already in vmlx-swift-lm (`1d3d1a2`); copy that wiring.

ZAYA1-VL extension:

- `Zaya1VLCache` = `ZayaCCACache` + `media_salt: bytes` segment derived from preprocessed image hash.
- Vision tower output is prefilled once per image and inlined at the `image_token_id` positions; cache restoration must match `(media_salt)` or refuse to reuse.
- Vision-LoRA contributes a per-token mask; the cache stores the LoRA-applied hidden states, not the raw projections.

### 3.3 JANGTQ + MXFP4 weight policy (per `jang_config.json`)

| Class | Routed experts | CCA / attention linears | `o_proj` | `embed` / `lm_head` | Router | norms / `temp` / residual scale / balancing biases | Vision tower | Vision LoRA |
|---|---|---|---|---|---|---|---|---|
| MXFP4 | 4-bit affine `gs=32` | bf16 passthrough | bf16 passthrough | 8-bit affine `gs=32` | bf16 passthrough | bf16 passthrough | bf16 passthrough (Qwen ViT precision floor) | bf16 passthrough |
| JANGTQ2 | 2-bit MXTQ pre-stacked | 8-bit affine `gs=32` | 8-bit affine `gs=32` | 8-bit affine `gs=32` | bf16 passthrough | bf16 passthrough | bf16 passthrough | bf16 passthrough |
| JANGTQ4 | 4-bit MXTQ pre-stacked | 8-bit affine `gs=32` | 8-bit affine `gs=32` | 8-bit affine `gs=32` | bf16 passthrough | bf16 passthrough | bf16 passthrough | bf16 passthrough |

Routed-expert `linear_fc1` splits into `gate_proj` / `up_proj` halves, then pre-stacks under one switch-MLP namespace per layer / projection. JANGTQ bundles ship `jangtq_runtime.safetensors` sidecar built by `python3 -m jang_tools.build_jangtq_sidecar`. JANGTQ3 attempts are rejected at converter-entry with a clear error.

### 3.4 Runtime-status matrix surface

Every bundle's Osaurus `README.md` carries this table (templated under `docs/runtime/2026-05-09-osaurus-bundle-readme-template.md`):

| Runtime | Supported version | Tested | Known issues / fixes | Notes |
|---|---|---|---|---|
| `vmlx-swift-lm` | commit ≥ `b9da180` (text) / pending VL | yes / pending | links to fix log entries | required vmlx pin per family |
| `jang-runtime` (Swift) | tag ≥ `2.6.0` | yes | (per-runtime issue log) | ditto |
| `jang-tools` (Python MLX) | `2.6.x` | yes / pending | (per-runtime issue log) | mlx-vlm version constraint when applicable |
| `Zyphra transformers fork` | `zaya1-vl` branch | source-side baseline only | quant kernels not in fork | reference for coherence diff |

Cross-runtime issue/fix log: `docs/runtime/issues/<YYYY-MM-DD>-<short-slug>.md`, one file per issue + fix, linked from the matrix.

---

## 4. Sequenced execution (P1 → P5)

### P1 — Foundation: ZAYA runtime in jang + bundles + examples + Osaurus upload

P1 is the headline deliverable and is itself sequenced internally:

| Step | Output | Gate |
|---|---|---|
| P1.0 | `jang_tools/capabilities.py` patch: `zaya` and `zaya1_vl` → `supports_thinking=False`. Re-run gates A + B from `.agents/RUNTIME_BUNDLE_EXAMPLES.md`. | Both gate A (structure) and gate B (capabilities) pass for the three text bundles. Cap fix is **necessary, not sufficient**, for upload. |
| P1.0b | Pinned-commit verification of `vmlx-swift-lm` runtime evidence: lock to a clean commit (or stash dirty changes), re-run the test suite, record the commit SHA in `Zaya/PROVENANCE.md`. | Pinned commit identified, test suite green, SHA recorded. |
| P1.1 | Source-of-truth copy: `jang-runtime/Sources/JANG/Zaya/{ZayaModel, ZayaCCACache, BatchZayaCCACache}.swift` + `PROVENANCE.md` + 3 tests | `swift test --filter Zaya` green; CCA disk round-trip and batch isolation pass |
| P1.2 | Python text runtime: `jang_tools/zaya/{runtime, cache, batch}.py` mirroring vmlx_engine patterns | `pytest jang-tools/tests/test_zaya.py` green |
| P1.3 | `jang-tools/examples/zaya/05_cache_roundtrip.py` + `06_batch_isolation.py` exercising the existing text bundles | scripts decode "2+2=4" and "Paris" coherently from each of the three text bundles, cache round-trip succeeds |
| P1.4 | OsaurusAI upload: `OsaurusAI/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}` with READMEs carrying the runtime-status matrix | **Full text-bundle upload bar (necessary AND sufficient):** (1) gate A structure pass; (2) gate B capabilities pass post P1.0; (3) sidecar present for JANGTQ; (4) gate C runtime smoke green on the intended runtime path with single-prompt low-token decode + cache-warning capture; (5) bundle metadata + README assert non-thinking, `tool_parser` and `reasoning_parser` match policy; (6) bundles visible on HF after `hf upload`; (7) no AI attribution anywhere. |
| P1.5 | `jang_tools/convert_zaya1_vl_{common, jangtq, mxfp4}.py` (NEW) | `--dry-run` prints expected output index for the bf16 source headers |
| P1.6 | `hf download Zyphra/ZAYA1-VL-8B → ~/models/Zyphra/ZAYA1-VL-8B/` (19.5 GB, low workers) | source bundle complete; `00_inspect_source.py` clean |
| P1.7 | Three VL bundles converted: `~/models/Zyphra/ZAYA1-VL-8B-{MXFP4, JANGTQ2, JANGTQ4}` | structure verifier passes (incl. vision_config preserved, image tokens preserved, sidecar present for JANGTQ) |
| P1.8 | `jang-runtime/Sources/JANG/Zaya1VL/` Swift module + tests | `swift test --filter Zaya1VL` green; image+text smoke passes; CCA + media salt cache round-trip passes; batch isolation passes |
| P1.9 | `jang_tools/zaya1_vl/` Python module + tests | image+text smoke passes; cache + media salt round-trip passes |
| P1.10 | Coherence proof: Zyphra fork bf16 reference outputs vs MLX bundle decode | small image+text suite within tolerance for MXFP4 / JANGTQ4; JANGTQ2 expected drift documented |
| P1.11 | OsaurusAI upload: `OsaurusAI/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}` with READMEs | bundles visible on HF, READMEs include runtime-status matrix and known-issue list |

P1.0–P1.4 land first as a clean text-only batch. P1.5–P1.11 form the VL batch. Both may not interleave: text upload publishes only after the VL converter at least has its `--dry-run` validated against the source headers, so the text README's runtime-status matrix can correctly say "VL conversion validated on 2026-05-XX, upload pending P1.11".

### P2 — Documentation expansion

| Doc | Path | Gate |
|---|---|---|
| Runtime-status matrix | `docs/runtime/2026-05-09-zaya-runtime-status-matrix.md` | covers vmlx-swift-lm + jang-runtime + jang-tools + Zyphra fork; updated whenever a runtime gains/loses support |
| VL layers cross-model | `docs/runtime/2026-05-09-vl-layers-cross-model.md` | Qwen3.5/3.6-VL, Holo3, Mistral-3 Pixtral, ZAYA1-VL plumbing in one place; image-token insertion, processor sidecars, vision tower precision policy, LoRA/projector merge policy, cache boundary, quant precision floors |
| Osaurus README template | `docs/runtime/2026-05-09-osaurus-bundle-readme-template.md` | reusable template with the runtime-status matrix the user mandated |
| Per-runtime issue/fix log | `docs/runtime/issues/<date>-<slug>.md` | one entry per known issue + fix linked from the matrix |
| Wiki entries | `~/wiki/docs/wiki/research/zaya1-vl-architecture.md`, `~/wiki/docs/wiki/research/zaya1-vl-conversion-pattern.md` | wiki check passes |

### P3 — Runtime sync from `~/vmlx` into `jang-tools`

Read-only diff of `~/vmlx/engine/vmlx_engine` against `jang-tools` patterns. Pull only minimal, source-backed patches with provenance recorded. Do not bulk-copy. Largely absorbed by P1.2 / P1.9.

### P4 — `jang` repo housekeeping

| Item | Action |
|---|---|
| Untracked triage | `codex_dsv4_fixkit/`, `kimi40_calib/`, `kimi_v3_calib/`, `swift-stage/`, `nemotron_omni_v2/`, `he_5probe.py`, `humaneval_dsv4_fixed.py`, `jangtq_issues_report.md`, `verify_quant_metadata.py` — categorize each as keep / move / delete with explicit Eric approval |
| `.gitignore` | already covers `.agents/`, `.claude/`, `models/`, `.env*`. Audit for: `.notes/`, `*-private/`, agent worktrees, hardcoded path leaks |
| Tag conflict | `origin/v2.1.5` mismatch — investigate, do not force; reconcile with explicit approval |
| Branch reconciliation | `jangtq-na-phase-a` is 4 ahead / 17 behind `origin/main`; rebase plan after P1 ships |
| Hardcoded paths | sweep for `/Users/eric/jang/models/Zyphra` and similar in untracked converters; replace with `JANG_MODELS_ROOT` env or CLI arg |

### P5 — Wiki updates

- `~/wiki/docs/wiki/research/zaya1-vl-8b.md` — entity page (model card summary, source HF, license, eval table snapshot)
- `~/wiki/docs/wiki/research/zaya1-vl-architecture.md` — `zaya1_vl` arch (40 layers, vision-LoRA, partial RoPE, Qwen2.5-VL ViT, special tokens, cache topology, JANGTQ profile recipe)
- `~/wiki/docs/wiki/research/vl-conversion-pattern.md` — generalized cross-model VL conversion recipe
- `~/wiki/docs/wiki/index.md` updated; `~/wiki/docs/wiki/log.md` appended

---

## 5. Error handling and verification

### 5.1 Verifier (`jang_tools.verify_directory`)

Verifier is extended (P1.0) so `model_type=zaya1_vl` is a known target. Checks:

- Every required file present (`config.json`, `jang_config.json`, `model.safetensors.index.json`, tokenizer + chat template, preprocessor + image processor for VL, `jangtq_runtime.safetensors` for JANGTQ).
- `config.json` has `model_type=zaya` or `zaya1_vl`; `architectures` matches; `vision_config` present for VL; image / vision-start / vision-end token IDs preserved.
- `jang_config.json` declares correct `weight_format`, `quantization.profile`, `mxtq_bits` or `routed_expert_bits`.
- Index totals match shard sums; no missing tensors; no leftover unswitched expert keys for JANGTQ pre-stacked path; `vision`, `lora`, `local_experts`, `router` counts match the source index for VL.
- `supports_thinking=False` for `zaya` and `zaya1_vl`.
- Sidecar header keys match expected schema for the declared profile.

Verifier failures block upload; warnings logged but non-fatal.

### 5.2 Coherence proof (P1.10)

Zyphra fork in a local venv decodes a small image+text suite (3 prompts: pure-text math, image-caption, image-VQA) at temperature 0 against bf16 source. Reference outputs persisted to `examples/zaya1_vl/coherence/reference.json`. MLX bundles decode the same prompts via `jang-tools/zaya1_vl/runtime.py`. Numerical diff per token is logged. JANGTQ2 expected to drift more; tolerance is profile-aware. Cache hit/miss test repeats the same prompt with same image and asserts cache hit; with new image asserts cache miss (media salt mismatch).

### 5.3 Cache invariants enforced in tests

Both Swift and Python suites cover, per `Zaya` and `Zaya1VL`:

1. KV-only restore is rejected (false hit).
2. CCA-inner serialization round-trips byte-exact.
3. Per-slot batch isolation: two prompts + two images, cross-slot leakage = test failure.
4. Disk store + reload after process restart yields identical next-token logits within float tolerance.
5. (VL) Same prompt + different image → cache miss; same prompt + same image → cache hit.

### 5.4 Failure modes per runtime documented (P2)

For each runtime that does not yet carry the fix, README plus issue log explicitly describe the failure mode (unsupported `model_type`, no VL dispatch, missing image path, etc.) so users on older pins see a clear error rather than silent garbage.

---

## 6. Rollout sequence and locking

P1.0 → P1.4 (text batch): days. P1.5 → P1.11 (VL batch): ≈ 2 weeks given Swift port + tests + coherence work. Parallel work within a step is fine; P1 steps land in order. P2–P5 follow P1 sequentially.

Pre-action lock entries in `.agents/CURRENT.md` for any of:

- 19.5 GB download
- full conversion run
- runtime smoke that loads weights
- `git commit` or `git push` from this checkout
- any `hf upload` invocation

---

## 7. Decisions locked in this spec (changeable on review)

1. JANGTQ3 excluded for ZAYA family.
2. `OsaurusAI` only; no `JANGQ-AI` for this batch.
3. ZAYA family non-thinking; `supports_thinking=False` everywhere.
4. `vmlx-swift-lm` and `~/vmlx` are read-only; copies into `jang-runtime` / `jang-tools` carry `PROVENANCE.md`.
5. Coherence proof path: Zyphra `transformers@zaya1-vl` fork in a local venv as bf16 baseline, MLX bundles diffed against it.
6. `Zaya1VL` Swift module home: `jang-runtime/Sources/JANG/Zaya1VL/` (new submodule).
7. Naming: text = `ZAYA1-8B-*`; VL = `ZAYA1-VL-8B-*`. Never mix.
8. Bundle storage outside repo at `~/models/Zyphra/`. Repo `.gitignore` already covers `models/`.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Zaya1VL Swift port underestimates effort | P1.8 has a hard checkpoint after image+text smoke; if smoke not green by day-7 the port re-scopes (e.g., land Python-only first, Swift later) |
| Vision-LoRA gating numerically unstable when fused with JANGTQ-quantized base linears | Keep LoRA bf16, apply additively at full precision; numerical sanity test compares `(quant_base + lora_bf16)` vs `(bf16_base + lora_bf16)` per layer |
| Quant kernels not present in Zyphra fork → no quant-side reference path in their runtime | Coherence diff stays bf16-source-vs-MLX-bundle; bundle-vs-bundle drift bounded by JANGTQ2/4 tolerance budget; this is documented |
| Concurrent edit overlap | `.agents/CURRENT.md` Status section + `Locked: <action>` pre-action notes; explicit "do not touch" lists per agent |
| Pre-commit / CI on `jang` blocks the converter PR | Run `pre-commit run --all-files` and `pytest jang-tools/tests` before any commit; explicit Eric review before push |
| `~/models/Zyphra/ZAYA1-VL-8B/` partial download corrupts on retry | Use `hf download --max-workers 2` and verify pinned commit SHA matches before any conversion |
| Existing untracked `convert_zaya_*.py` (text) and the new `convert_zaya1_vl_*.py` diverge in conventions | Common helpers extracted into `jang_tools/zaya_common/`; both converters import from it |

---

## 9. Open items for Eric

These remain "policy" choices that can flip the spec; defaults assumed if no redirect:

1. Coherence proof tolerance per profile. Default: per-token logit cosine ≥ 0.98 for MXFP4 / JANGTQ4, ≥ 0.92 for JANGTQ2.
2. Whether `OsaurusAI/ZAYA1-VL-8B-MXFP4` README should explicitly mark the bundle "preview" until a Swift `Zaya1VL` runtime tag ships, or hold the upload until P1.8 lands. Default: hold upload until P1.8 (matches "make sure it works with all caching first").
3. Whether to delete the existing untracked `convert_zaya_*.py` text-only files and replace with the new `convert_zaya_common.py` shared helpers. Default: keep them (audit surface) until P1.4 ships; refactor later in P4.
4. Branch policy: P1 lands on `jangtq-na-phase-a` or a fresh `feature/zaya-runtime-vl` branch. Default: fresh branch off current `origin/main` to avoid the `jangtq-na-phase-a` 4-ahead/17-behind divergence.

---

## 10. Definition of done

P1 done when:

- `OsaurusAI/ZAYA1-8B-{MXFP4,JANGTQ2,JANGTQ4}` and `OsaurusAI/ZAYA1-VL-8B-{MXFP4,JANGTQ2,JANGTQ4}` are live on HF.
- Each repo's README has the runtime-status matrix populated with verified entries.
- `swift test` and `pytest jang-tools/tests` green for ZAYA + ZAYA1-VL targets.
- `examples/zaya/` and `examples/zaya1_vl/` smoke + cache scripts succeed against each uploaded bundle.
- `jang_tools.capabilities` and `verify_directory` recognize `zaya1_vl` and stamp `supports_thinking=False` for the family.

P2–P5 done when:

- Cross-runtime documentation under `docs/runtime/` is current and linked from each Osaurus README.
- Wiki has the three new pages and the index/log are consistent (`wiki-check` passes).
- Repo housekeeping: `.gitignore` finalized, untracked triage resolved, branch reconciled with `origin/main`, tag conflict explained or closed.
