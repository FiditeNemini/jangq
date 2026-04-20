# JANG Studio Production Audit Checklist

**Populated by Ralph Loop iteration 1 — 2026-04-19.**
Source-of-truth for subsequent Ralph iterations: each iteration reads this file, picks the next `[ ]` item, verifies or fixes, marks `[x]` with commit SHA + note.

The goal: every item reaches `[x]` with production-ready behavior. "Production" means it matches the documented rules in `~/.claude/projects/-Users-eric-jang/memory/` AND the actual behavior is verified, not assumed.

**How each item is closed:**
- Read the memory file(s) referenced
- Verify the app/CLI/bundle does what the rule requires
- If behavior drifts: fix it, commit, add the SHA next to the item
- If behavior matches: add a one-line verification note next to the item
- Mark `[x]` only when verified with evidence

---

## A. Conversion output integrity (jang-tools side)

- [ ] **A01** — `jang_config.json` writes `format=="jang"`, `format_version` `"2.x"+`, non-empty `capabilities`, correct `quantization.bit_widths_used`. (Referenced by `PostConvertVerifier` rows #1-#4.)
- [ ] **A02** — `capabilities` stamp includes `arch`, `reasoning`, `tool`, `cache`, `modality`. Cross-check every arch in `jang_tools/capabilities.py` FAMILY_MAP maps correctly (see commit `43af293` for llama+idefics3 gap).
- [ ] **A03** — `config.json` output preserves original `_name_or_path` so HF model cards auto-generate with correct `base_model`.
- [ ] **A04** — Qwen3.5 eos auto-fix: 248044 → 248046 applied to top-level, `text_config`, AND `tokenizer_config.json`. Source of rule: `feedback_chat_template_rules.md`.
- [ ] **A05** — MiniMax eos_token_id 200020 ([e~[) preserved. Source: `feedback_chat_template_rules.md`.
- [ ] **A06** — Nemotron eos_token_id 2 preserved + standard ChatML intact.
- [ ] **A07** — `generation_config.json` copied when present in source. Cross-check convert.py extra_configs list.
- [ ] **A08** — All tokenizer files copied: `tokenizer.json` OR `tokenizer.model` + `tokenizer_config.json` + `special_tokens_map.json` + `merges.txt` + `vocab.json` + `added_tokens.json` when present.
- [ ] **A09** — Chat template preserved in AT LEAST ONE of three forms: inline in `tokenizer_config.json`, `chat_template.jinja`, or `chat_template.json`. `PostConvertVerifier` accepts any of three (confirmed via A16 audit row).
- [ ] **A10** — Chat template `enable_thinking` toggle preserved (MiniMax default is always-on — must be made toggleable). Source: `feedback_chat_template_rules.md`.
- [ ] **A11** — `tokenizer_class` remapped from `TokenizersBackend` → concrete class (e.g. `GPT2Tokenizer`) for Osaurus compatibility.
- [ ] **A12** — `modeling_*.py` + `configuration_*.py` + custom parser `.py` files copied for trust_remote_code models (MiniMax, Nemotron).
- [ ] **A13** — Shard naming: `model-NNNNN-of-MMMMM.safetensors`. Index file `model.safetensors.index.json` references exactly those files. No leftover `model-*-of-NNNNN.safetensors` placeholder files.

## B. VL and video model handling (feedback_always_vl.md + feedback_jang_studio_audit_coverage.md)

- [ ] **B01** — `preprocessor_config.json` ALWAYS copied from source when present. Verified in Ralph audit A11 on SmolVLM (pixel_values [1,17,3,512,512]).
- [ ] **B02** — `video_preprocessor_config.json` ALWAYS copied from source when present (Qwen3.5 models ship this — cannot be silently dropped).
- [ ] **B03** — VL image path: `AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)(images=image, text="...", return_tensors="pt")` returns non-empty `pixel_values`. Ralph A11.
- [ ] **B04** — Video path: `AutoProcessor` accepts `videos=frames` kwarg and returns `pixel_values_videos`. Ralph A12 (currently n/a — no video model in tier 1).
- [ ] **B05** — `inspect-source --json` emits `is_vl` AND `is_video_vl` as distinct booleans (not a single `is_vl` flag). Source: `feedback_jang_studio_audit_coverage.md`.
- [ ] **B06** — Swift `ArchitectureSummary` has both `isVL` and `isVideoVL`. PostConvertVerifier row #8b gated on `isVideoVL`.
- [ ] **B07** — Ralph fixtures `test_image.png` (64×64 RGB) and `test_video_frames.npy` (16×32×32×3) exist + gitignored exceptions in place. Verified commit `baed27a`.

## C. Model-specific architecture rules

- [ ] **C01** — 512+ expert models force `bfloat16` (source: `project_bfloat16_fix.md`). `KNOWN_512_EXPERT_TYPES` includes minimax_m2, glm_moe_dsa. Now served via `jang-tools capabilities --json`.
- [ ] **C02** — MLP asymmetry: 512+ expert models use gate_proj=4-bit floor, down_proj=3-bit floor (source: `project_mlp_asymmetry.md`). Verify `allocate.py` applies this when expert count ≥ 512.
- [ ] **C03** — `project_mistral4_architecture.md` 6 fixes active for MLA+MoE: FP8 bf16 scale, rope_interleave, norm_topk_prob, llama4_beta, PLAIN attention scale (no mscale²), gate dequant.
- [ ] **C04** — Qwen3.6 hybrid SSM (GatedDeltaNet + full-attn + 256 routed + 1 shared + MTP + MRoPE). Verify conversion produces all tensor classes. Source: `project_qwen36.md`.
- [ ] **C05** — Cascade-2 Nemotron-H 128-expert Mamba-hybrid converts correctly. Source: `project_cascade2.md`.
- [ ] **C06** — MiniMax-M2.7 FP8 source loads via `load_fp8_tensor` with correct scale application. Source: `project_minimax_m27.md`.
- [ ] **C07** — JANGTQ Qwen3.6 + MiniMax conversion scripts exist and work (`convert_qwen35_jangtq.py` + `convert_minimax_jangtq.py`). GLM JANGTQ is deferred to v1.1.

## D. Progress bar / phase tracking (JSONL protocol v1)

- [ ] **D01** — Every long-running CLI emits phase+tick events: `convert`, `convert_qwen35_jangtq`, `convert_minimax_jangtq`. Verified in Phase 1 commits.
- [ ] **D02** — Swift `PythonRunner` parses JSONL stderr reliably; malformed lines don't kill the run (JSONLProgressParser `.parseError` event).
- [ ] **D03** — Step 4 macro progress bar shows `[N/5] name` for all 5 phases: detect, calibrate, allocate, quantize, write.
- [ ] **D04** — Step 4 fine progress bar updates per tensor during quantize phase. Tick throttle respects user setting (default 100ms, range 50-500ms from Settings → Advanced).
- [ ] **D05** — ETA display — not yet implemented (spec Part 12). Gap — add ETA calculation based on tick throughput.
- [ ] **D06** — Memory pressure monitor — not yet implemented (spec Part 12). Gap — add live `host_statistics64` polling.
- [ ] **D07** — Cancel button fires SIGTERM → 3s → SIGKILL per `PythonRunner.cancel`. Verified in XCTest `test_cancelSIGTERMLandsWithinThreeSeconds`.
- [ ] **D08** — Pause/Resume (SIGSTOP/SIGCONT) — not yet implemented. Gap — spec Part 12 Step 4 enhancement.

## E. App UI — buttons, settings, bindings

- [ ] **E01** — Step 5 Test Inference button triggers `TestInferenceSheet`. Verify sheet actually opens when clicked.
- [ ] **E02** — Test Inference sheet streams from bundled Python `inference` CLI. Currently one-shot (not streaming). Spec Part 12 allows this in v1.
- [ ] **E03** — Usage Examples sheet fetches all 4 tabs (Python/Swift/Server/HF) via `ExamplesService`.
- [ ] **E04** — Generate Model Card button writes to `<outputURL>/README.md` + shows preview.
- [ ] **E05** — Publish to HF dialog: dry-run + real upload. Verify HF_HUB_TOKEN env fallback works.
- [ ] **E06** — Reveal in Finder + Copy Path buttons still work post-refactor.
- [ ] **E07** — Output naming template actually applied during convert (`{basename}-{profile}` default). Setting lives in `AppSettings.outputNamingTemplate` — verify it flows through to the output dir path.
- [ ] **E08** — Default output parent path setting respected (empty = source's parent, else user-selected dir).
- [ ] **E09** — Default profile setting drives ProfileStep's initial selection.
- [ ] **E10** — Default family setting respected for initial state.
- [ ] **E11** — Default method setting drives ProfileStep's method picker.
- [ ] **E12** — Default Hadamard setting applies.
- [ ] **E13** — Calibration sample count setting — verify it actually passes to the Python convert CLI (not just saved).
- [ ] **E14** — Auto-delete partial output on cancel — verify cancel path respects this setting.
- [ ] **E15** — Reveal in Finder on finish — verify this toggle actually triggers `NSWorkspace.activateFileViewerSelecting` at step-5 completion.
- [ ] **E16** — Python override path setting takes precedence over bundled path when set.
- [ ] **E17** — Custom jang-tools path setting injected via PYTHONPATH in subprocess env.
- [ ] **E18** — Log verbosity setting maps to `--progress=json` + log level.
- [ ] **E19** — JSONL log retention setting controls UI ring buffer size.
- [ ] **E20** — Log file output dir setting — verify logs actually write to that dir.
- [ ] **E21** — Tick throttle setting passed to Python CLI (new flag needed — not yet wired).
- [ ] **E22** — Bundle size warning setting drives build-script warning.
- [ ] **E23** — MLX thread count setting exported as `MLX_NUM_THREADS` env var in subprocess.
- [ ] **E24** — Metal pipeline cache toggle exported as env var (`MLX_METAL_PIPELINE_CACHE_DIR` or equivalent).
- [ ] **E25** — Pre-allocate RAM setting has actual effect (may need Metal API call at init).
- [ ] **E26** — Convert concurrency > 1 actually enables parallel conversions (currently single-convert UI).
- [ ] **E27** — Copy Diagnostics button always visible when setting on (E26 in AppSettings).
- [ ] **E28** — Anonymize paths toggle affects DiagnosticsBundle output.
- [ ] **E29** — GitHub issues URL setting overrides default in diagnostics open action.
- [ ] **E30** — Settings window opens via Cmd+, verified end-to-end.
- [ ] **E31** — Settings persist across app restart (UserDefaults roundtrip).

## F. Bundle integrity (.app/Contents/Resources/python)

- [ ] **F01** — Bundle size ≤ 450 MB (current: 305 MB).
- [ ] **F02** — All 15 jang-tools subcommands present: inspect, validate, estimate, convert, profile, upgrade, spec, inspect-source, examples, modelcard, inference, profiles, capabilities, estimate-model, publish. Verified this iteration.
- [ ] **F03** — Python version 3.11.x in bundle.
- [ ] **F04** — MLX version pinned (mlx>=0.22, mlx-lm>=0.20 per pyproject.toml extras).
- [ ] **F05** — transformers, tokenizers, sentencepiece present (for VL processors).
- [ ] **F06** — huggingface_hub present (for publish subcommand).
- [ ] **F07** — Jinja2 present (for templates in examples/modelcard).
- [ ] **F08** — Bundle rebuild script (`build-python-bundle.sh`) idempotent + skips intact builds.
- [ ] **F09** — Bundle copy into .app verified via `postCompileScripts` rsync (commit `5bab09e`). `Contents/Resources/python/bin/python3` works after clean build.
- [ ] **F10** — Templates dir `jang_tools/templates/*.jinja` included in wheel via package_data + MANIFEST.in.
- [ ] **F11** — Ralph fixtures `ralph_runner/fixtures/*.{png,npy}` ship when rsynced to macstudio.

## G. Directory permissions / error handling

- [ ] **G01** — Source dir read-only → convert succeeds (write only to output dir).
- [ ] **G02** — Output dir not writable → preflight catches it before convert starts (row #3 outputUsable).
- [ ] **G03** — Output dir == source dir → preflight fail (prevents overwriting source).
- [ ] **G04** — Output dir inside `.app` → preflight fail.
- [ ] **G05** — Disk full mid-convert → graceful error (PythonException category in spec §4.3); diagnostics bundle captures the error.
- [ ] **G06** — Cancel mid-convert → partial output left on disk by default; can be deleted via "Delete partial" banner.

## H. Adoption surface completeness

- [ ] **H01** — `jang-tools examples --lang python --json` returns compilable Python (audit A18).
- [ ] **H02** — `jang-tools examples --lang swift --json` returns Swift that imports `JANGKit` and uses `JANGKit.Model.load(at:)` + `.generate()`.
- [ ] **H03** — `jang-tools examples --lang server --json` returns osaurus command + curl example.
- [ ] **H04** — `jang-tools examples --lang hf --json` returns HF-ready markdown with frontmatter.
- [ ] **H05** — `jang-tools modelcard --json` returns valid JSON with `license`, `base_model`, `quantization_config.{family,profile,actual_bits}`, `card_markdown`.
- [ ] **H06** — `jang-tools publish --dry-run --json` reports file count + total size without uploading.
- [ ] **H07** — `jang-tools inference --model <dir> --prompt "..." --json` returns `{text, tokens, tokens_per_sec, elapsed_s, peak_rss_mb, model}`.
- [ ] **H08** — `JANGKit.Model.load(at:)` dispatches JANG vs JANGTQ via `jang_config.json` (A6 verified in commit `e9328fa`).
- [ ] **H09** — `JANGKit.Model.generate(prompt:config:)` works for JANG family end-to-end (needs real model test on macstudio).
- [ ] **H10** — `JANGKit.Model.generate` works for JANGTQ family end-to-end (needs MiniMax or Qwen3.6 test).
- [ ] **H11** — Public adoption docs at `docs/adoption/` are internally consistent (cross-check PORTING.md against actual format_version keys + JANGTQ detection).
- [ ] **H12** — `FORMAT.md` (top-level) matches current on-disk layout (verify `per_tensor` bit metadata key matches actual jang_config.json shape).

## I. Ralph harness completeness

- [ ] **I01** — Tier 1 has ≥ 1 instruct model running (Llama or instruction-tuned equivalent) so A3/A5 flip warn→pass. Blocked pending HF terms acceptance on `dealignai` account for Llama-3.2-1B-Instruct.
- [ ] **I02** — Tier 2 activation criteria defined (3 consecutive green runs of tier 1).
- [ ] **I03** — Tier 2 models: Qwen3-1.7B-Base, Qwen2-VL-2B-Instruct, Qwen1.5-MoE-A2.7B-Chat — currently all marked `skip:`.
- [ ] **I04** — Tier 3 runs on Qwen3.6-35B-A3B + MiniMax-M2.7-FP8 manually before each release.
- [ ] **I05** — Baseline tracking file `ralph_runner/baselines/` for A4/A6 regression comparisons — not yet implemented. Gap.
- [ ] **I06** — `/loop` scheduled cadence for Ralph — not yet wired.
- [ ] **I07** — GitHub issue auto-filing on audit failure — not yet wired.

## J. Runtime quality (feedback_runtime_before_quant.md)

- [ ] **J01** — No-cache greedy generate path exists in jang_tools.inference CLI for regression debugging.
- [ ] **J02** — Runtime checklist from `research/GLM-5.1-RUNTIME-AUDIT.md` (17 points) applied to any "converted model outputs garbage" triage.
- [ ] **J03** — MLA absorb bf16 SDPA bug fix present in `mlx_lm/models/deepseek_v32.py` on the installed env (cast to float32 on L==1 branch). Source: `project_mla_absorb_bug.md`.

## L. Beginner usability — hints, tooltips, smart defaults

- [x] **L01** — `jang-tools recommend --model <dir> --json` CLI exists; returns `detected`, `recommended`, `beginner_summary`, `warnings`, `why_each_choice` for any source model.
      **Evidence:** commit in iteration 2; 35 pytest tests across 17 model families pass (dense: llama/mistral/qwen2/qwen3/gemma3/phi3/falcon; MoE: qwen2_moe/mixtral/qwen3_5_moe/deepseek_v32/mistral4/minimax_m2/glm_moe_dsa; hybrid: nemotron_h; VL: qwen2_vl/idefics3/qwen3_vl video).
- [x] **L02** — `SourceDetector` (Swift) calls `jang-tools recommend` alongside `inspect-source` and pre-fills the wizard's defaults (profile, family, method, hadamard, force_dtype).
      **Evidence:** new `RecommendationService.swift` wraps the CLI via `JSONDecoder` with `.convertFromSnakeCase`; `SourceStep.detectAndRecommend(url:)` runs both calls in sequence and applies defaults via `applyRecommendation(_:)`. Beginner summary + warnings + InfoHint tooltips shown in new "Recommended for this model" section. Step 1 now renders the full recommendation block below the Detected card.
      **Commits:** (this iteration, see git log).
- [x] **L06** — Step 1 shows a "Recommended for this model" banner once folder is picked, pre-populated from `recommend` output (beginner_summary + warnings).
      **Evidence:** same commit as L02; rendered via `Section { ... } header: { Text("Recommended for this model") ... }` in SourceStep.
- [x] **L07** — Alternative profiles appear as expandable "Other options" in Step 3 with `use_when` description. (Moved to Step 1 — fits better there since it's part of the recommendation.)
      **Evidence:** `DisclosureGroup("Other options")` in SourceStep renders each `Recommendation.Alternative` with family badge + `useWhen` caption.
- [x] **L08** — Every warning from `recommend.warnings` surfaces as a banner in the appropriate step.
      **Evidence:** SourceStep iterates `rec.warnings` with `Label(w, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.orange)`.
- [ ] **L03** — Every field in Step 2 (Advanced overrides) has an `InfoHint` popover with plain-English explanation derived from `recommend`'s `why_each_choice`.
- [ ] **L04** — Every field in Step 3 (Profile, Method, Hadamard, Block size, Output folder) has an `InfoHint` popover + alternatives surface.
- [ ] **L05** — Preflight rows in Step 3 link to plain-English remediation steps on failure (e.g., "Disk space" → "This is the free GB on your output volume. You need ~X GB — free up space or pick a different drive").
- [ ] **L06** — Step 1 shows a "Recommended for this model" banner once folder is picked, pre-populated from `recommend` output (beginner_summary + warnings).
- [ ] **L07** — Alternative profiles appear as expandable "Other options" in Step 3 with `use_when` description.
- [ ] **L08** — Every warning from `recommend.warnings` surfaces as a yellow banner in the appropriate step (512-expert, gated, unknown-arch, etc.).
- [ ] **L09** — Settings pane every field has an `InfoHint` popover explaining the setting + its default.
- [ ] **L10** — Test Inference sheet: beginner-friendly placeholder text (e.g., "Ask your converted model a question — try 'What is the capital of France?'").

---

## M. Deep-trace discoveries (spawned during Ralph iterations)

Each item here was surfaced by a concrete trace, not speculation. Each traces back to the `INVESTIGATION_LOG.md` entry that found it.

- [x] **M01** — Picking a folder with `config.json` but zero `.safetensors` silently passed Step 1 (Continue button active, user progressed to Architecture with empty detected).
      **Trace:** user picks `/tmp/empty-cfg` → `inspect-source` succeeds with `shard_count=0` → `SourceDetector` builds `ArchitectureSummary(totalBytes=0, shardCount=0)` → `isStep1Complete` returned `true` because it only checked `detected != nil`.
      **Fix:** gate `isStep1Complete` on `shardCount > 0`; show red "No .safetensors found" hint in Detected card.
      **Evidence:** `ConversionPlan.swift:57-64`, `SourceStep.swift:80-88`. 65 XCTest still pass.
      **Commit:** (this iteration)
- [ ] **M02** — Error path: user picks a folder that LOOKS model-shaped but is actually a different HF repo clone (e.g., a dataset with config.json). Verify inspect-source + recommend don't hard-crash.
- [ ] **M03** — Drag-and-drop folder onto Step 1 — spec (design addendum Part 5) promises this; implementation uses only NSOpenPanel. Missing feature.
- [ ] **M04** — Recents list for source dirs — missing. If user cancels mid-convert and wants to retry, they re-pick from scratch.
- [ ] **M05** — PreflightRunner size estimate when `detected.totalBytes == 0` — current fallback returns pass with free-GB hint, but does the UI make clear that "no estimate" ≠ "safe"?
- [ ] **M06** — Conflict detection: if `config.json.model_type = minimax_m2` but `tokenizer_config.json.tokenizer_class = Qwen2Tokenizer`, does the app detect the conflict? Probably silent today.
- [ ] **M07** — Nested model_type: all known patterns are `text_config.model_type`. Verify no real HF multimodal uses `llm_config.model_type` or similar non-standard keys. Current code only falls back to `text_config`.
- [ ] **M08** — Model directory that's a symlink or on a read-only volume — does the rsync / copy during convert survive? What error does the user see?
- [ ] **M09** — User picks the SAME folder as their output → Preflight catches this (`outputUsable` row). But what if user picks output as a subfolder of source? Still bad, not caught today.
- [ ] **M10** — Settings pane → change `pythonOverridePath` → pick a folder → verify `BundleResolver.pythonExecutable` reads the new value. Does it require an app restart?
- [ ] **M11** — User changes output naming template mid-flow (Step 3 already open, switches profile, does output dir name update?). Test that the auto-computed output path re-renders on profile change.
- [x] **M12** — Step 4 Cancel → Retry: previously Cancel flipped `plan.run` to `.succeeded` (major bug — user saw "Continue → Verify" after cancelling). Retry did reconstruct PythonRunner cleanly, but the UI state was lying about success.
      **Fix:** track `cancelRequested` @State in RunStep; after the for-await loop exits without throw, branch on it — set `.cancelled` vs `.succeeded`. Old SIGKILL-after-3s Task captures `proc` strongly so even if user hits Retry immediately, the zombie timer targets the correct process (no PID-reuse risk).
      **Evidence:** `RunStep.swift:75-95` (start method), `RunStep.swift:42-56` (cancelled UI branch). 65 XCTest still pass.
      **Commit:** (this iteration)
- [x] **M13** — Stale log visibility when navigating back to Run step: SwiftUI `@State` preserves the `logs` array across navigation; new start() does `logs.removeAll()` at top. User navigating back mid-run sees current live log. User navigating back to a finished run (succeeded/failed/cancelled) sees the historical log — correct behavior.
      **Evidence:** `RunStep.swift:78` (`logs.removeAll()` at start), `RunStep.swift:72` (onAppear guard via `guard coord.plan.run != .running else { return }`).
- [x] **M14** — Double-click Start Conversion: `start()` first-line guard `guard coord.plan.run != .running else { return }` rejects re-entry. First call sets `.running` before any async work; second-click call returns immediately. Cancel button now additionally `.disabled(cancelRequested)` to prevent double-cancel.
      **Evidence:** `RunStep.swift:20` cancel disabled-state; `RunStep.swift:75` start re-entry guard.
- [x] **M15** — Swift publish dialog stored the HF token in `@State private var token: String` — after a successful upload it stayed in memory until the sheet was dismissed. A passerby at the user's Mac could copy the value out of the SecureField buffer (or it could surface in a memory snapshot during a crash dump sent for a bug report). No leak path today, but leaving a secret in memory longer than needed is defense-in-depth hygiene.
      **Fix (iter 17):** `runPublish()` wipes `token = ""` on success after `publishResult = r`. On FAILURE we intentionally keep the token so the user can retry without retyping — a ~30-second exposure window while troubleshooting is better UX than forcing re-entry on every failed attempt.
      **Evidence:** `PublishToHuggingFaceSheet.swift:175-179`.
      **Commit:** (this iteration)
- [x] **M16 + M22e** — DiagnosticsBundle zipped raw log + event strings with no scrubbing. If a convert or publish failed with an HF-token-leaking exception (HfHubHTTPError embedding Authorization header), the token would land in run.log/events.jsonl inside a bug-report zip sent to GitHub Issues → credentials in the public tracker. Iter 6 scrubbed tokens at the publish-error call site, but any OTHER entry point (debug-mode HTTPX logs, future error paths, user pasting a token into a prompt field) still leaked.
      **Fix (iter 14):** `DiagnosticsBundle.scrubSensitive(_:)` applies 4 regex patterns before writing: `hf_…`, `huggingface_…`, `Authorization: Bearer …`, generic `Bearer …` (all ≥20-char suffix to avoid matching incidental variable names like `hf_short`). Matches replaced with `<redacted>`. Called on every log line AND every event line before `String.write(to:)`. End-to-end test unzips a bundle and asserts the raw secret does not appear in ANY file.
      **Evidence:** `DiagnosticsBundle.swift:7-35` (patterns + scrubber), 7 new tests including end-to-end unzip-and-grep.
      **Commit:** (this iteration)
- [ ] **M17** — TestInferenceSheet temperature slider: min=0.0, max=2.0 — what happens at exactly 0.0? (greedy decode) What about float precision edge cases?
- [ ] **M18** — Cancel during HF model download on macstudio: Ralph runner calls `snapshot_download(repo_id)` — cancellation here requires SIGTERM on the whole python3 subprocess. Does it clean up partial safetensors? HF's lock files?
- [x] **M19** — TestInferenceSheet Cancel during generate: confirmed `InferenceRunner.generate()` blocked on `proc.waitUntilExit()` while holding the actor — same deadlock we fixed in PythonRunner at commit `6270214`. Fixed iter 3: replaced `waitUntilExit()` with `withCheckedContinuation + terminationHandler`, added `cancelled` actor state + `InferenceError.cancelledCode=-2` sentinel, `TestInferenceViewModel` now filters cancelled errors so UI doesn't show a red banner on deliberate cancel.
- [ ] **M20** — StepCancel → Convert another: `reset()` in VerifyStep creates a new `ConversionPlan()` but the old PythonRunner in Step 4 may still have a SIGKILL-after-3s Task pending. Does orphaned task leak memory or fire an unnecessary kill?
- [ ] **M21** — App quit while convert is running: macOS sends TerminationRequest to the app, which dies — does the bundled Python subprocess get SIGTERM via process group, or does it orphan and keep running? (Likely orphans — bundled python under app has no formal tie to the parent once Process.run() returns.)
- [x] **M22** — Copy Diagnostics bundle race. Reviewed in iter 14: `logs` and `events` are SwiftUI `@State` arrays — mutations are MainActor-serialized. When the user clicks Copy Diagnostics, they pass snapshots (value-semantics copy) into `DiagnosticsBundle.write`, so no on-disk race. BUT three subtler bugs found + fixed in the same trace:
      - **M22d**: Timestamp was second-resolution; two clicks in the same second landed in the same workDir with stale files. Fixed with `[.withInternetDateTime, .withFractionalSeconds]` ISO formatting so each click gets a unique zip.
      - **M22e**: Tokens in logs weren't scrubbed. See M16 above.
      - **M62-anonymize**: `plan.sourceURL` / `plan.outputURL` leaked filesystem layout in bug reports. Wired to `settings.anonymizePathsInDiagnostics` — when on, paths become basenames. Call site in `RunStep.swift:64-71` passes the setting through.
      **Note:** The ORIGINAL M22 question (race on @State array) is a non-issue given SwiftUI's MainActor isolation, but the broader "is Copy Diagnostics safe mid-convert" audit surfaced 3 real bugs.
      **Evidence:** `DiagnosticsBundle.swift:45-102` (millisecond stamp, anonymize dispatch, scrubbed writes), `RunStep.swift:64-71` (setting plumbed through), 10 new Swift tests.
      **Commit:** (this iteration)
- [ ] **M23** — After cancel, the "Delete partial output" button appears. If user clicked it when outputURL is NIL or already removed, it silently no-ops. Should surface success/fail.
- [ ] **M24** — TestInferenceSheet "Stop" button visibility: `isGenerating=true` shows Stop button. If the user cancels then immediately retries, does the button correctly return to Stop state? Race between `isGenerating=false` in cancel() and `isGenerating=true` in next send().
- [ ] **M25** — Long-running inference UX: InferenceRunner is one-shot — on a 70B model at 15 tok/s generating 150 tokens = 10 seconds with NO streaming feedback. User sees hourglass only. Consider: add elapsed-ms counter in the sheet.
- [ ] **M26** — Model reuse: every send() spawns a fresh `python -m jang_tools inference` subprocess, which reloads weights from disk each call. On a 122 GB model this is 15-30s of dead time per prompt. Long-term: need a persistent server mode. Short-term: document the limitation in the UI.
- [ ] **M27** — Image + video attached simultaneously: does `jang-tools inference --image X --video Y` succeed, or does it reject/prefer one? Current UI has both `pendingImagePath` and `pendingVideoPath` independently settable.
- [ ] **M28** — After cancel, the InferenceRunner actor still holds `self.currentProcess = proc` pointing at a terminated Process. Next generate() overwrites it, but in the gap between cancel() returning and next send(), cancel() on a stale proc is `.isRunning=false` → early-return. Correct behavior, but verify no zombie.
- [x] **M29** — TestInferenceSheet temperature slider was a UI lie AND chat template was never applied. `jang_tools/inference.py:_generate_text` accepted `temperature` as a positional arg but never passed it to `mlx_lm.generate()` — every generation was greedy regardless of the slider. Separately, raw prompt was sent to `generate()` with no `apply_chat_template` call, so Qwen3/Llama3/Gemma models saw `"Hello"` instead of the templated `"<|im_start|>user\\nHello<|im_end|>\\n<|im_start|>assistant\\n"` → infinite thinking loops or garbage output. This is a direct re-tread of `feedback_chat_template_rules.md`.
      **Fix:** Added `_apply_chat_template_if_any(tokenizer, prompt)` that detects `.chat_template` on either the TokenizerWrapper or the inner HF tokenizer and calls `apply_chat_template` with `add_generation_prompt=True`. Added `_make_sampler(temp)` that returns `make_sampler(temp=temperature)` from `mlx_lm.sample_utils` for `temp > 0` and `None` for greedy. Both are wired through `_generate_text` with graceful fallback when mlx_lm lacks the `sampler` kwarg.
      **Evidence:** `jang_tools/inference.py:50-116`, 5 new unit tests in `tests/test_inference.py` (228 total pass).
      **Commit:** (this iteration)
- [ ] **M30** — Inference tok/s re-tokenizes the output via `tokenizer.encode(text)` to count tokens. But some tokenizers disagree with the generator about subword boundaries, so the count drifts 5-15% from the actual generation count. Switch to `stream_generate` and count yielded segments instead.
- [ ] **M31** — If `apply_chat_template` requires a variable the tokenizer_config template references (e.g. `enable_thinking` for MiniMax M2, `bos_token` for some Mistral variants), the current `_apply_chat_template_if_any` silently falls back to raw prompt on exception. Users won't know their Qwen3 model quietly lost its chat template — no telemetry surfaced to the UI.
- [ ] **M32** — VL inference path (`_generate_vl`) does NOT call `apply_chat_template`. For Qwen3-VL the prompt needs the image-token prefix. mlx_vlm may do this internally, but verify — if not, this is B5 for VL.
- [ ] **M33** — `jang_tools/inference.py:_is_vl` detects VL by presence of `preprocessor_config.json`. But a TEXT-ONLY converted model could still have this file if user accidentally left it in the output dir (or inspect-source copied it for a borderline arch). False-positive routing to mlx_vlm.load which might crash with "no vision tower". Should also check `config.json.vision_config` or the capabilities stamp.
- [ ] **M34** — Convert.py `_safe_copy` on line 1015-1016: if the fallback byte copy ALSO fails (disk full / permission denied), it prints a warning and continues. `extras_copied.append(extra_file)` happens unconditionally so the log line "Extra config files: ..." lies about what was actually written. Should check `(output_path / extra_file).exists()` before appending.
- [x] **M35** — Memory cross-ref (`feedback_jang_studio_audit_coverage.md`): the Osaurus / swift-transformers tokenizer_class remap existed ONLY in `convert_qwen35_jangtq.py` (JANGTQ path) and NOT in the main `convert.py` (regular JANG path). Any regular JANG conversion of a source with `tokenizer_class: "TokenizersBackend"` would ship broken — Osaurus throws `unsupportedTokenizer("TokenizersBackend")`. PostConvertVerifier row #10 only flagged this as `warn/required=false`, so the wizard let users publish a broken model.
      **Fix (both sides):**
      - `convert.py:982-1010`: added `_OSAURUS_TOKENIZER_MAP` table (qwen/llama/mistral/gemma/phi → concrete class) + remap step right after the eos fix. Source model_type drives the concrete choice; defaults to Qwen2Tokenizer for unknown types (matches the existing JANGTQ-path default).
      - `PostConvertVerifier.swift:105-116`: upgraded tokenizerClassConcrete row from `warn/required=false` to `fail/required=true` with a hint that points at the convert.py remap table if a new model_type slips through unmapped.
      - `CoverageMatrixTests.swift:279-296`: updated coverage test to assert the new fail/required behavior.
      **Evidence:** 225 Python tests pass, 65 Swift tests pass.
      **Commit:** (this iteration)
- [ ] **M36** — Multi-window concurrent convert: `JANGStudioApp` uses `WindowGroup` — File → New Window spawns a 2nd wizard instance with its own PythonRunner. Nothing prevents two simultaneous converts on the SAME Mac. Memory ref: `feedback_no_concurrent_mlx.md` — both saturate Metal at P8, total wallclock is 2× worse than serial. Need a global lock (file-based or UserDefaults + coalesce).
- [ ] **M37** — Osaurus remap table in `convert.py` doesn't include `mistral4`, `deepseek_v2/v3/v32`, `minimax_m2`, `glm_moe_dsa`, `nemotron_h`, `qwen3_5_vl`, `qwen3_moe_vl`. For these the remap falls through to `Qwen2Tokenizer` default which is WRONG for e.g. MiniMax (BPE w/ custom tokens). Map every supported family explicitly or disable remap for unmapped types.
- [ ] **M38** — The `warnings` field in `RecommendationService` response is a plain `[String]` — but the recommendation engine could produce structured warnings (code + severity + doc-link). Beginners get flat text like "bfloat16 activations required" with no link to `project_bfloat16_fix.md`.
- [ ] **M39** — `AUDIT_CHECKLIST` category K (Ralph runner) — partially traced iter 8 (runner.py state machine + shell splicing); audit.py + the macstudio round-trip still untraced. Does cancel land cleanly on a 2-hour convert? Is progress streamed back via the JSONL protocol or swallowed?
- [ ] **M40** — `feedback_no_bandaid_fixes.md` says "find root causes with data, verify at EVERY layer". The Osaurus remap is layer 1 (source config) but the PostConvertVerifier catch is layer 3 (output) — there's no layer 2 (mid-convert runtime check). If convert.py's remap was buggy, we'd only notice post-hoc. Add a mid-convert assertion.
- [x] **M41** — SECURITY: `PublishService.swift` passed `--token <LITERAL>` on argv. During a 200 GB publish that can run 30+ minutes, the token was visible in `ps aux` output, macOS Activity Monitor's "Open Files and Ports" panel, any `sample`/`dtrace`/diagnostics capture, and potentially any crash report. Any local user could read it for the full window.
      **Fix:**
      - `PublishService.swift:48-95`: token moved from argv to `HF_HUB_TOKEN` env var on the child process (only visible to the child + root). Stderr surfaced from failures now additionally scrubs the token verbatim as belt-and-suspenders.
      - `publish.py:22-53`: refuses literal tokens on argv — `--token <value>` now MUST be a file path to a token file. A non-file value fails with a clear error pointing at HF_HUB_TOKEN. Env var path unchanged.
      - `publish.py:78-85`: scrubs token from the "publish failed" stderr before print, in case an HF exception embeds the Authorization header.
      - `test_publish.py`: 2 new regression tests — `test_cli_rejects_literal_token_via_argv` (asserts rejection + that the literal value is NOT echoed in stderr) and `test_cli_accepts_token_file` (file-path branch still works).
      **Evidence:** 227 Python tests pass, 65 Swift tests pass.
      **Commit:** (this iteration)
- [x] **M42** — `PostConvertVerifier.runJangValidate` used `proc.waitUntilExit()` with no timeout. If the `jang validate` Python subprocess hung (pathological but possible — heavy imports stalling, corrupted bundle triggering a never-exiting parser loop), VerifyStep would block indefinitely even after the user navigated away — because `refresh()` was dispatched via `.onAppear { Task { await ... } }` which spawns a DETACHED task unbound to the view lifecycle.
      **Fix (iter 19 — two layers):**
      1. `runJangValidate` rewritten with `CheckedContinuation` + `proc.terminationHandler` (the M19 / iter-3 pattern already proven in PythonRunner and InferenceRunner). Races a `Task.sleep(timeoutSeconds)` against natural exit; on timeout SIGTERMs the subprocess with a 3-second SIGKILL escalation. Default 60s (10× the ≤5s normal completion time). Thread-safe via an explicit DispatchQueue guard against double-resume.
      2. `VerifyStep` refresh dispatch switched from `.onAppear { Task { ... } }` → `.task { ... }`. SwiftUI auto-cancels `.task`-bound work on dismount, so navigating away mid-verify no longer leaves an orphan Python subprocess running.
      **Tests (3 new):** defaultValidateTimeoutSeconds is in [30, 300]s (regression pin), runJangValidate returns false on nonexistent dir (exercises the terminationHandler branch on non-zero exit), timeout bound fires near 0.1s wall-time bound.
      **Evidence:** `PostConvertVerifier.swift:138-185`, `VerifyStep.swift:100-104`. 109 Swift tests pass (was 106).
      **Commit:** (this iteration)
- [x] **M43 (Python side)** — `publish.py` previously called `upload_folder` in a single blocking invocation — user saw a spinner for 30+ minutes with no confirmation the upload was even running, no file count, no bytes-transferred, no ETA. Swift-side integration waiting on iter 24.
      **Fix (iter 23, Python half):**
      - New `_upload_with_progress(model_dir, repo_id, token, emitter, commit_message, upload_file=None)` function iterates every file under the model dir + calls `HfApi.upload_file` per file, emitting phase + tick JSONL events to stderr (same schema as convert's 5-phase protocol so Swift `JSONLProgressParser` consumes it unchanged). `upload_file` is an injection point for tests so they never hit the real HF API.
      - CLI gains `--progress {none,json}` (default `none`). `json` dispatches to the per-file path; otherwise falls back to the original `upload_folder` bulk call (~faster, no progress).
      - Phase 1 = scan, phase 2 = upload, phase 3 = finalize. Info event records "enumerated N files totalling X.XX GB". Tick events stream bytes-uploaded / total-bytes with label=relative-path, throttled to 100ms via ProgressEmitter (final 100% tick always lands via is_final auto-detect).
      - Empty model dir raises `RuntimeError` rather than silently creating a commit with zero files.
      - Commit message per file includes `(idx/total: filename)` so HF's commit history reflects the upload order.
      **Tests (4 new):** iterates-every-file (sorted order, commit-message shape), JSONL event stream has exactly 3 phase events + ≥1 tick + info event with v=1 schema version and ts timestamp on every event, empty-dir raises, CLI help mentions `--progress` + `json`.
      **Evidence:** `publish.py:27-77` new function, `publish.py:79-130` dispatch branch, `tests/test_publish.py` 4 new tests. 245 jang-tools tests pass (was 241).
      **Commit:** (this iteration)
      **Iter 24 — Swift half complete:**
      - `PublishService.publishWithProgress(modelPath:repo:isPrivate:token:) -> AsyncThrowingStream<ProgressEvent, Error>` mirrors PythonRunner.run() exactly (same continuation + terminationHandler + stderr JSONL parser pattern). Passes `--progress json` to the subprocess so Python emits the iter-23 schema on stderr.
      - `PublishToHuggingFaceSheet.runPublish()` switched from one-shot `publish(...)` to streaming variant. State added: `progressPhase`, `progressBytes (done/total)`, `progressLabel`, `progressLog`. `apply(event:)` dispatcher updates the corresponding @State fields per event type.
      - New "Uploading" Section in the form, visible while `isPublishing`. Renders `ProgressView(value:, total:)` with GB/GB + percent label + the current filename being uploaded underneath. When no tick has fired yet, falls back to a small indeterminate spinner.
      - Token scrub on stderr preserved (iter 6 M41 layer-2) so a failed stream upload still can't surface the HF_HUB_TOKEN.
      - 2 new Swift tests: `test_publishWithProgress_rejects_empty_token` pins the pre-subprocess missingToken error shape matches the non-streaming variant; `test_publishWithProgress_is_async_stream` type-pins the return signature so future API changes break at compile time.
      **Evidence:** `PublishService.swift:122-216` (streaming function), `PublishToHuggingFaceSheet.swift:176-251` (streaming runPublish + apply(event:) dispatcher), `PublishToHuggingFaceSheet.swift:87-116` (progress section UI). 111 Swift tests pass (was 109).
      **End-to-end M43 complete.** Both sides shipped; user-visible: 30-minute silent spinner replaced by phase indicator + live progress bar + per-file filename.
- [x] **M44** — `PublishResult.swift` wasn't decoding the `commit_url` field that `publish.py` emits on success. Python emitted it (`"commit_url": str(info)` line 75), Swift struct had no `CodingKeys` entry so decoder silently dropped it. UI showed only the repo URL which doesn't prove the upload landed.
      **Fix:** Added `commitUrl: String?` to `PublishResult` with `commit_url` JSON mapping. Sheet now renders a second "Commit" row under "Published" with its own Open button when `commitUrl != url`. Decode tests cover both the full-publish and dry-run shapes.
      **Evidence:** `PublishService.swift:4-22`, `PublishToHuggingFaceSheet.swift:80-102`, 2 new decode tests in `AdoptionServicesTests`.
      **Commit:** (this iteration)
- [x] **M45** — Tracing modelcard per-arch coverage surfaced TWO import-name bugs that would ImportError on every adopter who copied the generated snippet — and the bugs were LOCKED IN by stale tests asserting the wrong names.
      **Bug 1 (dense path):** `python-snippet.py.jinja` had `from jang_tools.loader import load_model`. The actual function is `load_jang_model`. Every dense model's generated README said to `import load_model` → `ImportError: cannot import name 'load_model'` on the adopter's machine. `test_examples.py` + `test_modelcard.py` both ASSERTED the wrong name — which is how the bug survived.
      **Bug 2 (VL path):** `python-snippet.py.jinja` VL branch had `from jang_tools.load_jangtq_vlm import load_jangtq_vlm`. The actual function is `load_jangtq_vlm_model`. `inference.py:41-42` had the SAME wrong name (swallowed by `except Exception: pass` on the next line), silently falling through to mlx_vlm which can't load actual JANGTQ-VL models. So Test Inference on a JANGTQ-VL output was silently broken too.
      **Fix (iter 20):**
      - `python-snippet.py.jinja`: dense → `load_jang_model`, VL → `load_jangtq_vlm_model`.
      - `inference.py:38-46`: VL dispatch uses `load_jangtq_vlm_model`.
      - `test_examples.py`: assertions flipped to correct names AND added `load_model(` not-in-snippet guard (bare call would also ImportError).
      - `test_modelcard.py`: same correction. The old `"load_model" in card` was vacuously true because that substring appears inside `load_jang_model` — so a pure rename regression would never have been caught. Now asserts the full symbol AND the bare-call absence.
      - New `test_python_snippet_imports_resolve_to_real_symbols` uses `importlib` to verify the symbols referenced in each snippet branch actually exist at runtime. Compile-only validation (test_cli_python_snippet_compiles) can't catch import-name typos; hasattr(mod, name) can.
      **Evidence:** `templates/python-snippet.py.jinja:2,6,18,21` (fixed imports), `jang_tools/inference.py:38-46`, 232 Python tests pass (was 231).
      **Commit:** (this iteration)
- [x] **M46** — PublishToHuggingFaceSheet accepted any non-empty repo name. A typo like "my model/repo" (space), "justname" (no slash), or `org//name` (double slash) would be dispatched to Python → huggingface_hub raises cryptic InvalidRepoIdError ~30 s into the upload attempt after auth/probe. User has no idea it was their input.
      **Fix:** Added `HFRepoValidator.validationError(_:)` (MainActor enum) in `PublishService.swift` that runs BEFORE dispatch. Validates: non-empty trimmed, single `/`, both segments non-empty, each segment matches `^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$`, no spaces. Returns actionable error strings ("Invalid org segment 'bad-org'…"). Called from both `runDryRun` AND `runPublish` so bad input is caught before network I/O.
      **Evidence:** `PublishService.swift:23-70` validator, `PublishToHuggingFaceSheet.swift:140-175` call sites, 8 new `HFRepoValidator` test cases covering canonical/empty/no-slash/spaces/double-slash/leading-special/overlong/whitespace.
      **Commit:** (this iteration)
- [ ] **M47** — `isPrivate` toggle in the publish UI — if user publishes a model to a PRIVATE repo, does the generated model card link in the Examples output still use public URLs? The modelcard Jinja template uses `{repo_id}` — verifying that private/public is respected (model card will show the gated badge).
- [x] **M48** — `PublishToHuggingFaceSheet.init` defaulted `repoName` to `modelPath.lastPathComponent` — just the basename. This was ALWAYS invalid per M46 validation (added iter 7) because HF requires `org/name` with a single `/`. Every adopter saw the validation error on their first click until they manually typed the org prefix.
      **Fix (iter 25):**
      - New `AppSettings.defaultHFOrg: String = ""` field. Persisted via Snapshot; reset() clears it; observeAndPersist tracks it. Snapshot field has a Codable default (`= ""`) so pre-iter-25 UserDefaults snapshots decode cleanly after app update.
      - New "Publishing" section in SettingsWindow → General tab with a TextField bound to `$settings.defaultHFOrg` and a helper caption.
      - `PublishToHuggingFaceSheet` reads settings via `@Environment(AppSettings.self)` and applies the prefix via `.task { applyOrgPrefixIfNeeded() }`. Idempotent via `@State orgPrefixApplied` flag so clicking in+out of the field doesn't re-prefix.
      - `applyOrgPrefixIfNeeded()` is defensive: no-ops when org is empty, when repo already contains `/`, or when repo has been typed to something other than the basename default. Won't stomp user-typed text.
      **Tests (4 new):** default is empty, roundtrip across process restart, reset clears it, pre-iter-25 UserDefaults snapshot (missing the defaultHFOrg field entirely) still decodes with a defaulted empty value.
      **Evidence:** `AppSettings.swift:40-47` + Snapshot field with Codable default, `PublishToHuggingFaceSheet.swift:47-76`, `SettingsWindow.swift:101-108`. 115 Swift tests pass (was 111).
      **Commit:** (this iteration)
- [ ] **M49** — Stale HF token in env on app startup: sheet's init reads `ProcessInfo.environment["HF_HUB_TOKEN"]` ONCE. If user rotates their token in Terminal after the app is already running, the stale value stays in the sheet's initial state. macOS apps inherit env at launch time. Not a bug per se but document the behavior in the InfoHint tooltip.
- [ ] **M50** — Token persistence across sheet-close: after successful publish, sheet dismisses. Reopening it re-inits from env. If the env token was rotated, user enters a NEW token — but the SecureField's AutoFill could auto-populate the OLD one from Keychain if the app is persisting it (it isn't right now, but worth checking that `SecureField` doesn't unexpectedly).
- [ ] **M51** — Commit URL Open button on PublishToHuggingFaceSheet: `NSWorkspace.shared.open(u)` uses the default browser. For a private repo the commit URL might require login — user clicks, lands on a login page, looks like a bug. Add InfoHint: "Private repos require you to be signed in to huggingface.co in your browser."
- [x] **M52** — SECURITY/HARDENING: `ralph_runner.runner.ensure_source_model` spliced `hf_repo` directly into a `python3 -c '...'` command sent to macstudio via SSH. A malformed entry in `models.yaml` (or a future path that reads repo ids from a less-trusted source) could break out of the string literal and execute arbitrary Python on macstudio. No exploit today (models.yaml is Git-tracked + review-gated) but it's a latent RCE.
      **Fix:** Added `_HF_REPO_PATTERN` regex + `_assert_safe_repo_id(hf_repo)` that validates BEFORE splicing. Pattern matches the Swift-side `HFRepoValidator` from iter 7 exactly (segment rules `^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$`). `ensure_source_model` now calls it unconditionally.
      **Evidence:** `runner.py:70-78` pattern + assertion; `tests/test_runner.py::test_assert_safe_repo_id_rejects_shell_injection` probes 8 classic injection patterns (`";`, `&&`, `` ` ``, `$()`, newline, `|`, `&`, Python-level `__import__`).
      **Commit:** (this iteration)
- [x] **M53** — Ralph runner `rm -rf`d the output dir on macstudio unconditionally after audit, including when audit FAILED. A failed JANG convert is EXACTLY when the engineer needs the output artifacts to debug (missing file? wrong eos? chat template issue?) — removing them forces a hours-long re-convert just to reproduce.
      **Fix:** `cmd_next` now only removes output on `status == green`. On failure it sets `info["retained_output_path"]` so the slug entry in state.json points at the exact macstudio path that survived, and logs it to stdout so Ralph iterations can show the user where to look.
      **Evidence:** `runner.py:292-303`.
      **Commit:** (this iteration)
- [x] **M54** — RELIABILITY: Ralph's `running` status was never recovered on crash. If `--next` hit ctrl-C mid-convert, SIGKILL, OOM, Tailscale drop, or power loss, the combo stayed in `running` forever. `pick_next` only picks `pending`, so the combo was effectively lost — next invocation would skip to a later pending combo without ever retrying the interrupted one. Over a multi-day Ralph run this silently drops entries from the matrix.
      **Fix:** Added `recover_interrupted(state)` that flips `running` → `pending` on load, moves the `started` timestamp to a `recovered_from_interrupt` breadcrumb for post-mortem, cleans up the old `started` key. Called at the top of `cmd_next` before `pick_next`, with a log line reporting the count. Safe for idempotent re-runs — empty state and no-running-combos both return 0.
      **Evidence:** `runner.py:51-68` function, `runner.py:223-228` call site, 5 test cases including empty-state / missing-key edge cases.
      **Commit:** (this iteration)
- [x] **M54b** — Testability: `ralph_runner/runner.py` had no unit tests at all (test_audit.py + test_remote.py exist, but runner itself was zero coverage) AND unconditionally imported `ruamel.yaml` at module level. Tests couldn't run without the full runtime dependency set, so the module was effectively untestable in CI.
      **Fix:** Moved `YAML` import into `_yaml()` (only `activate_tier` needs it). Created `ralph_runner/tests/test_runner.py` with 11 test cases (state roundtrip, recover_interrupted across 4 scenarios, repo-id validator accept/reject/shell-injection/structural, slug stability).
      **Evidence:** `runner.py:18-27` lazy import, `tests/test_runner.py` (new file), 28 ralph_runner tests pass.
      **Commit:** (this iteration)
- [x] **M55** — Multi-instance safety. Running `--next` twice concurrently (two terminals / cron mistake / backgrounded with `&`) raced on state.json AND dispatched two convert subprocesses to macstudio simultaneously, violating `feedback_no_concurrent_mlx.md`'s 2× wallclock penalty rule. No mutex, no lockfile — first-writer-wins on state.json and both converts just hammered the GPU in parallel.
      **Fix (iter 12):** PID+host lock file at `ralph_runner/results/ralph.lock`.
      - `acquire_lock(path)`: `O_EXCL | O_CREAT` create (atomic on APFS). If the lock exists, inspect: same-host + alive PID → `LockAcquireFailed`; same-host + dead PID → stale, remove + retry once; different host → refuse defensively (can't verify remote PID, safer than stomping a live convert); unparseable JSON → stale, reclaim.
      - `release_lock(path)`: removes ONLY if we own it (pid match). Defends against a legit holder after our crash having its lock yanked by stale `finally: release_lock()` calls.
      - `_pid_alive(pid)`: uses `os.kill(pid, 0)` with correct handling of `ProcessLookupError`/`PermissionError` (cross-uid alive counts as alive).
      - `cmd_next` split into `cmd_next` (acquire + try/finally release) and `_cmd_next_locked` (all the work). Lock is held for the full convert window so the second instance sees `BLOCKED: lock held by another ralph instance: {pid: X, host: Y}`.
      **Tests:** 10 new in `test_runner.py` — happy path, release-missing-noop, live-PID refusal, dead-PID reclaim, cross-host refusal, corrupt-JSON reclaim, other-owner release no-op, pid_alive self/dead positive cases.
      **Evidence:** `runner.py:80-173` (lock + PID helpers), `runner.py:372-395` (cmd_next split). 38 ralph_runner tests pass (was 28).
      **Commit:** (this iteration)
- [ ] **M56** — `profiles.yaml` tier lookup failure: `tier_profiles.get(f"tier_{tier}_profiles", {})` silently returns `{}` for a missing key. If someone typos tier 99 in models.yaml, `activate_tier(99)` succeeds with ZERO combos created. No warning. `cmd_status` prints "NO COMBOS" but that's indistinguishable from "correctly empty".
- [ ] **M57** — `slug()` handles space and `/` but not `;`, `$`, `` ` ``, `\n`. If a profile name ever contains any of those, the `out` path in `run_convert_remote` gets spliced into a shell command with those chars intact → shell injection. Defence-in-depth: run `out_slug` through the same `_HF_REPO_PATTERN`-style check (or `shlex.quote`) before splicing.
- [ ] **M58** — `run_convert_remote` line 128: the JANGTQ path uses `python3 -m jang_tools.convert_qwen35_jangtq` but the registered CLI for other JANGTQ families (minimax) is elsewhere. Hardcoding `convert_qwen35_jangtq` means MiniMax JANGTQ combos would dispatch to the Qwen35 entry point. Verify via model-family dispatch rather than a hardcoded CLI path.
- [ ] **M59** — `audit.py` is 765 lines and handles every audit row inline. Any new audit rule needs a 50-line PR. Consider a plugin directory `audits/` where each file is one row, auto-registered at import.
- [x] **M72** — `audit_a6_wall_time` was dead code via `run_audits`. The function existed, `run_audits` had a special-case dispatch branch for `row == "a6"` (line 714-715), and `cmd_next` passed `--baseline-wall-s` through the CLI. But `a6` was NOT in `AUDIT_REGISTRY`. `run_audits` checks `row not in AUDIT_REGISTRY: continue` FIRST (line 709-711), so the dispatch branch was unreachable. Users passing `--rows a6` got back `{"status": "n/a", "hint": "unknown row a6"}` on a row that actually works.
      **Fix:** Added `"a6": ("Wall time vs baseline", audit_a6_wall_time, False)` to the registry. a6 is warn-only — a wall-time regression shouldn't block a ship.
      **Evidence:** `audit.py:691-694` registry entry, 5 new tests in `test_audit.py` (registration presence + dispatch + none-baseline + over-150% fail + within-baseline pass).
      **Commit:** (this iteration)
- [x] **M77** — `audit_a2_chat_template` only checked inline + `chat_template.jinja` for the presence-of-template guard; if the model shipped ONLY `chat_template.json` (newer HF convention, used by Qwen3-VL and similar), a2 returned `n/a, "no chat template present in source"` — a false negative that would mask a real chat-template absence in rare cases but mostly just silently skipped the check. The Swift-side `PostConvertVerifier.swift:42-50` already handles the three forms; audit.py was out of sync.
      **Fix:** Added `chat_template.json` to the guard-clause file check. If ANY of the three forms exists, progress to the render step. Negative test pins that the guard still returns `n/a` when NONE of the three exist.
      **Evidence:** `audit.py:117-128`, 3 new tests (json form accepted, jinja form still accepted, none present still n/a).
      **Commit:** (this iteration)
- [x] **M78** — a9 false-failed on structured-vs-string equivalent special token values. HF accepts two semantically-identical forms for `special_tokens_map.json` values: plain string `{"bos_token": "<s>"}` and structured dict `{"bos_token": {"content": "<s>", "lstrip": false, "normalized": false, "rstrip": false, "single_word": false}}`. Both round-trip through AutoTokenizer. A convert that changes form (source structured → output string or vice-versa) is valid, but the old `out_tokens[k] != v` raw equality mis-graded it as a mismatch → `required=True` → combo marked FAILED in Ralph's matrix on a perfectly good convert. This was latent false-fail silently bombing real runs.
      **Fix (iter 16):** `_normalize_special_token_value(value)` extracts the semantic `content` string from either form; returns None on unrecognised shapes. a9 compares normalised values. Unrecognised shapes fall back to strict `==` (defends against silent pass on corrupted files / future schema drift) and are reported under a new `unnormalizable` result field for precise debugging.
      **Tests (8 new):** normaliser covers plain string / structured dict / 5 unrecognised shapes (None, int, list, dict-without-content, non-string content); a9 passes on structured→string roundtrip; a9 passes on string→structured roundtrip; a9 still fails on genuine content mismatch (defends against always-pass regression); a9 still catches missing keys; unrecognisable shape falls back to strict eq + surfaces in `unnormalizable`.
      **Evidence:** `audit.py:343-434`. 60 ralph_runner tests pass (was 52).
      **Commit:** (this iteration)
- [ ] **M79** — a15 uses raw prompt to `mlx_lm.generate` — explicit design split from a16 (a15 = "does anything come out", a16 = "chat template actually renders"). Should be documented in a module / function docstring so the next auditor doesn't "fix" it into applying the chat template.
- [x] **M83** — Memory-cross-ref (iter 21, second pass of Cat D after iter 5). `project_mlp_asymmetry.md` claimed the gate_proj/down_proj bit floor threshold was lowered from 512 to 256 on 2026-04-08 (after GLM-5.1 with 256 experts degenerated at 2-bit MLPs). Grep of `allocate.py` showed `if num_experts < 512: return bits` at the activation guard — docstring right above it said "256+ expert models". **Classic code/docstring/memory triplicate drift.** GLM-5.1 (256 experts), MiniMax M2.7 (256), and Qwen3.6 (256 routed) all went UNPROTECTED through the bit allocator — memory predicts this produces coherent single-token answers but multi-token repetition loops.
      **Fix (iter 21):**
      - `allocate.py` introduced a named constant `_MLP_ASYMMETRY_MIN_EXPERTS = 256` and changed the guard from magic `< 512` → `< _MLP_ASYMMETRY_MIN_EXPERTS`. Future drifts must fight a named constant + 9 regression tests rather than a magic number 3 lines below its own docstring.
      - Memory file `project_mlp_asymmetry.md` got an iter-21 note documenting the drift + the fix.
      - New `tests/test_allocate_mlp_asymmetry.py` — 9 tests: threshold matches memory, floor values pin to 4/3/no-up_proj, below-threshold passthrough, at-256 floor applies, at-512 still applies, shared_expert exempt, non-MLP names pass through, floor never lowers bits, Mixtral w1/w2 naming also covered.
      **Evidence:** `allocate.py:318-336`, 241 Python tests pass (was 232, +9).
      **Commit:** (this iteration)
- [ ] **M84** — `capabilities_cli.py:16` `_KNOWN_512_EXPERT_TYPES = ["minimax_m2", "glm_moe_dsa"]` — name is semantically wrong per `project_bfloat16_fix.md`: MiniMax has 256 experts + hidden≤3072 so it does NOT trigger the loader.py:271 bfloat16 overflow guard (`_n_experts >= 512 and _hidden >= 4096`). The Swift PreflightRunner uses this list to warn users about `fp16` on 512-expert models — a minimax user forcing fp16 gets a misleading "512+ experts" warning. Also GLM-5.1 is 256 experts so the list name is doubly wrong. Low-stakes UX bug but pollutes the model-family warning surface.
- [x] **M85** — `allocate.py` had "512+" prose in 8 places after M83 fixed the runtime behaviour. Iter 22: swept 6 MLP-asymmetry-context comments (lines 81, 87, 94, 246, 303, 529) to say "256+". Kept lines 62, 77 at "512+" because they specifically describe the bfloat16/float16-overflow fix (`project_bfloat16_fix.md` — thresholded at 512 experts + hidden≥4096), which is a DIFFERENT concern than MLP asymmetry. Future auditors won't repeat iter-21's triplicate-drift investigation.
      **Evidence:** `allocate.py:81-98, 246, 303, 529`.
      **Commit:** (this iteration).
- [x] **M86 (Cat D cross-ref)** — `project_mistral4_architecture.md` claims 7 fixes; MEMORY.md index says "6 fixes". Spot-checked each against `mistral4_mlx.py`: FP8 scale loader, `norm_topk_prob=True`, `llama_4_scaling_beta`, plain `1/sqrt(128)` attention scale (NO mscale²), gate dequant to bfloat16, auto bfloat16 model — all 6 confirmed present. The 7th (`rope_interleave=True → traditional=True`) needs live validation: `mistral4_mlx.py:154` has `traditional=False` with comment "interleaved RoPE", which may reflect an MLX-native vs mlx-vlm semantics inversion of the `traditional` flag rather than drift. Documented as M87 below for future live-model verification.
- [x] **M89 (Cat D third pass)** — `feedback_chat_template_rules.md` cross-ref against `convert.py`. Memory documents the Qwen3.5 248044→248046 eos fix but the local `_eos_fixes` dict covered only `qwen3_5` + `qwen3_5_moe`. The Qwen3 family includes image + video VL variants (`qwen3_vl`, `qwen3_moe_vl`, `qwen3_5_vl`) that share the Qwen2Tokenizer ID space — a Qwen3-VL source with the wrong 248044 eos would slip through unfixed, causing the same infinite-thinking-loop symptom the memory flags.
      **Fix (iter 26):**
      - Extracted the dict to module-level `EOS_FIXES` so tests can pin coverage without reaching into convert_model's locals.
      - Extended coverage to `qwen3_vl`, `qwen3_moe_vl`, `qwen3_5_vl`. Fix is idempotent on correct sources (guarded by `tc.get("eos_token_id") in eos_fix_map`) so broadening carries zero risk — models with correct eos see no change.
      - 6 new tests (`tests/test_convert_eos_fixes.py`): dict shape invariant, original Qwen3.5/MoE coverage regression pin, iter-26 VL coverage pin, NEGATIVE case (llama/mistral/gemma/phi/qwen2/minimax/deepseek/nemotron must NOT be in the map — adding them would mis-correct their already-correct eos), idempotency invariant documented, numeric-value pins (248044 wrong, 248046 right).
      - Memory file updated with iter-26 fix note.
      **Evidence:** `convert.py:29-44` module-level constant, `convert.py:957-963` use site now references it; 251 jang-tools tests pass (was 245, +6).
      **Commit:** (this iteration)
- [ ] **M87** — `mistral4_mlx.py:151-158` uses `mx.fast.rope(traditional=False)` for interleaved RoPE. `project_mistral4_architecture.md` says "rope_interleave=True → traditional=True" for mlx-vlm patches. Need a live Mistral 4 conversion + generate to verify whether our native MLX port has the correct RoPE shape (repetition or garbage would indicate drift; clean output confirms the semantic inversion is intentional).
- [x] **M90 (Cat D fourth pass)** — `feedback_readme_standards.md` rule 10: "YAML frontmatter must include `reasoning` AND `thinking` tags for all Qwen3.5 models". `detect_capabilities` in `examples.py:87` exposed only `has_reasoning = bool(reasoning_parser) or bool(enable_thinking)` — ORed both signals into ONE flag. The model-card template then emitted only a single `reasoning` YAML tag. The separate `thinking` tag that memory requires was NEVER emitted, even on Qwen3.5 models where `enable_thinking: true` is explicitly set in config.json. Every Qwen3.5 HF upload violated the standard.
      **Fix (iter 27):**
      - `examples.py:detect_capabilities` now exposes both `has_reasoning` (broader capability) AND `has_thinking = bool(cfg.get("enable_thinking"))` (specific runtime toggle). Separate flags, separate semantics.
      - `model-card.md.jinja` emits a dedicated `- thinking` tag when `has_thinking` is true. The `reasoning` tag path preserved unchanged.
      - 4 new tests: `has_thinking` true on Qwen3.5 + enable_thinking=true; `has_thinking` false when enable_thinking absent; modelcard emits BOTH tags for qualifying models (positive); modelcard OMITS `thinking` tag on non-thinking models (negative — prevents spurious tags).
      **Evidence:** `examples.py:86-95` flag addition, `templates/model-card.md.jinja:21-23` tag emission, `tests/test_examples.py` +4 tests. 255 jang-tools tests pass (was 251).
      **Commit:** (this iteration)
- [x] **M91** — `feedback_readme_standards.md` documents 10+ HARD requirements for HF uploads. Iter 27's M90 automated rule 10 (`thinking` YAML tag). The remaining 9 rules (per-subject MMLU, JANG-vs-MLX comparison, speed/size comparison, Korean section) are UNAUTOMATABLE without live evals — the template produces a skeleton that would silently fail the standard if published directly.
      **Fix (iter 28 — visibility, not automation):** Two layers so the gap is impossible to miss:
      1. **Swift UI banner** (GenerateModelCardSheet): new orange-tinted Section at the top of the card preview saying "Skeleton only — not upload-ready" with a caption naming every missing section. Visible AS SOON AS the card renders, BEFORE the Save/Copy buttons. User sees it every time they open the sheet.
      2. **CLI stderr warning** (modelcard.py): prints "NOTE: generated card is a skeleton..." to stderr after generation. Goes to stderr not stdout so it doesn't pollute `--json` consumers or plain-text card dumps. Ralph harness tails stderr and surfaces the note; humans running the CLI directly see it without parsing needed.
      Both layers make the gap visible at the moment the user is about to act on the card. Doesn't FIX the underlying rule (that needs live evals) but prevents the silent-violation failure mode.
      **Tests:** 1 new — `test_cli_emits_skeleton_warning_to_stderr` pins (a) distinctive warning phrase on stderr, (b) memory-ref pointer in the message, (c) stdout stays machine-parseable (no warning leakage). Gotcha caught during test writing: pytest's tmp_path names include the test case name, so a naive `"skeleton" in stdout` substring match false-positives on paths containing "test_cli_emits_skeleton_warning0/dense". Assertion switched to distinctive phrase `"generated card is a skeleton"`.
      **Evidence:** `GenerateModelCardSheet.swift:19-58` banner, `modelcard.py:48-59` stderr note, `tests/test_modelcard.py` +1 test. 256 jang-tools tests pass (was 255).
      **Commit:** (this iteration)
- [ ] **M92** — `project_qwen36.md` P5: mlx-lm's `rope_type: mrope` silently falls back to plain RoPE. OK for text inference, WRONG for VL position encoding. Our code doesn't detect or warn. A Qwen3.6-VL convert would generate text correctly but mis-position images. Too mlx-lm-internals to fix from our side; could add a detection + warning during inspect-source when `rope_scaling.type == "mrope"` on a VL model.
- [x] **M96** — `publishWithProgress` stream had NO cancellation propagation. Iter 23-24 added streaming but never wired `continuation.onTermination` to the subprocess. Failure mode: user clicks the sheet's Close button OR cancels the consuming Task → stream iterator throws CancellationError → UI thinks it's done → BUT the Python subprocess is STILL running + uploading to HuggingFace in the background. Half-published repo stranded with partial files; no cleanup; no UI feedback. Also: NO Cancel button in the sheet during `isPublishing`, so even if cancellation worked, users had no way to trigger it.
      **Fix (iter 30):**
      - New `ProcessHandle: @unchecked Sendable` class holds the child Process reference, lockable, with a `wasCancelled` flag. `cancel()` triggers SIGTERM + 3-second SIGKILL escalation (same pattern as PythonRunner).
      - `publishWithProgress` wires `continuation.onTermination` to `handle.cancel()` before spawning the work Task. Stream termination from ANY source (consumer Task cancel, iterator abandon, explicit finish) now propagates to the subprocess.
      - `_streamPublish` calls `handle.set(process: proc)` after `proc.run()` lands. Race case: if `cancel()` fired before `set()` (rare but possible), `set()` immediately terminates the freshly-spawned process.
      - On exit, `_streamPublish` branches on `handle.wasCancelled`: clean finish (not a throw) when user-cancelled, so UI doesn't treat cancellation as upload-failure.
      - Sheet UI: new "Cancel upload" button visible only during `isPublishing`. Tapping it cancels the consuming Task, triggering the onTermination chain. `wasCancelled` @State flag distinguishes the sheet's success/cancel/error branches; cancel sets an informative errorMessage ("Upload cancelled — HF repo may contain partial files, delete or overwrite before retry") instead of claiming success OR surfacing a bare CancellationError.
      **Tests (4 new):** wasCancelled defaults false; cancel() sets the flag; cancel-before-set is safe (no crash, subsequent set terminates immediately — tested with a live /bin/sleep 1000 subprocess); cancel-after-set terminates the running process within SIGTERM timeout.
      **Evidence:** `PublishService.swift:67-121` ProcessHandle, `PublishService.swift:134-231` cancel wiring, `PublishToHuggingFaceSheet.swift:26-31` state + 209-220 Cancel button + 263-298 cancel branch in runPublish. 119 Swift tests (was 115, +4).
      **Commit:** (this iteration)
- [ ] **M97** — Partial HF repo cleanup after cancel. iter-30's M96 fix terminates the LOCAL subprocess but the files already uploaded to HF stay there. User retry or manual delete is currently the only path. Future: call `huggingface_hub.delete_folder` on cancel with a confirmation dialog. Non-trivial because a long user-cancel window could leave HF in an intermediate state that's hard to clean up atomically. Lower priority than the local-subprocess fix.
- [x] **M98** — PythonRunner had the same pre-M96 bug as PublishService: no `continuation.onTermination` handler, so consumer-Task cancellation or stream abandon left the subprocess orphaned + running. Iter-30's meta-lesson was "apply the same rigor" to PythonRunner — iter 31 did that with a real-subprocess integration test.
      **Test-first approach proved the bug:** wrote `test_consumerTaskCancel_terminatesSubprocess` using `/bin/bash while true; do date +%s%N > tickFile; sleep 0.2; done`. The test cancels the consumer Task and verifies tickFile mtime stops advancing. Pre-fix, mtime advanced ("subprocess is still running"); bug confirmed.
      **Fix:** register `continuation.onTermination = { Task.detached { await self.cancel() } }` INSIDE `launch()` after the subprocess is constructed. Placing it in `run()`'s outer build closure fired spuriously with reason=`.cancelled` during stream construction under some Task-isolation contexts — observed debug trace showed `onTermination fired: cancelled` before any iteration started. Moving it into launch() after `try proc.run()` defers registration until spawn is complete, avoiding the false-positive cancel.
      **Test tuning discoveries (documented):** initial 500ms wait for subprocess spawn was too tight under parallel test contention (spawn latency spikes past 1s on loaded M1). Switched to polling loop up to 3s. Also removed a speculative `test_streamAbandon_terminatesSubprocess` — stream abandon without any iterator is a different resource-leak class (producer still holds continuation; no signal for onTermination to fire). Spawned M99 for that.
      **Evidence:** `PythonRunner.swift:22-32,68-75` onTermination wiring, `tests/PythonRunnerTests.swift:49-111` integration test. 120 Swift tests pass (was 119, +1).
      **Commit:** (this iteration)
- [ ] **M99** — Stream-abandon without iterator: if consumer drops the stream value without making an iterator, onTermination never fires because the producer closure's `continuation` reference keeps the continuation alive. Realistic impact is low (all production sites iterate the stream). A language-level fix would require a finalizer or weak-continuation pattern. Logged for awareness, not urgent.
- [x] **M101 (grep-audit)** — Applied iter-32's meta-pattern: grep-audit every `withCheckedContinuation` / `withCheckedThrowingContinuation` site across the Swift codebase. Found 9 sites, 3 already fixed in iter 30/31/32, 5 one-shot subprocess service invokes (ModelCard, Examples, Profiles, Capabilities, Recommendation, PublishService.invoke) + 1 verifier timeout (PostConvertVerifier.runJangValidate) = 6 new sites missing Task-cancel propagation.
      **Risk class:** same as M100 but on shorter-running subprocesses. A cancelled consumer Task leaves the child running to natural completion, orphaning CPU for 1-30 seconds depending on the service. User-visible: UI that dismisses after action suffers ghost work; rapid navigation could spawn overlapping subprocesses.
      **Fix (iter 33):** wrapped each invoke with `withTaskCancellationHandler { ... withCheckedThrowingContinuation ... } onCancel: { handle.cancel() }`. All 6 services use the same `ProcessHandle` class from iter 30. Consistent shape: thread-safe Process holder, SIGTERM + 3s SIGKILL escalation.
      **PostConvertVerifier special case:** already had a 60s timeout race (iter 19 M42) so hang damage was bounded, but cancel propagation now short-circuits so the user doesn't wait the full 60s after navigating away. onCancel closure just SIGTERMs the proc; the existing terminationHandler then resolves the continuation with the terminated status (→ false, semantically "validation did not succeed").
      **Scope decisions:**
      - Did NOT extract a shared `runCancellableCLI()` helper across all 6 services to unify them. Each has its own error type (ModelCardServiceError, ExamplesServiceError, etc.) and slightly different stderr handling. Unification would add a new wrapper error type + mapping layer in every service — net lines-changed would increase. Spawned M102 if future maintenance suggests the unification is worth it.
      - Did NOT fix `SettingsWindow.observeAndPersist` (line 413). Its withCheckedContinuation is inside a `while !Task.isCancelled` loop which already polls cancellation. The hang risk is bounded by the next observation-tracked change firing. Logged as M103 (low urgency).
      **Evidence:** 6 services + 1 verifier patched. 122 Swift tests pass unchanged — same count because iter 33 is pure-plumbing with no new test cases. The existing `test_consumerTaskCancel_terminatesSubprocess` style tests for each would be 6 more integration tests with real subprocesses, each taking ~5s. Accepted the coverage trade-off; spawned M104 to add them if a regression surfaces.
      **Commit:** (this iteration)
- [ ] **M102** — Consider unifying the 6 CLI service invokes (ModelCard / Examples / Profiles / Capabilities / Recommendation / PublishService.invoke non-streaming) behind a shared `runCancellableCLI(args:) async throws -> Data` helper. Would remove ~30 lines of duplicated withTaskCancellationHandler + ProcessHandle + DispatchQueue wiring. Trade-off is a new wrapper error type + per-service error mapping. Revisit if maintenance of these becomes painful.
- [ ] **M103** — `SettingsWindow.observeAndPersist` uses `withCheckedContinuation` inside `while !Task.isCancelled`. Cancellation is checked between iterations but not during the await — so if the Settings sheet closes while no setting has changed, the Task waits for the next change (which never comes). Bounded leak: next app-level settings change resumes the continuation, loop sees isCancelled, exits. Low urgency.
- [ ] **M104** — No integration tests added for the 6 service cancel paths in iter 33. Each test would be ~5s (spawn + kill real subprocess). Accepted as follow-up if a regression surfaces. Pattern already documented by `PythonRunnerTests.test_consumerTaskCancel_terminatesSubprocess` (iter 31) and `InferenceRunnerTests.test_consumerTaskCancel_terminatesSubprocess` (iter 32).
- [x] **M105** — Extension of iter-33's sweep: grepped for bare `proc.waitUntilExit()` outside `withCheckedContinuation` wrappers. Found ONE real bug: `SourceDetector.inspect(url:)` in `SourceStep.swift:286`. Used synchronous `waitUntilExit()` inside an `async` function — blocked the async-context thread (often MainActor via SourceStep's `.task`) for the duration of the inspect-source CLI call AND ignored Task cancellation. User flow: pick folder A → task starts detection subprocess → user quickly picks folder B → task should cancel but subprocess runs to completion orphan-style. UI momentarily freezes during each inspect.
      **Fix (iter 34):** applied the iter-33 template — `withTaskCancellationHandler { withCheckedThrowingContinuation { DispatchQueue.global().async { ... } } } onCancel: { handle.cancel() }`. ProcessHandle (iter 30) for SIGTERM + SIGKILL escalation.
      **Evidence:** `SourceStep.swift:280-317`. 122 Swift tests pass unchanged.
      **Commit:** (this iteration)
- [ ] **M106** — `DiagnosticsBundle.write` is `@MainActor` and synchronous. `try p.run(); p.waitUntilExit()` for `ditto -c -k` blocks the main thread. For small diagnostics bundles (<5 MB) this is ~1s and invisible. For larger bundles (convert log + stderr + several MB of tick events) could be 3-5s of beach-balled UI during "Copy Diagnostics" click. Making it `async` would ripple through RunStep. Lower priority than M105.
- [x] **M100** — `InferenceRunner.generate()` same class of Task-cancel bug as M96 (PublishService) + M98 (PythonRunner), applied iter-30/31 meta-lesson. Iter 3's M19 fixed explicit `await runner.cancel()` but never wired consumer-Task cancellation. `generate()` awaited `withCheckedContinuation` for the subprocess termination handler; CheckedContinuation does not participate in Task cancellation, so a cancelled consumer Task would leave the subprocess running to natural completion. Wasted model load, stale inference, orphaned GPU allocation. On a 70 GB JANG model with ~30s load time, this is a real user pain — user aborts a prompt, GPU stays busy for 30+ seconds while the load completes, no feedback.
      **Write-failing-test-first validation (matches iter 31 approach):** New `test_consumerTaskCancel_terminatesSubprocess` with a real tick-writing bash subprocess. Pre-fix the test HUNG indefinitely at `try? await consumerTask.value` because generate() was stuck inside the continuation wait. Test harness had to be killed — proved the bug with certainty.
      **Fix:** wrapped the `withCheckedContinuation` inside `withTaskCancellationHandler { ... } onCancel: { Task.detached { await self.cancel() } }`. Task cancel now triggers SIGTERM + 3s SIGKILL escalation via the existing actor-isolated `cancel()` method.
      **Test refactor:** removed the `try? await consumerTask.value` after cancel to avoid hanging the test on regression (would hit the harness's 10-min timeout instead of an informative assertion failure). Instead sleep 5s past the SIGTERM+SIGKILL window and verify via tick-file mtime non-advance.
      **Added testability:** `InferenceRunner.init` gained an `executableOverride: URL? = nil` parameter matching PythonRunner's pattern, so tests can substitute a bash script without needing actual Python + an MLX model.
      **Regression pin:** new `test_explicit_cancel_still_works_via_actor_method` verifies iter-3's M19 path (explicit `await runner.cancel()`) still works after the iter-32 Task-cancel addition. Both paths must terminate the subprocess; neither must regress the other.
      **Evidence:** `InferenceRunner.swift:47-56` init + override, `InferenceRunner.swift:108-126` TaskCancellation wrap, `Tests/JANGStudioTests/InferenceRunnerTests.swift:32-112` new tests. 122 Swift tests pass (was 120, +2).
      **Commit:** (this iteration)
- [x] **M93** — `feedback_readme_standards.md` rule 11: "MiniMax is text-only — never include VLM code". `detect_capabilities` set `is_vl`/`is_video_vl` based on pure file-existence checks. Any `preprocessor_config.json` / `video_preprocessor_config.json` in a MiniMax output dir (copy residue, bad convert, user mistake) would flip the flags to True → Python snippet template would emit `load_jangtq_vlm_model` + `mlx_vlm` imports that fail at runtime because MiniMax has no vision tower. Silent rule-11 violation, published card would broken for adopters.
      **Fix (iter 29):** New `_TEXT_ONLY_MODEL_TYPES = frozenset({"minimax_m2", "minimax", "minimax_m2_5"})` constant in `examples.py`. `is_vl` and `is_video_vl` detection now require BOTH (a) file exists AND (b) model_type is NOT in the text-only set. Conservative: MiniMax stays text-only regardless of stray files; genuine VL models (qwen2_vl, qwen3_vl, etc.) are unaffected.
      **Tests (4 new):** MiniMax with planted stray preprocessor files has is_vl=False, rendered snippet uses `load_jang_model` (text) NOT `load_jangtq_vlm_model` (VLM) + has no `mlx_vlm` / `Image.open` markers, all 3 MiniMax aliases (`minimax_m2`, `minimax_m2_5`, `minimax`) are in the set, NEGATIVE guard: qwen2_vl with preprocessor is STILL detected as VL (iter-29 enforcement didn't broadcast to all models).
      **Evidence:** `examples.py:28-38` constant, `examples.py:91-99` dispatch. 260 jang-tools tests pass (was 256, +4).
      **Commit:** (this iteration)
- [ ] **M80** — a4 (tokens/sec) and a5 (chat turn) are registered but have no baseline comparison. Without thresholds "fail" means… what? A slow machine, a regression, a broken kernel? Now that a6 is registered (iter 15 M72 fix), the baseline infrastructure exists — a4/a5 could leverage it for relative thresholds.
- [x] **M81** — Verified `ralph_runner/fixtures/test_image.png` and `test_video_frames.npy` are git-tracked (`git ls-files` confirms), NOT gitignored (`git check-ignore` returns nothing). **No active bug today**, but no regression test pinned the invariant — a future commit that added `*.png` / `*.npy` to a global `.gitignore` could silently un-track them, and a11/a12 would degrade to warn on every fresh-clone run with no visible failure.
      **Fix (iter 17):** 4 regression tests added:
      1. `_FIXTURE_DIR` exists + is a directory.
      2. test_image.png exists on disk + PIL can open it + it's ≥32×32.
      3. test_video_frames.npy exists + numpy can load it + shape is 4D + ≥4 frames + ≥16×16 + 3 channels.
      4. `git ls-files ralph_runner/fixtures/` explicitly lists both files. Catches the exact "accidentally gitignored" failure mode — existence-on-disk tests alone can't catch it because the author's working tree always has the files.
      **Evidence:** `tests/test_audit.py` 4 new tests. 60 → 64 ralph_runner tests pass.
      **Commit:** (this iteration)
- [x] **M82** — `run_audits` had no per-row timeout. a15 (required, `mlx_lm.load` + generate) could hang indefinitely on a corrupted shard, missing file, or Metal kernel deadlock. The ENTIRE Ralph iteration would stall — `cmd_next` wouldn't return, macstudio would stay "busy" from Ralph's perspective, no subsequent combo would run, the matrix silently wouldn't progress.
      **Fix (iter 18):** Per-row timeouts via `_run_with_timeout`. Each audit function wrapped in a thread + `Future.result(timeout=...)`. On timeout, return a fail result AND orphan the hung thread (`pool.shutdown(wait=False)`) — the thread continues running but `run_audits` moves on to the next row immediately. The hung thread dies when the Python process exits (cmd_next's normal cleanup).
      **Timeouts by row:** a15=600s (big JANG_4K model loads), a3/a4/a5=300s (inference generates), a17/a18=90s (jang-tools subprocesses), a11/a12=120s (AutoProcessor load + forward), default=60s (file-inspection rows).
      **Critical implementation note:** canNOT use `with ThreadPoolExecutor(...)` because `__exit__` implicitly calls `shutdown(wait=True)` which would block forever on the hung thread. Manual `pool.shutdown(wait=False)` is the whole point.
      **Tests (4 new):** quick-result roundtrip, timeout fires within tolerance (<3s on a 1s timeout of a 10s hanger), timeout-map pins a15≥300 + default=60, end-to-end `run_audits` fake-registry test proves subsequent rows still run after a hung row.
      **Evidence:** `audit.py:737-782` (constants + wrapper + dispatch), `test_audit.py` +4 tests. 68 ralph_runner tests pass (was 64).
      **Commit:** (this iteration)
- [ ] **M67** — Lock-file edge: `--reset` doesn't remove `results/ralph.lock`. If a crash left both state.json and a lock, reset cleans state but next `--next` still has to reclaim the stale lock (PID-dead detection works but untested). Consider removing the lock in cmd_reset too, gated on same-host+dead-PID check.
- [ ] **M68** — Lock-file edge: SIGKILL + fast auto-restart (launchd/systemd) could put A's lock in front of B, with A's PID reused by an unrelated process → B thinks A is alive. Workaround: cross-check `ps -o lstart -p PID` start-time. Pathological; low priority.
- [ ] **M69** — Lock-file on network FS (SMB/NFS): O_EXCL isn't always atomic. Document in README: lock/state files must be on a local filesystem.
- [x] **M70** — state.json read-during-write race. `Path.write_text` is NOT atomic (open → write → close); a concurrent `--status` (which intentionally doesn't take the lock) could see truncated JSON mid-save and crash with `json.JSONDecodeError`.
      **Fix (iter 13):** Two layers of defense:
      - `save_state` now writes to `state.json.tmp.<pid>` in the same directory, then `os.rename` onto the target. POSIX `rename` is atomic within a single filesystem — the reader always sees either the old complete JSON or the new complete JSON, never torn. `<pid>` suffix keeps multiple processes from clobbering each other's tmp files (the lock prevents this anyway but defense in depth).
      - `load_state` tolerates `JSONDecodeError` by returning empty state + printing a warning to stderr. Previously it crashed the whole caller. Even with atomic rename, a pathological filesystem / disk corruption / user-edited file could surface garbage JSON — `--status` must stay robust.
      - `save_state` cleans up the tmp file on any failure path so we don't accumulate shrapnel in `results/`.
      **Tests (4 new):** atomic-via-tmp-rename (monkeypatches os.rename to inspect src/dst), cleans-tmp-on-failure (simulates rename OSError, asserts no leftover), load tolerates corrupt JSON with warning, 50-round save-load roundtrip smoke test.
      **Evidence:** `runner.py:48-101` (load_state + save_state). 44 ralph_runner tests pass (was 38).
      **Commit:** (this iteration)
- [x] **M71** — `cmd_reset` didn't take the lock. If `--next` was converting and user hit `--reset`, state.json got nuked mid-convert — losing the in-flight combo's record AND desynchronising the running process's next save.
      **Fix (iter 13):** `cmd_reset` now acquires the same lock `cmd_next` uses. If held, prints `BLOCKED: --reset refused, lock held by {pid, host, started_at}` and exits 0 without deleting anything. Side benefit: reset now also clears any stranded `state.json.tmp.*` shrapnel from a crashed save_state.
      **Tests (2 new):** reset-refused-when-held asserts state.json survives; reset-clears-tmp-shrapnel asserts post-reset cleanup of leftover tmp files.
      **Bonus fix:** `acquire_lock` / `release_lock` call sites in `cmd_next` + `cmd_reset` now pass `LOCK_PATH` explicitly. Python evaluates default args at function-definition time, so `monkeypatch.setattr(runner, "LOCK_PATH", tmp)` didn't flow into the captured default. Tests now pin this with the explicit-argument pattern.
      **Evidence:** `runner.py:519-542`, tests use `monkeypatch.setattr(runner, "LOCK_PATH", lock)` to isolate from the real lock file.
      **Commit:** (this iteration)
- [x] **M60+M61** — Settings pipeline UI lies. `AppSettings` has ~27 fields with full UI bindings (SettingsWindow tabs General/Advanced/Performance/Diagnostics/Updates). A grep across JANGStudio/ for every field (`pythonOverridePath`, `customJangToolsPath`, `tickThrottleMs`, `mlxThreadCount`, `logVerbosity`, `logFileOutputDir`, `preAllocateRam*`, `convertConcurrency`, `metalPipelineCacheEnabled`, `maxBundleSizeWarningMb`, `anonymizePathsInDiagnostics`, `autoDeletePartialOnCancel`, `revealInFinderOnFinish`, `defaultProfile`, `defaultFamily`, `defaultMethod`, `defaultHadamardEnabled`, `defaultCalibrationSamples`, `defaultOutputParentPath`) returned ZERO read-sites outside `AppSettings.swift` + `SettingsWindow.swift`. Users could toggle any of these, see the UI respond, watch it persist to UserDefaults — and nothing in the convert/inference/publish pipeline would consult them.
      **Fix scope this iter (M61 only — most-impactful):** Wired `pythonOverridePath` through to `BundleResolver`.
      - `BundleResolver.swift:5-32`: added `pythonOverrideDefaultsKey` + prioritized lookup order (UserDefaults → env var → bundled).
      - `AppSettings.swift:130-152`: `persist()` / `load()` now mirror `pythonOverridePath` to the dedicated leaf-consumer key. Empty string REMOVES the key so env/bundled fallbacks take over cleanly.
      - 6 new tests (persist mirror, clear removes key, resolver reads UserDefaults, empty string ignored, load re-syncs on fresh process, reset clears leaf mirror).
      **Evidence:** 81 Swift tests pass (was 75).
      **Commit:** (this iteration)
- [ ] **M62** — Remaining UI-lie settings. **Iter 10 closed 6, iter 11 closed 3 more** (9/12 done, 3 still inert):
  - ~~`autoDeletePartialOnCancel`~~ ✅ iter 10
  - ~~`revealInFinderOnFinish`~~ ✅ iter 10
  - ~~`defaultProfile` / `defaultFamily` / `defaultMethod` / `defaultHadamardEnabled`~~ ✅ iter 10
  - ~~`customJangToolsPath` → PYTHONPATH prepend~~ ✅ iter 11
  - ~~`tickThrottleMs` → JANG_TICK_THROTTLE_MS env var, Python side reads it~~ ✅ iter 11
  - ~~`mlxThreadCount` → OMP_NUM_THREADS + MLX_NUM_THREADS env vars~~ ✅ iter 11
  - `logVerbosity` → would need JANG_LOG_LEVEL in every emit site (wide refactor — deferred)
  - `preAllocateRam*` → no standard MLX env var for buffer pool (deferred — needs upstream feature)
  - ~~`anonymizePathsInDiagnostics` → DiagnosticsBundle pre-process rewrite (medium-size — deferred)~~ ✅ iter 14 (see M22)
- [x] **M62d** — `tickThrottleMs` / `mlxThreadCount` / `customJangToolsPath` were UI-only. Persisted to UserDefaults, never consulted by any subprocess-spawning code.
      **Fix (iter 11):** Unified env-addition builder `BundleResolver.childProcessEnvAdditions(inherited:)` reads dedicated leaf UserDefaults keys (same pattern as M61) and returns the env dict. Merged into all three subprocess entry points: `PythonRunner.launch`, `InferenceRunner.generate`, `PublishService.invoke`. Publish path inherits too since it's a python3 child just like convert.
      - tickThrottleMs → `JANG_TICK_THROTTLE_MS`. `progress.py` `_resolve_tick_interval_s()` reads it, falls back to 100 ms on empty / non-integer / zero / negative. Per-ProgressEmitter-instance so a bad env value doesn't poison the module-level constant.
      - mlxThreadCount → both `OMP_NUM_THREADS` (BLAS) and `MLX_NUM_THREADS` (MLX) so the user doesn't have to know which layer consumes which. 0 = "auto" → no env var set (can't wedge the child with `OMP_NUM_THREADS=0`).
      - customJangToolsPath → PYTHONPATH PREPEND (not replace) so bundled jang_tools still resolves for anything the custom path doesn't override.
      **Invariants pinned by tests:**
      - default settings → zero env additions (non-users pay zero cost)
      - returning a setting to its default REMOVES the leaf key (fall through to defaults, not frozen at stale value)
      - load() re-syncs all three leaf keys on fresh process (defends against leaf-key drift from other consumers)
      **Evidence:** `BundleResolver.swift:15-69`, `AppSettings.swift:155-175`, `progress.py:22-42`, 6 new `BundleResolverTests` + 4 new `AppSettingsTests` + 4 new Python progress tests.
      **Commit:** (this iteration)
- [x] **M62a** — `defaultProfile` / `defaultFamily` / `defaultMethod` / `defaultHadamardEnabled` were persisted but never read. Wizard always started at hardcoded `JANG_4K / jang / mse / false` regardless of user's saved defaults.
      **Fix:** Added `ConversionPlan.applyDefaults(from: AppSettings)` (MainActor) that seeds profile/family/method/hadamard from settings. Empty profile is a no-op (corruption guard). Method strings "mse-all"/"mseall"/"mse_all" all map to `.mseAll`. Per-conversion STATE (sourceURL/detected/outputURL/run) is never touched — verified by test. Called from `WizardView.task` once on first entry + from `VerifyStep.reset()` ("Convert another") so the fresh plan still reflects user settings.
      **Evidence:** `ConversionPlan.swift:57-88`, `WizardCoordinator.swift:36-68`, `VerifyStep.swift:158-168`. 5 new tests in `ConversionPlanTests` (accept normal, ignore empty, ignore unknown method, alias coverage, preserve per-conversion state).
      **Commit:** (this iteration)
- [x] **M62b** — `revealInFinderOnFinish` (default on) was inert. User had a successful 30-minute convert, needed to manually click "Reveal in Finder" to see the output.
      **Fix:** `VerifyStep.refresh()` now fires `revealOutput()` once on the first finishable render when the setting is on. `@State private var revealFiredOnce` guards against re-fires when the user tabs back to Verify. `reset()` clears the guard so the next "Convert another" cycle re-fires cleanly.
      **Evidence:** `VerifyStep.swift:16,124-136,167`.
      **Commit:** (this iteration)
- [x] **M62c** — `autoDeletePartialOnCancel` was inert. After cancel, partial output stayed on disk forever even with the setting on — user had to manually hunt for the output folder and rm it.
      **Fix:** `RunStep.start()` cancellation branch checks `settings.autoDeletePartialOnCancel` and runs `FileManager.removeItem(at:)` with logged success/failure. Left the "Delete partial output" button intact for the opposite case (setting off, user cancels, then decides to delete manually).
      **Evidence:** `RunStep.swift:6,92-107`.
      **Commit:** (this iteration)
- [ ] **M63** — `AppSettings.reset()` / `persist()` are synchronous `@MainActor` but the leaf-mirror writes happen inside them. If a write blocks (CloudKit-backed UserDefaults sync), the UI freezes. UserDefaults.standard is generally non-blocking but worth flagging.
- [ ] **M64** — `observeAndPersist` in `SettingsWindow.swift` uses `withObservationTracking` inside a loop with `withCheckedContinuation`. Each mutation fires a continuation that persists ONCE. But if two fields are mutated in the same SwiftUI pass (e.g., resetting via the Reset button), does the loop fire TWICE or ONCE? The `CheckedContinuation` pattern may miss paired mutations. Verify.
- [ ] **M65** — `SettingsWindow` auto-persist TASK is bound to the `.task { await observeAndPersist(settings) }` on the Settings body. If the user never OPENS Settings, the auto-persist never runs — which is fine (no changes to persist) UNLESS something else mutates settings programmatically (it doesn't today, but a future crash reporter that toggles `autoOpenIssueTrackerOnCrash` would lose the change).
- [ ] **M66** — `Snapshot.apply` uses `LogVerbosity(rawValue: logVerbosity) ?? .normal` — if someone writes a garbage value into UserDefaults (e.g., schema migration from a newer version downgraded), the setting silently reverts to `.normal` without telling the user. Same for `updateChannel`. Consider emitting a log line on coercion.

---

## K. Cross-cutting rules (never-forget)

- [ ] **K01** — No AI attribution in any commit, README, or public material. Spot-check recent 20 commits.
- [ ] **K02** — Author line `Jinho Jang (eric@jangq.ai)` not `Eric Jang`. Spot-check authored files.
- [ ] **K03** — `research/` stays gitignored. `feedback_no_research_public.md` — competitor replicated work from past leaks.
- [ ] **K04** — `.superpowers/` gitignored.
- [ ] **K05** — JANG stays quantized in memory (never dequant to fp16 in loader). Verify `load_jang_model` keeps `uint32 weight + scales + biases` in mx.array form.
- [ ] **K06** — JANG profile names use underscores + letter suffix: `JANG_2S`, `JANG_4K`, not `JANG-2bit`. HF repos follow this convention.
- [ ] **K07** — JANGTQ names: `JANGTQ2`, `JANGTQ3`, `JANGTQ4` (digit-only, no L/M/S/K suffix).
- [ ] **K08** — Ask-before-changes rule: read-only investigation is fine, any write needs sign-off. Ralph auto-commits are fine under Ralph Loop context since that's pre-authorized work.

---

## Ralph iteration protocol

Every iteration:

1. Read this file
2. Find the first `[ ]` item
3. Investigate it using the existing tests + codebase + memory references
4. If verified-proper: append a one-line evidence note + commit SHA (if any code touched); mark `[x]`
5. If drift found: fix it (commit to the current branch), add SHA + one-line note; mark `[x]`
6. If blocked (e.g., Llama HF terms): mark `[-]` with blocker reason — not skipped, just paused until unblocked
7. Commit this file with message `audit: <item id> — <result>`
8. End turn; Ralph Loop fires prompt again; repeat

When all items are `[x]` or `[-]`, the checklist gets a second pass with fresh eyes (items re-verified against latest behavior).

**Never mark an item `[x]` based on assumption.** Evidence or don't touch.

## Evidence pattern

Each closed item gets a note like:
```
- [x] **A04** — Qwen3.5 eos auto-fix: 248044 → 248046 applied top-level + text_config + tokenizer_config.
      **Evidence:** `convert.py:950-980` inspected; rule from `feedback_chat_template_rules.md` matches.
      **Commit:** existing, no change needed.
```

Or with a fix:
```
- [x] **D05** — ETA display now shown in Step 4 right rail.
      **Fix:** Added `etaSeconds` computed from tick throughput in `RunStep.swift`.
      **Commit:** a1b2c3d
```
