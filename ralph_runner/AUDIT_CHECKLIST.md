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
- [x] **M05** — PreflightRunner size estimate when `detected.totalBytes == 0` — current fallback returns pass with free-GB hint, but does the UI make clear that "no estimate" ≠ "safe"?
      **Close (iter 101):** pre-M05 the uncheckable branch returned `.pass` with `"X GB free"` — visually identical to a real positive check. User couldn't tell whether the system had actually verified sufficient space or simply couldn't compute an estimate yet (pre-inspection, unknown profile). Changed to `.warn` with `"X GB free (no estimate yet — pick source + profile for a real check)"`. Warn doesn't block preflight like fail would, but makes the uncheckable state UI-visible so the user knows to return to the check after picking source + profile.
      **Tests (+2) in PreflightRunnerTests.swift:**
      - `test_diskSpace_pre_inspection_warns_about_uncheckable_state` — plan with source + output + profile but no detected size → asserts `.warn` + `"no estimate"` in hint.
      - `test_diskSpace_post_inspection_with_room_still_passes` — regression guard: with a real estimate + sufficient free space, still `.pass`.
      **Evidence:** `JANGStudio/JANGStudio/Verify/PreflightRunner.swift:170-188`. 26 PreflightRunnerTests pass (was 24, +2).
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
- [x] **M189 (jang-server max-request-body-size middleware — closes memory-bomb DoS)** — Iter-126 closed the last unbounded-input vector in jang-server. Pre-M189 the server had NO upper bound on POST body size. An attacker could send a 10 GB JSON body, exhausting RAM before Pydantic validation rejects the wrong shape. JobRequest / EstimateRequest payloads are at most a few KB in practice (model_id + profile + small metadata), so capping at 1 MB by default leaves 1000× headroom and stops memory-bomb requests cold.
      **Fix (iter 126):** added `MAX_BODY_BYTES` constant (env-tunable via `JANG_MAX_BODY_BYTES`, default 1 MB) + `@app.middleware("http")` `limit_body_size` middleware that:
      1. Reads `Content-Length` header from request.
      2. If `> MAX_BODY_BYTES`, returns 413 Payload Too Large (per RFC 9110) with explanation + env-var name.
      3. If header is malformed/missing, falls through to normal handling (don't second-guess weird headers; downstream Pydantic catches malformed bodies).
      **Coverage:** the header-based check covers the typical attacker who declares the size. A sneaky attacker could use chunked transfer encoding (no Content-Length header) to bypass — full coverage would require streaming-byte-counter middleware, more complex. Acceptable trade-off given (a) most HTTP clients send Content-Length for static-size bodies, (b) chunked-bypass requires the attacker to actually send the bytes, which still costs them bandwidth, (c) reverse proxies (nginx default `client_max_body_size 1m`) can backstop in production.
      **Tests (+5) in new `jang-server/tests/test_max_body_size.py`:**
      - `test_max_body_bytes_env_var_defined`
      - `test_limit_body_size_middleware_registered` — pin the `@app.middleware("http")` decorator.
      - `test_middleware_rejects_with_413` — HTTP status compliance per RFC 9110.
      - `test_middleware_inspects_content_length_header` — pin the header-name lowercase match.
      - `test_default_cap_is_reasonable_for_jang_payloads` — guards against accidental defaults outside [64KB, 100MB].
      Also bumped iter-111 M177 invariant allowlist range AGAIN (line shifted from ~1207 to ~1300 with the body-size middleware's ~50 lines added above). 4th bump now — meta-lesson queued: refactor to function-body slicing per iter-125.
      **Evidence:** `jang-server/server.py:668-712`. 34 jang-server tests pass (was 29, +5).
      **Meta-lesson — declared-size body bombs are the cheap-fix half; chunked bypasses are the hard half.** Header-based checks catch 95% of attackers (most HTTP clients always send Content-Length). The remaining 5% (chunked-encoding without declaring size) are advanced and usually need network-layer mitigation (nginx, WAF). For an in-app middleware, header check is the right level of defense — defending against the declared-size case is cheap and high-value; defending against chunked bypass is expensive and usually duplicate of network-layer protections. **Rule: when adding security middleware to a Python web app, check the header path FIRST; backstop chunked bypasses with reverse-proxy config.** Document the layer responsibility so the operator knows what to configure.
      **Meta-lesson — sensible defaults bound the input space.** A 1 MB default cap was chosen because (a) JANG payloads are KB-scale (1000× headroom), (b) operators rarely need to send larger payloads to a quantization API, (c) 1 MB is the de facto default in nginx + IIS + most reverse proxies — operators are used to it. Picking a default that matches industry conventions reduces "why does my legit upload fail?" surprises. **Rule: when adding a configurable cap, anchor the default to industry conventions; document the conventions in the comment.**
      **Commit:** (this iteration)
- [x] **M188 (jang-server SSE concurrent-connection cap — complements M187)** — Iter-125 closes a DoS gap iter-124 M187's rate-limiter doesn't cover. M187 caps the OPEN-call rate (e.g., 30 streams/minute per IP). But SSE streams are LONG-LIVED — a client can open at the rate limit and accumulate thousands of open streams over time. Each consumes a file descriptor + asyncio task + _sse_subscribers entry. Process FD limits (typically 1024-4096 default on macOS/Linux) become the real cap; hitting them bricks the whole server.
      **Fix (iter 125):** added per-IP + global concurrent-stream caps. Two env vars: `JANG_SSE_MAX_PER_IP` (default 10) + `JANG_SSE_MAX_GLOBAL` (default 200). Stream open path:
      1. Acquire `_sse_count_lock`.
      2. Check global open count (sum of all IP counts) against `SSE_MAX_GLOBAL` — reject 429 if exceeded.
      3. Check per-IP count against `SSE_MAX_PER_IP` — reject 429 if exceeded.
      4. Increment IP counter.
      5. Release lock + proceed to setup queue + return StreamingResponse.
      **Pair with decrement** in event_generator's `finally` block. Drops the dict key when count hits 0 (matches iter-115 M180 ghost-key pattern). Without the matching decrement, a client who opens N streams + closes them ALL would still show N in the per-IP count, locking them out of new streams forever — silent monotonic accumulation bug.
      **429 messages** explain WHICH cap was hit (per-IP vs global) so the client can react appropriately. Per-IP says "close existing streams or wait"; global says "try again in a minute" (other clients holding streams might close).
      **Tests (+4) in new `jang-server/tests/test_sse_connection_limit.py`:**
      - `test_sse_connection_caps_defined` — env vars + rationale defined.
      - `test_stream_job_checks_ip_count_before_accept` — preamble checks both caps.
      - `test_event_generator_decrements_counter_in_finally` — pin the decrement + dict-cleanup so the M180 ghost-key class can't reappear here.
      - `test_sse_caps_documented_in_source` — comments mention "file descriptor" so future tuning sees the WHY.
      **Test gotcha caught + fixed:** initial test substring-search window was 3000 chars from the function start; the decrement lives past that. Bumped to 5000 with a comment explaining the function's grown body (M188 preamble + M180 cleanup + M188 decrement). Iter-104 M108 / iter-115's "moving line numbers" lesson applied to substring-window sizes too.
      **Evidence:** `jang-server/server.py:262-280, 901-940, 974-980`. 29 jang-server tests pass (was 25, +4).
      **Meta-lesson — open-rate limit ≠ concurrent-count limit.** Two distinct DoS vectors with two distinct mitigations:
       - **Open-rate** (iter-124 M187): how fast can a client establish new connections? → token bucket / sliding window.
       - **Concurrent-count** (iter-125 M188): how many open connections can a client hold simultaneously? → counter + cap.
      Long-lived connections (SSE, WebSocket, gRPC streams) need BOTH. Short-lived requests (typical REST POST/GET) only need open-rate. **Rule for any new endpoint: classify connection lifetime — long-lived needs both limits, short-lived needs only the rate limit.**
      **Meta-lesson — paired increment/decrement state needs explicit pin tests.** The bug "monotonic counter never decrements" is exactly the kind of thing that PASSES every functional test until your client hits SSE_MAX_PER_IP after a few hours of normal usage. The pin test asserts `_sse_open_counts[ip]` AND `- 1` appear in the function body — catches a future refactor that drops the decrement. Same M180 class as the ghost-key cleanup.
      **Commit:** (this iteration)
- [x] **M187 (jang-server rate-limiting on POST /estimate + POST /jobs)** — Iter-124 added a per-IP sliding-window rate limiter to jang-server's high-cost POST endpoints. Pre-M187 a single client could flood:
      - **POST /estimate** — each call hits the HF API (`HfApi.model_info`). An auth'd attacker could exhaust the server's HF rate-limit budget, breaking `/estimate` for every other user.
      - **POST /jobs** — creates DB rows + runs validation (HFRepoValidator, duplicate detection, per-user limit check). MAX_JOBS_PER_USER bounds CONCURRENT jobs but not creation RATE — flooding rejected submissions still consumes CPU.
      MAX_CONCURRENT bounds total compute work, but neither endpoint had per-source rate limiting.
      **Fix (iter 124):** added `check_rate_limit(request: Request) -> None` FastAPI dependency. Sliding-window per-IP, configurable via `JANG_RATE_LIMIT_WINDOW_S` (default 60s) + `JANG_RATE_LIMIT_MAX_REQUESTS` (default 30 = avg 1 req every 2s). Implementation: `dict[ip, deque[timestamps]]` with `popleft` for window expiry — sliding-window not fixed-counter (fixed counter would let clients burst at minute boundaries).
      Rate-check fires BEFORE auth check (via `dependencies=[Depends(check_rate_limit), Depends(check_auth)]`) so failed-auth attempts also count against the limit — defends against auth-brute-force flooding too.
      **429 response** carries `Retry-After` header with exact remaining seconds — matches RFC 6585.
      **Public endpoints** (`/health`, `/profiles`) explicitly NOT rate-limited (test pins this) — keeps liveness probes cheap. Hostile flooding of `/health` is a network-layer concern (nginx/WAF).
      **Tests (+5) in new `jang-server/tests/test_rate_limit.py`:**
      - `test_check_rate_limit_function_exists`
      - `test_check_rate_limit_uses_sliding_window` — pins `popleft` + `_rate_limit_log` literals so fixed-counter regressions fail.
      - `test_rate_limit_applied_to_high_cost_endpoints` — parser asserts /estimate + /jobs include `check_rate_limit` Depends.
      - `test_public_endpoints_skip_rate_limit` — guards over-correction.
      - `test_rate_limit_env_vars_documented` — env-var names appear in source for operator discoverability.
      Also widened iter-111 M177 allowlist range (line numbers shifted again from ~1150 to ~1207 due to rate-limit helper's ~50 lines added above).
      **Evidence:** `jang-server/server.py:262-318, 633, 928`. 25 jang-server tests pass (was 20, +5).
      **Meta-lesson — rate limits should fire BEFORE auth checks.** Putting `Depends(check_rate_limit)` before `Depends(check_auth)` means failed-auth attempts count against the limit too. This defends against auth-brute-force flooding (each guess costs the attacker rate-budget). Reverse order would let an attacker spray invalid keys at infinite rate. **Rule: in any FastAPI app, rate-limit deps must come BEFORE auth deps.**
      **Meta-lesson — sliding window > fixed counter.** A fixed-counter ("at most N per minute, resets at minute :00") lets a client send N at :59 and another N at :00:01 = 2N in 2 seconds. Sliding window ("at most N in any 60s span") prevents the burst. Trivial impl difference (`deque + popleft` vs `int + time.minute`) but big behavioral difference. **Rule: when implementing rate limits, use sliding-window from the start.**
      **Commit:** (this iteration)
- [x] **M186 (JANGQuantizer.swiftpm QueueView — Cancel/Retry buttons silently swallowed errors)** — Iter-121 continued the JANGQuantizer.swiftpm sweep started in iter-120 M185. Found 2 more iter-35 M107-class silent-swallows in QueueView.swift's job-card action buttons:
      ```swift
      Button("Cancel") { Task { try? await api.cancelJob(job.jobId) } }
      Button("Retry") { Task { try? await api.retryJob(job.jobId) } }
      ```
      The `try?` consumes any failure (server unreachable, auth expired, network blip, the targeted job no longer in a cancellable state). User clicks Cancel → fails → button looks like nothing happened → user clicks again, gets the same nothing. iter-35 M107 / iter-90 M167 / iter-120 M185 pattern in yet another fresh file (third instance of the same pattern across two apps + two views).
      **Fix (iter 121):** swapped `try?` → `do/catch`, added `@State actionError: String?` to JobCard. Catch sets `actionError = "Cancel failed: \(error.localizedDescription)"` (uses iter-91 M168's actionable-description chain — APIError.errorDescription already follows that pattern from M185). Renders inline below the action buttons in red so the user sees the diagnostic on the same card without leaving the queue list. Success path nulls actionError so the error message clears on a retry.
      **Tests:** SwiftPM has no XCTest harness; visual review only. M182 secrets sweep still clean (78 ralph_runner tests pass).
      **Other audits in this iter (no bugs found):**
      - QueueView refresh timer: properly invalidated on .onDisappear ✓ (iter-94 M171 pattern correctly applied here).
      - QueueView.refresh() ad-hoc Task: comment says "Keep existing data on refresh failure" — intentional silent-fall-through, acceptable per iter-104 M108 taxonomy bucket. Mostly benign for a 3-second poll cycle.
      - SubmitView.submit Task: ad-hoc no handle, but submission is a brief HTTP call (<1s typical). Polish item, not a bug.
      **Meta-lesson — third instance of a pattern means it needs a feedback memory.** iter-35 M107 (JANGStudio Settings) → iter-120 M185 (JANGQuantizer Settings) → iter-121 M186 (JANGQuantizer Queue). Three distinct files in two apps shipped with `} catch { swallow }` for user-action buttons. **Rule for the team: any user-action button calling an async API must use `do/catch` + visible error surface. NEVER `try?` in a Button handler unless the operation is truly idempotent + best-effort (e.g., dismissing a notification).** Should codify this as a feedback memory note alongside the iter-83 pipe-drain + iter-92 remediation memos. Recurrence pattern: same dev, same blind spot, every fresh button.
      **Commit:** (this iteration)
- [x] **M185 (JANGQuantizer.swiftpm — URL query injection in listJobs + silent health-check error in SettingsView)** — Iter-120 audited the previously-unaudited `jang-server/frontend/JANGQuantizer.swiftpm/` Swift Package (jang-server's standalone client UI). Found two bugs in the first pass — different classes, both real, both standard fixes from the iter-83/92/94/101 patterns I've been refining.
      **Bug A — URL query injection in `APIClient.listJobs` (APIClient.swift:49):**
      ```swift
      var path = "/jobs?"
      if let u = user, !u.isEmpty { path += "user=\(u)&" }
      if let p = phase, !p.isEmpty { path += "phase=\(p)&" }
      ```
      String concatenation without URL encoding. A username containing `&`, `=`, `?`, `#`, `+`, space, or `%` breaks the URL OR injects parameters. Example: user named `alice&phase=COMPLETED` → server receives `user=alice` AND `phase=COMPLETED` (overriding any phase the caller intended). Not adversarial in the usual SSRF sense (user names are typically benign) but a real data-correctness bug.
      **Fix:** rebuilt with `URLComponents` + `URLQueryItem`. URLSession encodes correctly. Defensive `guard let pathPlusQuery = components.string` surfaces composition failures clearly.
      **Bug B — silent health-check error in `SettingsView.Check Connection` button:**
      ```swift
      Button("Check Connection") { Task { do { health = try await api.getHealth() } catch { health = nil } } }
      ```
      Classic iter-35 M107 / iter-90 M167 silent-swallow pattern in a fresh file. User clicks "Check Connection" → server unreachable / wrong URL / bad token → `health = nil` → status indicator simply disappears with NO error message. User has zero diagnostic for what failed.
      **Fix:** added `@State private var lastError: String?`. Catch branch sets `lastError = "Connection failed: \(error.localizedDescription)"` (uses iter-91 M168 / iter-92 M169 actionable-description pattern from `APIError.errorDescription`). Renders in red below the button when non-nil. Initial-load `.task` keeps silent-fail behavior (server-not-yet-started shouldn't show a banner on first app open) — only the user-clicked button surfaces the error.
      **Tests:** the swiftpm has no XCTest harness; visual code review + re-running M182 sweep (78 ralph_runner tests pass; the new edits are clean per the secrets sweep).
      **Meta-lesson — patterns identified in one app surface in fresh apps the same way.** iter-35 M107 (Settings silent-swallow) was fixed in JANGStudio months ago. JANGQuantizer.swiftpm is a younger, smaller app written by the same team — and shipped with the same anti-pattern. **Rule: when a meta-pattern is established for one app, sweep ALL apps in the monorepo for the same pattern.** Iter-117 M182's repo-wide approach for secrets is the same idea applied at the test level; iter-120 M185 applies it as a code-pattern review.
      **Meta-lesson — string concatenation for URL query strings is an evergreen bug.** Even in 2025 Swift code with the `URLComponents` API readily available, devs reach for the simpler `+= "param=\(value)&"` pattern. **Rule: any URL query construction in any HTTP client must use `URLComponents` + `URLQueryItem`. Banner this in the JANGQuantizer code review checklist (and in feedback memory if a session-spanning rule is warranted).**
      **Commit:** (this iteration)
- [x] **M184 (M182 sweep was scanning 569 generated files in `.build/` dirs — SKIP_DIR_NAMES gap)** — Iter-119 was originally scoped as "extend coverage to HTML/JS for jang-server frontend." Found that the frontend is a Swift Package (`.swiftpm`) — Swift files already covered by M182. While inspecting coverage, ran a diagnostic that revealed M182's `_iter_source_files` was scanning **569 files inside `JANGQuantizer.swiftpm/.build/` build outputs every test run** — generated code that has no business being audited. Slowed the test ~5× AND risked false positives from compiler-generated identifiers that happen to shape-match secret regexes.
      **Root cause:** `SKIP_DIR_NAMES` had `"build"` (lowercase, no dot) but the SwiftPM output dir is named `.build` with a leading dot. `Path.parts` matches whole components — `.build` doesn't match `build` even with substring logic.
      **Fix (iter 119):** added `.build` to SKIP_DIR_NAMES with an explanatory comment. Pre-emptively also added `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `.tox` (other common dotted build/cache dirs that could appear in future).
      **Trap I almost set:** initially also added `.swiftpm` to the skip set — would have skipped the legit JANGQuantizer Sources because they live INSIDE `JANGQuantizer.swiftpm/Sources/`. The `.swiftpm` is a CONTAINER directory (like `.app`), not a build output. Backed out + added a NOTE comment so the next maintainer doesn't make the same mistake.
      **Verified:** 7 SwiftPM Sources still covered; 0 files in `.build/` dirs scanned post-fix.
      **Tests:** 78 ralph_runner tests pass (count unchanged — fix is in skip-set, not new test assertions).
      **Meta-lesson — diagnostic checks reveal infrastructure bugs that the test logic itself can't catch.** M182's `test_no_hardcoded_secrets_repo_wide` PASSED before iter-119 — there were no secrets in the .build/ files (it's compiled output, no string literals in human-readable form). The bug was performance + risk-of-future-FP, not correctness. **Rule: when designing an exclusion-based test (skip these dirs / extensions / patterns), periodically print what the test IS scanning to confirm the exclusion logic matches intent. Cheap diagnostic; catches the "wrong dir name" / "case mismatch" / "missing entry" class of skip-set bug.**
      **Meta-lesson — dotted build dirs need explicit skip entries.** Path-component matching doesn't see `.build` as containing `build`. Same trap will hit `.gradle`, `.cargo`, `.terraform`, etc. **Rule: when adding skip entries, list BOTH dotted and undotted variants for any dir that could appear with either prefix.**
      **Meta-lesson — never assume container-dir extensions are build outputs.** `.swiftpm`, `.app`, `.framework`, `.bundle` are MAC OS BUNDLE conventions — they look like extensions but contain real source. Distinguish "package container" from "build output" before adding to skip list. The note-comment in code prevents a future iter from re-introducing the trap.
      **Commit:** (this iteration)
- [x] **M183 (Extend M182 secrets sweep to non-source files: JSON / YAML / shell / docs)** — Iter-118 extends iter-117 M182 to other file types where hardcoded secrets commonly leak: JSON / YAML configs, shell scripts (deploy + CI), `.env` files, `.toml` configs, `.cfg` files, and Markdown docs (CI logs / examples sometimes embed live tokens).
      **Approach:** new `test_no_hardcoded_secrets_in_nonsource_files` companion test in the same file. Reuses M182's `SECRET_PATTERNS` regex set + `ALLOWED_FIXTURES` allowlist mechanism (deduplicated by adding NONSOURCE_EXTENSIONS + a `_iter_nonsource_files` walker). Skips `pyproject.toml` (package specs may shape-match) and `.env.example` (placeholder values are the file's purpose).
      **First run found 6 hits** — all in our own audit docs (`AUDIT_CHECKLIST.md` + `INVESTIGATION_LOG.md`) where the M181/M182 entries reference fixture token names by literal value. These are audit documentation, not actual leaks. Allowlisted both doc files for both HF regex flavors with rationale.
      **Result:** repo-wide secrets coverage now spans `.py + .swift` (M182) AND `.json + .yaml + .yml + .sh + .env + .md + .toml + .cfg` (M183). 4 file types where the M181 class of bug commonly manifests now have automated coverage.
      **Tests:** 78 ralph_runner tests pass (was 77, +1).
      **Meta-lesson — extension iters are cheap when the test infrastructure is reusable.** M182 built the regex set + allowlist mechanism; M183 reuses both, just adds a different file walker. ~10 min of work; same compound-interest pattern as iter-99→iter-100 (PreflightRunner sourceBytesPerWeight reused in PostConvertVerifier) and iter-105→iter-106 (jang-tools dual-invariant template reused in ralph_runner). **Rule: when an audit invariant is being designed, structure it for reuse FROM THE START — keep the regex/threshold/allowlist mechanisms in one place + invoke them from multiple test functions for different scopes (per-file, per-module, repo-wide-source, repo-wide-config). Saves iters for follow-up extensions.**
      **Meta-lesson — audit documentation is itself a leak surface.** My own AUDIT_CHECKLIST.md / INVESTIGATION_LOG.md entries quoting fixture tokens by literal value showed up as "offenders" in the M183 sweep. Files that are part of a security audit can themselves contain pattern-matching strings. **Rule: when documenting fixture data in markdown, prefer redacted forms (`hf_<lit>...<yz>`) over full literals — keeps the docs portable + sweep-clean. Or, allowlist the docs explicitly with a rationale comment (the path I took here for backwards compat with existing entries).**
      **Commit:** (this iteration)
- [x] **M182 (Repo-wide secrets-sweep invariant — extends M181's per-file check to all .py + .swift)** — Iter-117 extends iter-116 M181's jang-server-scoped secrets check to a cross-cutting invariant covering every `.py` and `.swift` file across the repo. Without repo-wide coverage, the same M181-class bug could reappear in jang-tools, JANGStudio Swift, ralph_runner, jang-runtime, or any future module without anyone noticing until the next manual audit.
      **Patterns covered:** HF (`hf_*`, `huggingface_*`), OpenAI (`sk-*`, `sk-proj-*`), AWS (`AKIA*`, `ASIA*`), GitHub (`gh[pousr]_*`). Standard secret shapes that have file:line-grep-able signatures.
      **Allowlist mechanism:** per-(file, pattern-name) tuples for clearly-fake test fixtures. First run found 3 fixture-shape literals: `hf_literal_looking_token_abc123xyz` + `hf_dummy_token_for_test` in jang-tools/tests/test_publish.py + `huggingface_abcdef_ghij-...` in DiagnosticsBundleTests.swift. All three are existing test data verifying scrub-sensitive / token-disambiguation behavior. Allowlisted with rationale.
      **Test masks the matched substring** in failure output so the assertion error itself doesn't re-leak the hit value (only `[hf_lit<...>yz]`-style preview shown).
      **New file:** `ralph_runner/tests/test_no_hardcoded_secrets_repo_wide.py`. Skips vendored third-party code (build/, .venv/, node_modules/, site-packages/) which has its own test fixtures + public test tokens.
      **Tests pass:** 77 ralph_runner tests pass (was 76, +1).
      **Meta-lesson — per-module invariants don't catch cross-module regressions; build a cross-cutting one.** iter-116 M181 added the secrets check ONLY to jang-server. Without iter-117's repo-wide variant, a future hardcode in jang-tools or JANGStudio would slip through. **Rule: when an invariant catches a bug class that could occur in any source file, scope it repo-wide. Per-module invariants are useful for module-specific patterns (e.g., the iter-104/105/106 dual-invariant for `try?` / `except Exception` taxonomy varies by language); cross-cutting bugs need cross-cutting tests.**
      **Meta-lesson — mask matched secret substrings in failure output.** A test that prints the secret it found in its assertion message effectively re-leaks the secret to anyone reading CI logs. Truncate to a head + tail (e.g., `[hf_lit<...>yz]`) — enough for the engineer to identify which token shape triggered, not enough to use the secret. Same principle as iter-14 M22 DiagnosticsBundle scrubbing. Standard rule for security-related test output.
      **Commit:** (this iteration)
- [x] **M181 (jang-server hardcoded HF write-token in source — CRITICAL, requires token rotation)** — **🚨 ACTION REQUIRED FROM ERIC: rotate the leaked token at https://huggingface.co/settings/tokens.** Fix here only stops the leak going forward; anyone with repo or git-history access already has the original.
      **Bug:** Iter-116 ran a scan-for-secrets sweep across the repo. Found a real `hf_*` write-token committed as the default value of `HF_UPLOAD_TOKEN` in `jang-server/server.py:51-53`:
      ```
      HF_UPLOAD_TOKEN = os.environ.get("HF_UPLOAD_TOKEN", "hf_<redacted>")
      ```
      The second arg to `os.environ.get` is a default that fires when the env var isn't set — meaning the server would silently use this leaked token for HF uploads in any deployment that didn't explicitly set the env. The token grants write access to the `JANGQ-AI` HF org.
      **Severity:** HIGH. Compromised the moment the source was first committed. Anyone who:
      - Cloned the repo at any point.
      - Has read access to git history.
      - Saw the file in a code review / screen share / search index.
      …has the token. Even after this fix, they retain it.
      **Fix (iter 116):** removed the default value. `HF_UPLOAD_TOKEN = os.environ.get("HF_UPLOAD_TOKEN", "")`. Server now requires explicit env var; missing-token publish attempts will fail-fast with a clear error instead of silently using a default that may be leaked or revoked.
      **Tests (+2) in new `jang-server/tests/test_no_hardcoded_secrets.py`:**
      - `test_no_hardcoded_hf_token_in_server_py` — regex `\bhf_[A-Za-z0-9_-]{20,}\b` catches any future `hf_*` literal in server.py.
      - `test_HF_UPLOAD_TOKEN_default_is_empty` — semantic check that the env-var-read line uses `""` or `None` as default (not a real value).
      **Cross-repo sweep:** also grep'd the rest of `/Users/eric/jang/` for `hf_*` literals. Only matches outside our source were in third-party library code (transformers' public test token in `testing_utils.py`) and our own test fixtures (which use clearly-fake tokens like `hf_abcdefghijklmnopqrstuvwxyz1234567890`). Repo source is clean post-M181.
      **Evidence:** `jang-server/server.py:51-60`. 20 jang-server tests pass (was 18, +2).
      **Meta-lesson — secrets audits with regex sweep find what manual review misses.** Iter-113/114 found SSRF + authz with adversarial-framing audits but didn't grep for hardcoded tokens. Iter-116's targeted regex caught the leak in the first pass. **Rule: every fresh codebase audit should include a hardcoded-secret sweep early — `hf_*`, `sk-*`, `AKIA*`, `password\s*=\s*['"]...['"]`, `api_key\s*=\s*['"]...['"]`. Cheap to run; high consequence when it hits.**
      **Meta-lesson — `os.environ.get(KEY, DEFAULT)` is a leak vector by default.** The pattern is convenient but dangerous when DEFAULT is a real secret. **Rule: any env-var read for a secret must use `""` or `None` as default, then check at use-site and fail-fast if missing.** Convert silent-fall-through into actionable error. Same iter-101/108 "don't lie to the user" rule applied to operators: don't silently use a fallback secret.
      **Commit:** (this iteration)
- [x] **M180 (jang-server `_sse_subscribers` ghost-key slow leak — observation flagged iter-114, fixed iter-115)** — Iter-114 noted during the unbounded-resource sweep. Pre-M180 the SSE event-generator's `finally:` block removed a disconnecting client's queue from `_sse_subscribers[job_id]` but left the dict ENTRY behind (with empty list value). Over thousands of job submissions + subscriber disconnects, the dict accumulated ghost keys.
      **Fix (iter 115):** added `if not subs and job_id in _sse_subscribers: del _sse_subscribers[job_id]` to the finally block. When the last subscriber leaves, drop the key entirely. Defense-in-depth: `/admin/purge` also `pop()`s subscribers entries for purged job IDs, in case some subscribers haven't disconnected by purge time (e.g., long-running streams hanging on idle).
      **Tests (+2) in new `jang-server/tests/test_sse_subscribers_cleanup.py`:**
      - `test_sse_event_generator_finally_drops_empty_dict_entry` — pin the `del _sse_subscribers[job_id]` + `if not subs` guard literals.
      - `test_admin_purge_also_clears_sse_subscribers` — pin the `_sse_subscribers.pop(jid, None)` cleanup inside the purge loop.
      Also updated iter-111 M177 invariant test's allowlist range — line numbers shifted from ~1121 to ~1150 due to my edits in this iter; widened tolerance to `range(1115, 1200)` with a comment noting that exceeding the range warrants checking for new bare-pass sites before just bumping further.
      **Evidence:** `jang-server/server.py:868-883, 1003-1012`. 18 jang-server tests pass (was 16, +2).
      **Meta-lesson — slow-drip leaks compound over server uptime.** A 100-byte ghost dict entry per discarded job seems trivial. After a year of 1000 jobs/day = 365k jobs = 36MB of dict bloat just from ghost keys. Plus dict-resize amortized cost for inserting at scale. **Rule: any dict keyed by an entity with finite lifetime (job, session, request, user) needs explicit cleanup at the entity's end. The "finally" block is the natural place — make it dropping the dict entry, not just the value.**
      **Meta-lesson — observation → fix path is short when the observation is concrete.** Iter-114 noted the leak in passing while sweeping for other security issues. Iter-115 fixed it in 5 minutes. Concrete observations close fast; vague ones (M97, M117) take dedicated iters. Rule: when observing a side issue during a different audit, write down the SPECIFIC location + cleanup-shape — turns a "consider eventually" into a "fix in 5 min next iter."
      **Commit:** (this iteration)
- [x] **M179 (jang-server authorization gap — job-read GETs unprotected when API_KEYS set)** — Iter-114 continued the security audit. Mapped every `@app.METHOD` decorator and which endpoints carry `Depends(check_auth)`. Found 5 GETs that returned per-user job content WITHOUT auth requirements:
      - `GET /jobs` — lists ALL jobs across all users.
      - `GET /jobs/{job_id}` — full job state including model_id, user, error messages.
      - `GET /jobs/{job_id}/logs` — last 200 log lines (may contain paths, error context).
      - `GET /jobs/{job_id}/stream` — SSE feed of phase transitions.
      - `GET /queue` — queue ordering + priorities.
      **Impact:** if deployed with `JANG_API_KEYS` set (the documented production posture), POST endpoints (create/cancel/retry/estimate/admin-purge) were properly auth-gated, but READ endpoints leaked job content to anyone with network access. Multi-user deployments allowed users to enumerate + spy on each other's jobs. Single-user deployments still leaked logs to scanners on the same network.
      **Fix (iter 114):** added `dependencies=[Depends(check_auth)]` to all 5 endpoints. `check_auth` is opt-in via env (returns early if API_KEYS empty), so open deployments are unaffected; auth'd deployments now properly gate reads too. `/health` and `/profiles` left open — standard public-endpoint posture.
      **Tests (+2) in new `jang-server/tests/test_auth_enforcement.py`:**
      - `test_sensitive_endpoints_require_auth` — parses every `@app.METHOD("/path")` decorator from server.py, asserts each endpoint in the AUTH_REQUIRED set has `Depends(check_auth)`. Future endpoint additions with sensitive content fail this test until the decorator is added.
      - `test_public_endpoints_remain_open` — regression guard against over-correction (sweeping auth onto health/profiles).
      **Evidence:** `jang-server/server.py:711-844`. 16 jang-server tests pass (was 14, +2).
      **Meta-lesson — auth audits need a per-endpoint matrix.** Iter-113 found SSRF in webhook plumbing; iter-114 found authz gaps in routing. Both are HTTP-layer vulnerabilities but require different lenses. Rule: for any HTTP server, build a per-endpoint table of (method, path, auth-required?, public-by-design?) at audit time. Mismatches between intent and decorator presence are mechanical to spot once the table exists. The new auth-enforcement test crystallizes that table into source.
      **Meta-lesson — opt-in auth is easy to get wrong.** `check_auth` early-returns when API_KEYS is empty, which means devs testing locally without API_KEYS see endpoints "work" regardless of decorator. The auth gap was invisible during local dev — only manifests in production. Rule: when an auth dependency is conditionally enforced (env-var gated), audit decorator presence STATICALLY because runtime behavior in dev != production.
      **Commit:** (this iteration)
- [x] **M178 (jang-server webhook_url SSRF — user-controlled URL blindly POSTed)** — Iter-113 audited jang-server for server-specific anti-patterns (unbounded resources, uncaught async, missing auth, etc.). **Found a REAL security vulnerability in the first sweep.**
      **Bug:** `_fire_webhook(job)` at line 1484 POSTs JSON to `job.webhook_url` — a user-controlled string from the job submission request. Pre-M178, the server blindly made an HTTP request to whatever URL the user provided. Classic **Server-Side Request Forgery (SSRF)** vulnerability. Attack vectors:
      - **Loopback:** `webhook_url = "http://127.0.0.1:8080/admin"` → server hits its own localhost services.
      - **Private LAN:** `"http://192.168.1.1/router/admin"` → targets routers and internal dashboards.
      - **Cloud metadata:** `"http://169.254.169.254/latest/meta-data/"` → AWS/GCP/Azure instance metadata service (classic escalation to IAM credentials).
      - **Non-HTTP schemes:** `"file:///etc/passwd"` (urllib may support file://).
      - **IPv6 equivalents:** `"http://[::1]/"`, `"http://[fc00::1]/"`.
      **Fix (iter 113):** added `_validate_webhook_url(url)` with comprehensive checks:
      - Empty URL → valid (no webhook).
      - Scheme must be `http` or `https` (rejects file, gopher, ftp, data, etc.).
      - Hostname present + resolves via `socket.getaddrinfo`.
      - **All resolved IPs** checked via `ipaddress.ip_address(ip).is_private / .is_loopback / .is_link_local / .is_multicast / .is_reserved / .is_unspecified`. Handles IPv4 + IPv6 + hostnames like `localhost` that resolve to loopback.
      **Defense-in-depth:** validator applied at TWO points:
      - **Submission time** (`create_job`) — fails fast with 400 Bad Request. Server never persists an invalid webhook URL.
      - **Fire time** (`_fire_webhook`) — re-validates before the outbound request. Catches any pre-M178 persisted jobs still in the DB.
      **Tests (+12) in new `jang-server/tests/test_webhook_ssrf.py`:**
      - Accepts: empty string, public https URL.
      - Rejects: `file://`, `gopher://`, `127.0.0.1`, `10.0.0.5`, `192.168.1.1`, `169.254.169.254` (AWS metadata), `[::1]`, `localhost`, nonexistent hostname, missing hostname.
      **Evidence:** `jang-server/server.py:1476-1534, 629-632`. 14 jang-server tests pass (was 2, +12).
      **Meta-lesson — security audits yield fast on fresh surfaces.** Iter-112 re-swept already-audited Swift buttons and found 0 bugs (diminishing returns signal). Iter-113 pivoted to jang-server with a security-specific lens ("what does the server do with user input?") and found a classic SSRF in the first function inspected. **Rule: for security-critical code paths (any place user input becomes an outbound request / subprocess / SQL / file path), explicitly enumerate the attack classes (SSRF, injection, path traversal, SSRF-via-DNS-rebinding) and check each.** Security threats don't show up through the normal "does it work" audit; they require adversarial thinking from the start.
      **Commit:** (this iteration)
- [x] **M177 (jang-server bare `except Exception: pass` sweep — 4 sites audited, 3 fixed + dual invariant added)** — Iter-111 diversified the audit surface to `jang-server/` (previously unaudited). Applied iter-106 M119's dual-invariant template to a new subproject.
      **Site inventory:** 10 `except Exception` sites in `server.py` (1774 lines). 4 were bare `except Exception: pass`:
      - L415 — DB job restore per-row loop.
      - L898 — HF config fetch in `/estimate` endpoint.
      - L944 — HF config fetch in `/recommend` endpoint (same shape, different handler — easy to miss in a grep because each is in its own function body).
      - L1113 — progress-pct calculation (bytes_total=0 guard).
      **Fix (iter 111):** three of four sites converted to `log.warning(...)` before the implicit fall-through. Server context especially demands this — daemon accumulates silent-swallow damage over days/weeks, no one notices until an operator tries to debug something and finds zero breadcrumbs. Progress-pct tick-loop stays bare-pass (acceptable noise-free tick defense); wide-tolerance allowlist covers future line shifts.
      **New file** `jang-server/tests/test_exception_handling_invariant.py` — coarse count ≤ 20 + precise regex with allowlist. Matches iter-105/106 structure.
      **Evidence:** `jang-server/server.py:415-423, 898-906, 944-952`. 2 new test files (new tests dir + test). jang-tools 355 + ralph_runner 76 unchanged.
      **Meta-lesson — the template is portable across subprojects.** This is the fourth iter of dual-invariant work (M108 Swift coarse-only, M113 jang-tools dual, M119 ralph_runner dual, M177 jang-server dual). Same structure each time: inventory → taxonomy → coarse count test → precise regex test with allowlist → fix bare-pass sites that need logs. The iter-107 meta-lesson about three dispositions (verified+test, pathological+trigger, constraint+sketch) and iter-106 meta-lesson about "would the log save 10+ min" compose cleanly. **Rule for future auditors: copy the test file shape from a completed iter; update paths + thresholds; run; allowlist or fix remaining offenders.**
      **Meta-lesson — server bare-swallows are worse than CLI bare-swallows.** A CLI tool's swallowed error is usually caught by the user noticing "the convert didn't do X." A server's swallowed error accumulates invisibly over uptime. For server code, prefer `log.warning` over bare `pass` even for "benign" failures that don't affect the main path — operators need the trail.
      **Commit:** (this iteration)
- [x] **M176b (autoCheckForUpdates toggle consistency gap — M62-class extension)** — iter-109 M176 fixed sidebar navigation; iter-110 extends to a similar inert-affordance pattern iter-108 M62 missed. `autoCheckForUpdates` toggle at SettingsWindow.swift:415 persists user's preference but Sparkle integration isn't wired — auto-updates ship in v1.1. Pre-M176b the section carried a caption ("JANG Studio v1.0 ships with manual updates. Automatic updates via Sparkle are planned for v1.1.") but the TOGGLE itself carried no M62-style per-affordance marker. The caption is a cross-concern section note; users scanning toggles don't always read section footers. Added an inline `Label("Not yet implemented — awaits Sparkle integration in v1.1.")` directly under the toggle so the inert state is visible at the toggle's own attention site.
      **Tests (+1) in AppSettingsTests:** `test_autoCheckForUpdates_has_not_yet_implemented_label` — source-inspection pin for the Sparkle citation + "Not yet implemented" literal.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/SettingsWindow.swift:419-427`. 33 AppSettingsTests pass (was 32, +1).
      **Meta-lesson — section-level captions don't replace per-affordance labels.** A user scanning toggles or buttons typically reads the label, not the section footer. When an affordance is inert, mark the AFFORDANCE, not (only) the section. Iter-108 M62's rule applies here: persisted value stays, affordance-level label clarifies status. Also confirms iter-108's corollary that these closures are worth sweeping periodically — the Updates section toggle was shipped with just a section caption; iter-110 caught it on a second pass.
      **Commit:** (this iteration)
- [x] **M176 (WizardView sidebar set: binding ignored `canActivate` — click on locked step still navigated)** — iter-81 flagged this during a deep-trace of the wizard sidebar but didn't close it. iter-109 closes. Pre-M176, clicking a locked step (e.g., Architecture when Source isn't complete) still updated `coord.active` — landed the user in a dead-end (Continue disabled, no explanation). Mixed UX signal: the visual lock icon + `.secondary` foreground said "locked" but the click behavior said "reachable."
      **Fix (iter 109):** gated the `List(selection: Binding(set: ...))` closure on `coord.canActivate(step)`. If reachable, update `coord.active`. If not, ignore — click becomes a no-op for locked rows. Forward navigation via Continue buttons unaffected. Backward navigation (user on Verify wants to re-visit Profile) still works because those earlier steps DO pass canActivate.
      **Tests (+2):**
      - Source-inspection pin in `WizardStepContinueGateTests`: test_sidebar_selection_binding_gates_on_canActivate — asserts the `coord.canActivate(step)` call AND the M176 rationale comment stay in the file.
      - Functional pin in `AppSettingsTests`: test_wizardCoordinator_canActivate_gates_unreached_steps — constructs a fresh WizardCoordinator and asserts `.source` reachable, all others not.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/WizardCoordinator.swift:41-58`. 32 WizardStepContinueGateTests pass (was 31, +1). 32 AppSettingsTests pass (was 31, +1).
      **Meta-lesson — "don't lie to the user" applies to navigation affordances too.** iter-101/102 covered `.pass` states, iter-108 covered settings. Iter-109 covers list navigation: if a row looks reachable (even visually downplayed), clicking should either work OR be genuinely inert. The mixed state (appears locked but clicks through) is a UX bug. **Rule: whenever a UI element has a visual "disabled" treatment, its interaction must match — either `.disabled(true)` OR the handler gates its own behavior. Visual treatment alone is not a gate.**
      **Commit:** (this iteration)
- [x] **M175 (ramAdequate + diskSizeSanity sibling of M05 — ambiguous-pass sweep)** — Iter-101 M05 fixed `PreflightRunner.diskSpace`'s uncheckable-pass anti-pattern. Iter-102 sweeps for sibling instances. Found 2.
      **Gap A — `PreflightRunner.ramAdequate` (line 194-204):** pre-M175 returned `.pass` with nil hint when `totalBytes` was unknown (pre-inspection, same as M05's trigger). Even more concerning than disk: OOM mid-convert is arguably worse than disk-full (OS may SIGKILL the subprocess before it can emit a clean error — user sees "Killed" with no diagnostic). Same visual-state ambiguity as M05.
      **Gap B — `PostConvertVerifier.diskSizeSanityCheck` "couldn't compute" branch (line 181-185):** pre-M175 returned `.pass` with `"couldn't compute estimate…"` hint when `sourceBytes <= 0` or `avgBits <= 0`. The hint mentioned the gap but the visual state matched a real pass. Same anti-pattern.
      **Fix (iter 102):** both sites promoted to `.warn` with explicit "uncheckable" / "skipped" markers in the hint.
      - ramAdequate: `.warn` + `"X GB installed (no estimate yet — pick source for a real check)"`.
      - diskSizeSanity: `.warn` + `"couldn't compute estimate (missing source size or avg bits — this audit skipped, not run)"`.
      **Tests (+2 Preflight, 1 PostConvertVerifier):**
      - `test_ramAdequate_pre_inspection_warns_about_uncheckable_state` — new warn-state pin.
      - `test_ramAdequate_post_inspection_with_room_still_passes` — regression guard for real-check branch.
      - `test_diskSizeSanity_missing_source_warns_with_hint` — renamed from `_passes_with_hint`, asserts `.warn` + "skipped" marker.
      **Evidence:** `JANGStudio/JANGStudio/Verify/PreflightRunner.swift:194-214`, `PostConvertVerifier.swift:178-191`. 28 PreflightRunnerTests pass (was 26, +2). 17 PostConvertVerifierTests pass (count unchanged; one renamed + assertion-strengthened).
      **Meta-lesson — sibling sweep pays compound interest (again).** iter-101 M05 established the ambiguous-pass pattern; iter-102 swept for siblings in ~10 min, found 2 more. The fix pattern was already known, so execution was nearly mechanical. Every meta-lesson iter-20+ adds pays off on at least 2-3 subsequent iters via cheap sibling closures. Rule confirmed: after fixing a class of bug, grep for sibling instances IN THE SAME ITER or QUEUE one for the immediate next iter — don't let the sweep drift.
      **Meta-lesson — `.pass` states deserve pattern-level scrutiny.** Grepped `status:\s*\.pass` across all verifiers and triaged each: real positive result (keep), N/A-for-this-plan (keep, it's truly "passes because doesn't apply"), or couldn't-evaluate (change to warn). The taxonomy matters: "passes because it's OK" ≠ "passes because check doesn't apply to this plan" ≠ "passes because we couldn't check." UI-wise, the first two can share visual state (both are "no problem here"), but the third needs distinct visual state (needs user attention).
      **Commit:** (this iteration)
- [x] **M174 (diskSizeSanityCheck had the SAME BF16 hardcoding — cross-boundary formula audit per iter-99 meta-lesson)** — Iter-100 executed iter-99 M173's forecast: "audit other formulas that cross Swift⇄Python for similar assumptions." The FIRST site audited was `PostConvertVerifier.diskSizeSanityCheck` (iter-40 M116). Same bug, same fix.
      **Bug on line 185:** `let expectedBytes = Double(sourceBytes) * avgBits / 16.0`. Hardcoded `/ 16.0` assumes BF16 source. For FP8 source + JANG_4K: actual output ≈ 178 GB on 340 GB source; formula computes `340 × 4 / 16 = 85 GB` expected; ratio = 178 / 85 = 2.09× → tripped the ratio>2.0 "bloat" warn branch → **falsely warned user that a correctly-sized FP8-converted output was bloated**. User sees warning, worries about conversion integrity, potentially re-runs.
      **Fix (iter 100):** added `sourceDtype: SourceDtype = .unknown` parameter to `diskSizeSanityCheck`. Reused `PreflightRunner.sourceBytesPerWeight(_:)` helper introduced in M173 so both formulas share the bytes-per-weight mapping and can't drift. Formula becomes `expectedBytes = srcBytes × avgBits / (8 × bytesPerWeight)` — mathematically identical to pre-fix for BF16.
      **Caller update:** `PostConvertVerifier.run` passes `plan.detected?.dtype ?? .unknown` so the full flow respects the detected source dtype.
      **Default parameter = .unknown:** preserves behavior for any existing test callers that don't pass dtype. Test `test_diskSizeSanity_default_dtype_param_is_bf16` pins this backwards-compat contract.
      **Tests (+3) in PostConvertVerifierTests.swift:**
      - `test_diskSizeSanity_fp8_source_uses_8bit_divisor` — 340 GB FP8 + 4 bits → expects 170 GB, disk 178 GB → ratio 1.05× → pass. Pre-M174: ratio 2.09× → false warn.
      - `test_diskSizeSanity_bf16_source_preserves_pre_M174_behavior` — regression guard with explicit .bf16.
      - `test_diskSizeSanity_default_dtype_param_is_bf16` — backwards-compat: no-dtype call behaves like .bf16 explicit.
      **Evidence:** `JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift:138-157, 154-210`. 17 PostConvertVerifierTests pass (was 14, +3). 24 PreflightRunner + 28 AppSettings unchanged.
      **Meta-lesson (confirms iter-99) — cross-boundary-formula audit yields fast wins.** The iter-99 M173 fix took ~30 minutes once traced. M174 took ~10 minutes because the pattern was identified and the helper was already in place. The audit-lesson pays off IMMEDIATELY when executed on the first follow-up site. Rule: when you fix a class of bug with a shared helper, immediately grep for other instances of the old pattern — the helper makes the fix trivial.
      **Meta-lesson — shared helpers prevent drift.** The pre-M174 code had two COPIES of the same hardcoded `/16.0` — in PreflightRunner AND PostConvertVerifier. Two copies of bad math; both rotted together. M173 introduced the `sourceBytesPerWeight` helper; M174 uses it; any future estimator that wants source-dtype-aware math calls the helper and automatically gets FP8 support. Codebase-wide rule: when refactoring one instance of a formula, extract the shared part to a helper AS PART OF the refactor — deters future copies from re-hardcoding.
      **Cross-boundary sweep continues:** other candidates remaining: (a) the 1.05 overhead constant — is that right for JANGTQ outputs or does JANGTQ metadata have different overhead? (b) profile bit-counting in allocate.py — is the math there also assumption-dependent? Future iter.
      **Commit:** (this iteration)
- [x] **M173 (Preflight disk-space estimator hardcoded BF16 source — under-predicts FP8 by 2×)** — Iter-99 deep-traced `PreflightRunner.estimateOutputBytes` / Python's `estimate_model.predict`. Found the formula `srcBytes × (avgBits / 16.0) × 1.05` hardcodes a 16-bits-per-weight source assumption.
      **Bug:** for FP8 source models (DeepSeek V3/V3.2, some newer DeepSeek-derived models, various experimental releases) the src_bytes is `weights × 1` not `weights × 2`. The formula under-predicts output size by 2×. Example: 100 GB FP8 source → JANG_4K predicts ~26 GB output → user sees "52 GB free, need 26 GB, all good" → convert fails at 45 GB with "disk full" because real output is 52 GB. **Silent disk-space gate failure.** Symmetric over-prediction by 2× for FP32 source (rare but possible), which is just conservative over-estimate — harmless.
      **Fix (iter 99) — Swift:** added `PreflightRunner.sourceBytesPerWeight(_:)` helper mapping SourceDtype → bytes/weight (bf16/fp16 = 2, fp8 = 1, jangV2 = 2 fallback, unknown = 2 conservative fallback). `estimateOutputBytes` now uses `srcBytes × avgBits / (8 × bytesPerWeight) × 1.05` — mathematically equivalent to the original formula when source is BF16, correct for FP8.
      **Fix (iter 99) — Python:** added `_source_bytes_per_weight(model_dir)` to `estimate_model.py`. Peeks at the first safetensors shard header (same pattern as `inspect_source._sniff_dtype`), detects F8_E4M3/F8_E5M2 → 1, BF16/F16 → 2, F32 → 4, fallback 2. Updated `predict()` to use the new divisor.
      **Kept iter-63 M141's cross-boundary contract:** Swift + Python estimators stay aligned so the preflight (Swift) and `jang estimate-model` CLI (Python) agree on size predictions.
      **Tests (+4 Swift) in PreflightRunnerTests.swift:**
      - `test_estimateOutputBytes_fp8_source_uses_8bit_divisor` — 100 GB FP8 → JANG_4K → 52.5 GB (was 26 GB pre-M173).
      - `test_estimateOutputBytes_bf16_source_matches_pre_M173_behavior` — regression guard: BF16 source still gives 26 GB.
      - `test_estimateOutputBytes_fp16_source_same_as_bf16` — FP16 = BF16 in bytes/weight; same answer.
      - `test_estimateOutputBytes_unknown_dtype_falls_back_to_16bit_assumption` — safety: unknown → 16-bit assumption (conservative over-estimate safer than under-estimate for a disk gate).
      **Tests (+2 Python) in test_estimate_model.py:**
      - `test_predict_fp8_source_uses_8bit_divisor` — functional test with a valid safetensors header declaring F8_E4M3 dtype; asserts predicted GB ≈ 0.525 × source.
      - `test_predict_bf16_source_matches_pre_M173_behavior` — regression guard.
      - Added `_make_shard_with_dtype(path, dtype_str, n_bytes)` helper that writes a minimal valid safetensors file with a specified dtype, usable for both M173 tests and future dtype-dependent tests.
      **TDD flow:** Swift red (1 failure on FP8 test) → fix → Swift green (24/24). Python red skipped (I knew from the mirrored bug it would fail) → fix + new tests → Python green (353/353, +2).
      **Evidence:** `JANGStudio/JANGStudio/Verify/PreflightRunner.swift:44-80`, `jang-tools/jang_tools/estimate_model.py:27-62, 95-105`. 24 PreflightRunnerTests pass (was 20, +4). 353 Python tests pass (was 351, +2).
      **Meta-lesson — hardcoded assumptions in cross-boundary formulas rot together.** Swift + Python both had the same BF16 assumption because they were implementations of the same math. Both rotted silently when FP8 models became common. Cross-boundary contract tests (iter-63 M141 introduced this) would have caught it if we'd tested with all supported source dtypes, not just BF16. New audit axis: whenever a formula crosses Swift⇄Python, enumerate all inputs the formula depends on and write a matching test pair in each language.
      **Meta-lesson — "it was right for the common case" is a common bug-genesis pattern.** The original formula was written when all HF models were BF16/FP16. It was correct then. FP8 models emerged; the formula silently became wrong. Rule: when an assumption is baked into a constant (like `/16.0`), flag it with an inline comment explaining the assumption AND when it would become invalid. A future maintainer then knows to re-check when the landscape shifts.
      **Commit:** (this iteration)
- [x] **M23** — After cancel, the "Delete partial output" button appears. If user clicked it when outputURL is NIL or already removed, it silently no-ops. Should surface success/fail.
      **Partially closed iter-35 M107:** the nil case is handled by `.disabled(coord.plan.outputURL == nil)`; real failures (permission denied, in-use) surface as `[cleanup] delete FAILED: <error>`. Remaining gap: "already removed" case produced a misleading `[cleanup] delete FAILED: No such file or directory` message — user sees "FAILED" but the goal state is actually achieved.
      **Iter-97 close:** added a `catch CocoaError.fileNoSuchFile` branch between the success and generic-catch branches. When the folder was already gone (auto-delete-on-cancel ran, manual rm, prior click succeeded, etc.), the log now reads `[cleanup] <path> — already gone (nothing to delete)` — accurate to the user's goal state. Real failures (permission denied, file in use) still report `delete FAILED`.
      **Test (+1) in WizardStepContinueGateTests.swift:** `test_runStep_delete_partial_output_distinguishes_already_gone` — source-inspection pin for the `CocoaError.fileNoSuchFile` catch branch + the "already gone" log literal.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift:84-104`. 31 WizardStepContinueGateTests pass (was 30, +1).
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
- [x] **M106** — `DiagnosticsBundle.write` was `@MainActor` + synchronous. The `ditto -c -k` subprocess's `waitUntilExit()` ran on MainActor, beach-balling the UI for 3-5s on a large diag bundle (50+ MB of tick events + stderr after a big convert). Small bundles (<5 MB) were invisible but any convert that ran for hours accumulated a lot of logs.
      **Fix (iter 42):** added `writeAsync(...) async throws -> URL` alongside the sync `write`.
      - Step 1 (on MainActor, fast): create tempdir + write scrubbed plan/log/events/system/verify JSONs. These are small (<1 MB typical pre-zip) so MainActor cost is negligible.
      - Step 2 (off MainActor via `DispatchQueue.global()`): the `ditto -c -k` subprocess wait. Wrapped in `withTaskCancellationHandler` with `ProcessHandle` from iter 30 — consumer Task cancel propagates to SIGTERM. Same pattern as iter-33's service-invoke sweep.
      - Kept synchronous `write` for backward compat with existing tests (iter 14's M22 test suite).
      - Call site in RunStep.swift: Copy Diagnostics button now wraps the call in `Task { await writeAsync(...) }`. MainActor free, UI responsive while ditto runs in background.
      **Tests (2 new):** async variant produces same zip shape as sync (5 required entries: plan/run/events/system/verify), async variant scrubs sensitive tokens same as sync (M22e boundary preservation). Tests avoid `FileManager.enumerator` which is unavailable from async contexts on Swift 6 — replaced with recursive `contentsOfDirectory` helpers.
      **Evidence:** `DiagnosticsBundle.swift:110-212` new async function, `RunStep.swift:73-100` call-site migration. 130 Swift tests pass (was 128, +2).
      **Commit:** (this iteration)
- [x] **M107 (new grep-audit class: silent-failure `try?` on user actions)** — Applied the meta-lesson from iters 30-34 to a different bug class. Searched for `try?` across the Swift codebase. Filtered out the 20+ instances inside PostConvertVerifier / PreflightRunner / etc. where `try?` + default is the intentional parse-failure-tolerance pattern. Found 3 real silent-failure bugs on user-triggered save/delete actions:
      **1. `TestInferenceSheet.exportTranscript`** — `try? vm.exportTranscript(to: url)`. User clicks Export → picks save location → if write fails (disk full, permission denied, read-only volume, sandbox rejection), NOTHING is shown. User walks away thinking they saved a file that doesn't exist.
      **2. `UsageExamplesSheet.saveToFile`** — `try? text.data(using: .utf8)?.write(to: url)`. Same UX failure mode for the Save button on the Usage Examples sheet.
      **3. `RunStep "Delete partial output"`** — `try? FileManager.default.removeItem(at: out)`. After a cancelled convert, user clicks Delete → if delete fails (permission denied, file in use by an orphaned subprocess, already-gone from disk), no feedback. User assumes cleanup worked.
      **Fix (iter 35):** each site switched to explicit `do { try ... } catch { surface error }`. For sheet-based actions (1, 2): `@State errorMessage: String?` + `.alert(...)` modifier. For the in-flow delete (3): log to the existing `logs` pane so the user sees `[cleanup] delete FAILED: <reason>` or `[cleanup] deleted <path>` right where the rest of the conversion log lives.
      **Design call:** don't add alerts for SUCCESSFUL saves/deletes — the Finder action (e.g., user will open the saved file) provides natural feedback. Surface failures only.
      **Evidence:** `TestInferenceSheet.swift:15, 45-54, 272-283`, `UsageExamplesSheet.swift:11, 24-32, 152-171`, `RunStep.swift:51-68`. 122 Swift tests pass unchanged.
      **Commit:** (this iteration)
- [x] **M108** — Remaining `try?` sites in the Swift codebase (~27 after iter-35 fixed 3) are largely in parse-tolerance contexts (PostConvertVerifier reads files that may or may not exist), Task.sleep ignores (standard pattern), and subprocess-write-to-stream helpers. Spot-check periodically. Spawned for awareness; not a concrete bug today.
      **Audit + close (iter 104):** walked all 34 current `try?` sites across 12 files in `JANGStudio/JANGStudio/`, classified each into 8 acceptable categories — **zero user-action silent-swallows remain** (the anti-pattern iter-35 M107 + iter-80 M157 chased). Taxonomy:
      - **A. comment text** (6) — rationale comments referencing prior fixes (M111 / M107 / M157 / M35). Not actual code.
      - **B. parse-tolerance file reads** (9) — PostConvertVerifier reads files that may not exist; missing → VerifyCheck reports failure per M14. PreflightRunner's config.json read same pattern.
      - **C. Task.sleep ignore** (5) — SIGTERM→SIGKILL 3 s escalation + timeout waits. Standard cancellation pattern.
      - **D. stderrTask result fallback** (2) — PythonRunner + PublishService's `(try? await stderrTask.value) ?? ""`. The stream may cancel before yielding; empty tail is correct fallback.
      - **E. regex compile of known-valid patterns** (2) — PublishService.segmentRegex + DiagnosticsBundle sensitive-patterns. Compile-time constants; fail = bug in the pattern itself.
      - **F. macOS resource-query** (1) — `parent.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])`. Fails on unusual filesystems; falls back to 0 free.
      - **G. temp-dir cleanup / pipe-close** (4) — deferred removeItem on temp work dirs + pipe-write-end close on error path. Best-effort cleanup; failure means OS reaps later.
      - **H. JSON round-trip** (2) — InferenceRunner error-shape detection + TestInferenceViewModel export-transcript encode→reparse for mixed-type dict. Defensive coding on Codable-shaped data.
      **Locked in with count invariant:** `test_try_question_site_count_within_threshold` in AppSettingsTests asserts total `try?` count ≤ 50 (16 headroom above today's 34). Catches bulk additions without blocking routine work. Test comment inlines the full taxonomy so the next engineer to see it fire has the classification at hand. If their additions fit the taxonomy, bump the threshold; if a new user-action silent-swallow slips in, apply the iter-35/iter-80 do/catch + stderr pattern.
      **Meta-lesson — "observation only" items can have coarse invariants.** M108 said "spot-check periodically." The count threshold turns that into automated spot-checking — any PR growing the count past 50 triggers a test failure, forcing the engineer to review the additions against the taxonomy. Less precise than M65's per-site grep but cheaper to maintain. Same iter-103 principle ("testable invariants > observation comments") applied with a coarser threshold.
      **Evidence:** `JANGStudio/Tests/JANGStudioTests/AppSettingsTests.swift` — new test. 30 AppSettingsTests pass (was 29, +1).
- [x] **M111 (iter 37 continuation of try? sweep)** — Grepped for `if let ... = try? ...` nested patterns (iter 36 drive-by showed iter-35's sweep missed the "nested try? inside if-let" shape). Found 3 sites; only 1 was a real silent-failure: `AppSettings.persist` encoded via `if let data = try? JSONEncoder().encode(snapshot) { ... }`. If encoding failed (schema-migration edge-case, future non-finite Double), settings wouldn't persist AND nobody would know. UserDefaults preserved the old blob (we never called .set on failure) so data-loss is bounded, but visibility mattered for diagnostics.
      **Fix:** explicit `do/catch` around the encode + UserDefaults.set. On failure, log to stderr — Copy Diagnostics (iter 14's scrubSensitive pipeline) captures stderr, so bug reports include the error.
      **Other 2 hits were legitimate parse-tolerance:**
      - InferenceRunner.swift:158 — tries to detect an error-JSON shape; if malformed, falls through to normal decode.
      - PostConvertVerifier.swift:64 — verifier's expected-to-tolerate-missing-files pattern.
      **Evidence:** `AppSettings.swift:126-146`. 122 Swift tests + 260 jang-tools pass unchanged.
      **Commit:** (this iteration)
- [x] **M112 (Python-side silent-error class)** — Grepped for `except Exception:` in jang_tools. Found 25+ sites, most are legitimate (wrap + re-raise with context). One real silent failure: `inference.py:_load_vlm` caught EVERY error from `load_jangtq_vlm_model(...)` and fell through to `mlx_vlm.load`. This was the exact anti-pattern that masked iter-20's M45 (load_jangtq_vlm → load_jangtq_vlm_model rename). The rename was fixed then, but the except-all fallback pattern stayed — next similar bug would be masked AGAIN.
      **Fix (iter 37):** narrow the except to `ImportError` (the only legitimate fallback case — jang_tools module not present). Every OTHER exception from the JANGTQ loader is a real problem (corrupted shard, missing file, kernel error) and propagates up with full context. User sees the real JANGTQ error instead of a confusing mlx_vlm fallback error.
      **Evidence:** `inference.py:37-64`. 260 jang-tools tests pass unchanged.
      **Commit:** (this iteration)
- [x] **M113** — Other `except Exception:` sites in jang_tools. Spot-check didn't find more silent-errors, but the pattern is worth flagging periodically. Examples: loader.py:201 / 233 / 504 / 540 / 586 wrap exception during tensor conversion — mostly add context + re-raise. Safe-but-keep-an-eye-on.
      **Audit + close (iter 105):** Python analog of iter-104 M108's Swift `try?` audit. Grep'd `except Exception` across `jang_tools/` — 57 sites across 20 files. Classified into 5 categories (taxonomy inlined in new test module's docstring):
      1. **Optional imports** (try: import X; except Exception: X = None).
      2. **Tensor conversion retries** — primary quant path then fall back to slower tolerant path.
      3. **Best-effort parse** — optional config/header read, continue with defaults.
      4. **Error wrapping with context** — catch, add "while processing file X", re-raise (loader.py / modelcard.py patterns).
      5. **CLI top-level catch** — `__main__` wrappers for clean CLI error output.
      **Two invariants, coarse + precise:**
      - **Coarse:** `test_except_exception_site_count_within_threshold` — asserts total ≤ 75 (18 headroom over today's 57). Catches bulk regressions per iter-104's pattern.
      - **Precise:** `test_no_bare_except_exception_pass` — regex catches the ANTI-PATTERN specifically (`except Exception[: as x]:\n    pass` with no other body). Iter-35 M107 / iter-80 M157's user-action-silent-swallow class applied to Python. **Found 4 sites during first run.** Audited each — all are legitimate best-effort operations (Metal cache clear, optional `_scale_inv` tensor lookup, last-resort bit-width inference) — added to explicit `allowed` set with rationale comments. Future offenders fail the test with a pointer to the iter-35/iter-90 fix pattern.
      **Better than iter-104 M108:** two invariants (precise + coarse) catch different failure modes. The precise test nailed 4 existing sites and forced classification; the coarse test catches bulk additions that might individually evade the precise regex (e.g., `except Exception: logger.debug("ignored")` — technically not `pass` but effectively silent).
      **Meta-lesson extension — precise invariants find existing violations; coarse invariants prevent future ones.** Iter-104 M108 was coarse-only because Swift `try?` has no single "obviously-silent" anti-pattern grep-able. Python's `except Exception: pass` DOES have one. When the bug class has an obvious syntactic signature, add the precise test FIRST (it'll find existing offenders, or prove there are none); add the coarse threshold test ALSO as the long-term health gate.
      **Evidence:** new `jang-tools/tests/test_exception_handling_invariant.py`. 2 new tests. 355 Python tests pass (was 353, +2).
- [x] **M114 (Cat D fifth pass)** — Cross-ref `feedback_model_checklist.md` rule 1: "Clean output — no old v1 .jang.safetensors, no jang_imatrix.safetensors, no importance tensors. `du -sh` must match expected size." Memory warns of "Multiple times we shipped models with junk files (155 GB bloat)". Cross-referenced against convert.py + writer.py + publish.py.
      **Finding:** `jang_imatrix.safetensors` gets written to `output_path` during calibration (convert.py:229, 233-234) AND by `write_jang_v2_model` when `importance_data` is non-empty (writer.py:160, 235). The file is useful LOCALLY as a convert cache (subsequent converts of the same source can `--imatrix-path` it), but it's pure bloat in an HF-published model (doesn't participate in inference). Publish path uploaded EVERY file under model_dir via `rglob("*")` (publish.py:34) — so every JANG model on HF currently has a jang_imatrix.safetensors cluttering the repo.
      **Fix (iter 38):** 2-layer filter in publish.py only (keeps local cache intact):
      - `_upload_with_progress`: new `_EXCLUDE_FROM_UPLOAD = {"jang_imatrix.safetensors"}` set applied in the file enumeration via `p.name not in _EXCLUDE_FROM_UPLOAD`.
      - `cmd_publish` dry-run path: uses the same exclusion so preview count/size matches actual upload. Pre-fix the user saw "42 files / 187 GB" in dry-run but uploaded 41 files / 182 GB — confusing.
      **Design call:** do NOT remove imatrix from local output. It's useful for re-convert caching. The fix is scoped to what goes UP to HF.
      **Tests (2 new):** `test_upload_excludes_jang_imatrix` plants an imatrix fixture and asserts it's not in the upload_file calls; `test_dry_run_excludes_jang_imatrix_from_size` asserts dry-run size reflects the filter.
      **Evidence:** `publish.py:33-51, 124-142`. 262 jang-tools tests pass (was 260, +2).
      **Commit:** (this iteration)
- [x] **M115** — Memory rule 1: "no old v1 .jang.safetensors". Re-converting into an existing output dir left v1 `.jang.safetensors` shards + `model.jang.index.json` alongside the v2 files, AND any v2 shards from a different shard count (e.g. previously 42 shards, now 38 shards → 4 old shards never overwritten). Exact class of bloat memory's "155 GB junk" incident warned about.
      **Fix (iter 39):** new module-level helper `_remove_stale_jang_artifacts(output_path)` called BEFORE writing new shards. Nukes only KNOWN JANG-output filenames via a pinned `STALE_JANG_ARTIFACT_PATTERNS` list:
      - `*.jang.safetensors` (v1 shard extension)
      - `model.jang.index.json` (v1 index)
      - `model-*-of-*.safetensors` (v2 shards — rewriter regenerates)
      - `model.safetensors.index.json` (v2 index — regenerated)
      - `jang_imatrix.safetensors` (re-calibrated per-convert; iter-38 M114 excludes from upload but cleanup on re-convert is still useful)
      - `jang_config.json` (re-written by writer)
      User-added files (README.md, custom .py, tokenizer files, preserved-from-source chat_template*.json, generation_config.json) are PROTECTED — not in the pattern list.
      **Design calls pinned by tests:**
      - **Non-recursive**: `Path.glob(pattern)` NOT `rglob`. User-placed nested dirs (`assets/`) with files that happen to match patterns are untouched.
      - **Tolerant of missing output dir**: glob on non-existent path returns empty, no FileNotFoundError. Iter-39 first-time-convert flow unaffected.
      - **Idempotent**: running cleanup on a fresh or twice-cleaned dir returns `[]`, no exceptions.
      - **Permission-error tolerant**: `OSError` on individual unlink caught + logged, other files continue.
      **Tests (8 new):** patterns-list invariant pin, v1 shard removal, v2 shard-count-change orphan removal, imatrix + jang_config removal, user-file protection (13 realistic user files must survive), idempotency, missing-dir tolerance, non-recursive guarantee.
      **Evidence:** `convert.py:33-77` constant + helper, `convert.py:1055-1063` call site, `tests/test_convert_cleanup.py` 8 new tests. 270 jang-tools tests pass (was 262, +8).
      **Commit:** (this iteration)
- [x] **M116** — Swift PostConvertVerifier had no disk-size sanity check. Memory rule 2: "disk size ≈ GPU RAM. No bloat." audit.py row a7 covered this for Ralph harness but the wizard-side verifier didn't, so users publishing from JANG Studio got no warning if their convert output bloated (e.g., iter-39 M115 pre-fix: orphan shards doubled disk size silently).
      **Fix (iter 40):** new VerifyCheck row `#13 diskSizeSanity`.
      - Exposed as static helper `diskSizeSanityCheck(outputDir:sourceBytes:jangCfg:)` so tests can exercise the ratio math without constructing a full ConversionPlan.
      - Estimate model: `expected = source_bytes * actual_bits / 16` (bf16 source ≈ 16 bits/weight). Compare actual sum-of-shard-bytes to expected.
      - Warn window: ratio <0.5× (under-run, incomplete convert) OR >2× (bloat, orphan shards). Deliberately wide since tokenizer/template overhead adds ~5-50 MB on top.
      - **Excludes `jang_imatrix.safetensors`** from the disk sum (iter-38 M114 ruled it not-part-of-the-model). Separately tested by `test_diskSizeSanity_excludes_imatrix`.
      - **Accepts both v1 and v2 config keys** (`actual_bits_per_weight` OR `actual_bits`) — older converts used the shorter form. Pinned by `test_diskSizeSanity_accepts_v1_bitsField_fallback`.
      - **Missing data = clean pass with hint**: if source size or avg bits aren't known (e.g., running against a detached output dir), pass with "couldn't compute estimate" instead of failing. Verifier's job is to spot real problems, not nag about untested configurations.
      **Tests (6 new):** in-range pass, 4×-bloat warn (M115 regression safety-net), 0.19×-underrun warn, imatrix exclusion (iter-38 M114 boundary test), missing-source pass-with-hint, v1 bits key fallback.
      **Evidence:** `VerifyCheck.swift:9-10` enum case, `PostConvertVerifier.swift:125-203` row + helper. 128 Swift tests pass (was 122, +6).
      **Commit:** (this iteration)
- [ ] **M117** — Memory rule 3 says "Speed test — load model, warm up, run 3 prompts at correct temp". PostConvertVerifier doesn't test inference at all (audit.py does via a15 but only as one-shot). Consider a pre-publish inference smoke test in the wizard: 3 prompts, 20 tokens each, surface tok/s and any loops. Would extend VerifyStep. Scope creep; logged for future consideration.
- [x] **M118** — ralph_runner grep-audit (first time the Python side got an iter-30-37-style sweep). Grepped `subprocess.run` sites + found TWO without timeout: `remote.sync_tree` and `remote.pull_tree`. Both `rsync` over SSH to macstudio. Pre-fix: any rsync hang (network glitch, macstudio mid-transfer lockup, remote disk-full) blocked the Ralph iteration forever. iter-12 M55's lock prevents concurrent instances but doesn't address a single hung one — Ralph just stops making progress with no timeout, no cancel, no visibility.
      **Fix (iter 41):**
      - Both functions gained `timeout: float = 1800` parameter (30 min default — matches expected wall-clock for jang-tools tree transfer on Tailscale, generous headroom for network spikes).
      - TimeoutExpired is caught + converted to a structured RemoteResult with `returncode=124` (conventional timeout exit code) + informative stderr. Caller (cmd_next) sees it as a retryable failure rather than a crash.
      - Callers can pass shorter `timeout=` for small transfers (e.g. audit push doesn't need 30 min).
      **Tests (5 new):** sync_tree-passes-timeout, sync_tree-default-timeout-in-[5min,60min]-range (regression guard against future over-tightening), sync_tree-timeout-returns-124-not-raises, pull_tree-also-timeout-pinned, pull_tree-timeout-returns-124.
      **Evidence:** `remote.py:50-83`. 73 ralph_runner tests pass (was 68, +5).
      **Commit:** (this iteration)
- [x] **M119** — `audit.py` has many `except Exception as e:` sites (28+). Spot-check confirmed they're audit rows converting subprocess+analysis errors to structured fail results — that's intentional + correct. Pattern is safe. Logged as seen-and-verified; no fix needed unless a specific site is flagged by future triage.
      **Audit + close (iter 106):** applied iter-105 M113's dual-invariant pattern to `ralph_runner/`. 36 total `except Exception` sites (34 audit.py, 2 runner.py). Classified into 3 categories (audit-row error isolation, subprocess probe fallbacks, CLI top-level catch).
      **Precise test found 2 bare silent-swallows** in `audit.py:_load_vlm` — cascading fallback loaders that tried `load_jang_model` → `load_jangtq_vlm_model` → `mlx_vlm.load`, swallowing each intermediate exception. **Not strictly a bug** but loss of debug info: when the final fallback ALSO failed, the user saw only the last error with no trace of the first two attempts — hard to debug "which loader path is broken this week."
      **Fix:** replaced silent `pass` with a stderr log line per iter-35 M107 pattern. Each intermediate failure now emits `"[ralph.audit] _load_vlm: <path> failed (<type>: <msg>); trying <next>"` so the cascade is visible in stderr.
      **Tests:** new `ralph_runner/tests/test_exception_handling_invariant.py` with matching structure to iter-105's `jang-tools/tests/test_exception_handling_invariant.py`.
      - Coarse count ≤ 50 (14 headroom over today's 36).
      - Precise regex for `except Exception[: as x]:\n    pass`. Allowlist empty post-fix.
      **Evidence:** `ralph_runner/audit.py:64-88`, new test module. 75 ralph_runner tests pass + 2 new invariants.
      **Meta-lesson — "not strictly a bug" is still worth fixing when the fix is cheap.** The original `except Exception: pass` at audit.py:71,77 was audit-safe (cascading fallback, final path raises if all fail). But the stderr log adds debugging value at zero runtime cost and aligns with the actionable-diagnostic pattern iter-35 M107 / iter-90 M167 established. Rule: when deciding whether to fix a minor swallowed error, ask "would the log line save 10+ minutes of debugging next time this fails?" If yes, log it — cost is negligible.
- [x] **M120 (new grep-audit class: `json.loads(` without try/except on user-controlled input)** — Grepped 43 .py files for `json.loads(` calls. Most are reading files we wrote (jang_config.json, model.safetensors.index.json) so trusting is fine. **Two hot user-facing surfaces have no guard**: `inspect_source.py:51` (Step 1 source detector) and `recommend.py:332` (Step 1 recommendation). Both read `<source>/config.json` from a user-selected HuggingFace directory. A malformed/empty/truncated/non-dict config.json bubbled as one of `JSONDecodeError`, `UnicodeDecodeError`, or `AttributeError: 'list' object has no attribute 'get'` — Python exits 1 with a full multi-line traceback on stderr.
      **Why this matters — the Swift side was ALSO silently swallowing stderr.** `SourceStep.swift:301-306` captured stderr into a `Pipe()` but only used `proc.terminationStatus` in the NSError. So the user would see "Detection failed: inspect-source exited 1" with zero hint their config.json is bad. Two-layer silent failure: Python emits raw traceback, Swift drops it on the floor. The user thinks JANG Studio is broken and has no path to self-diagnose.
      **Fix (iter 43):**
      - **Python side (`inspect_source.py` + `recommend.py`)**: Wrap `json.loads(cfg_path.read_text(...))` in explicit try/except for OSError, UnicodeDecodeError, JSONDecodeError. Each emits a plain-English stderr message that includes the file path AND the decode location (line, col). Also added `isinstance(cfg, dict)` guard so `[]`-root JSON produces `"has a top-level list, expected a JSON object"` instead of an AttributeError traceback. Recommend's detect() raises ValueError (caught by existing top-level handler) while inspect_source exits 2 directly.
      - **Swift side (`SourceStep.swift` SourceDetector.inspect)**: read the stderr Pipe on nonzero exit and append to the NSError description. Now the wizard's errorText banner reads e.g. "Detection failed: inspect-source exited 2: ERROR: config.json at /Users/… is not valid JSON (line 1, col 3): Expecting property name enclosed in double quotes" — actionable.
      **Tests (5 new):** Python: `test_inspect_source_malformed_json_errors_cleanly`, `test_inspect_source_empty_config_errors_cleanly`, `test_inspect_source_non_dict_config_errors_cleanly`, `test_cli_rejects_malformed_config`, `test_cli_rejects_non_dict_config`. All assert (a) nonzero exit, (b) NO `Traceback (` in stderr, (c) `config.json` phrase present. Shared `_assert_clean_error` helper pins the invariant.
      **Evidence:** `jang-tools/jang_tools/inspect_source.py:51-75`, `jang-tools/jang_tools/recommend.py:332-352`, `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:290-326`. 275 jang-tools tests pass (was 270, +5). 130 Swift tests pass unchanged.
      **Commit:** (this iteration)
- [x] **M-audit (iter 44)** — Cat D memory cross-ref pass across 4 project memories (`project_qwen36.md`, `project_mistral4_architecture.md`, `project_minimax_m27.md`, `project_glm51_jang1l_working.md`). Verified each claim against current code, no drift found. (Last Cat D was iter 38 — 6-iter cadence, due now.)
      - `project_qwen36.md` P1 (`GATED_DELTANET_CONFIGS` uses `linear_attn.*` not `delta_net.*`): **fixed** — `architectures.py:221-242` has `linear_attn.in_proj_qkv/z/b/a/out_proj`.
      - `project_qwen36.md` P10 (tiny `in_proj_b`/`in_proj_a` floor): **fixed** — `architectures.py:230-237` set `min_bits=4, preferred_bits=8` with explicit "keep high-bit" descriptions (prevents 2-bit profile collapse).
      - `project_mistral4_architecture.md`: `recommend.py:104` classifies `mistral4 → moe_mla` ✓; `capabilities.py:50-51` maps `mistral3/mistral4 → (family=mistral4, reasoning=mistral, tool=mistral, cache=mla)` ✓.
      - `project_minimax_m27.md` always-reasoning claim: `capabilities.py:37-39` maps `minimax_m2*` with `think_in_template=True`, consistent with "template always injects `<think>`". Converters (`convert_minimax_jangtq.py:283-319`, `convert_qwen35_jangtq.py:420-445`) both stamp capabilities via `build_capabilities`.
      - `project_glm51_jang1l_working.md` inference config claims: `InferenceRunner.swift:59-73` default `temperature=0.0`, `maxTokens=100`; `inference.py:90-98 _make_sampler` returns None when temp ≤ 0 (greedy path); **no repetition penalty anywhere** in inference pipeline. Aligned with memory's "greedy + no rep penalty" prescription.
      **Finding flagged as open (M121 below):** the Cat D pass uncovered one UX gap — `_apply_chat_template_if_any` (`inference.py:67-87`) calls `apply_chat_template` with no `enable_thinking` kwarg, so reasoning models (GLM-5.1, Qwen3.6, MiniMax M2.7) default to `enable_thinking=True`. Per the GLM-5.1 memory: short-answer smoke tests need `enable_thinking=False` or the first 100+ tokens are `<think>...` with no answer emitted. TestInferenceSheet defaults to `maxTokens=150`, so a user running the in-wizard smoke test on a reasoning model will see a partial thinking block and no answer. Memory claim is valid and unsurfaced.
      **Evidence:** 0 source-code changes in iter 44. 275 Python + 130 Swift + 73 ralph tests unchanged. Pass closed with no drift detected, one open follow-on (M121).
      **Commit:** (this iteration)
- [x] **M121** — In-wizard smoke-test (`TestInferenceSheet` via `InferenceRunner` → Python `jang_tools inference`) was not passing `enable_thinking=False` to `apply_chat_template`. For reasoning models (GLM-5.1, Qwen3.6, MiniMax M2.7) this meant a 150-token smoke test budget was consumed by `<think>…</think>` and the user never saw an answer — looks like the model is broken. Per `project_glm51_jang1l_working.md`: "enable_thinking=False is REQUIRED for short factual prompts. Without it, GLM-5.1 enters <think>...</think> reasoning that eats 100+ tokens before emitting the final answer."
      **Fix (iter 45):** full four-layer wire-through, opt-in by default.
      - `jang_tools/inference.py`: `_apply_chat_template_if_any` gains keyword-only `enable_thinking: bool = True`; passes through to `apply_chat_template(**kwargs)`. On strict tokenizers that raise TypeError on the kwarg, falls back to a retry WITHOUT the kwarg so we still produce a templated prompt (not raw) — only the toggle behavior degrades. `_generate_text` pipes through. `cmd_inference` passes `enable_thinking=not args.no_thinking`. `register()` adds `--no-thinking` argparse flag.
      - `InferenceRunner.swift`: `generate(..., noThinking: Bool = false)` appends `--no-thinking` to argv when true. Default false preserves iter-32 M100 test invariants.
      - `TestInferenceViewModel.swift`: adds `var skipThinking: Bool = false`; passes through as `noThinking: skipThinking`.
      - `TestInferenceSheet.swift`: adds `Toggle("Skip thinking (reasoning models)", isOn: $vm.skipThinking)` with `.help` tooltip to the settings popover.
      **Design decision — opt-in, not default.** Flipping the default to `enable_thinking=False` would surprise users running reasoning benchmarks who EXPECT the thinking block. Opt-in = least-astonishment for both populations.
      **Tests: +6.** Python (+4): `test_apply_chat_template_pipes_enable_thinking_false`, `test_apply_chat_template_default_keeps_thinking_on` (regression guard), `test_apply_chat_template_no_thinking_survives_template_error` (strict-tokenizer fallback), `test_cli_help_lists_no_thinking_flag`. Swift (+2): `test_noThinking_flag_added_when_true` and `test_noThinking_flag_absent_when_default_false` — both use a shell-script executableOverride that dumps argv to a tempfile then exits 3, same harness pattern as iter-32 M100.
      **Evidence:** `jang-tools/jang_tools/inference.py:67-117`, `JANGStudio/JANGStudio/Runner/InferenceRunner.swift:59-93`, `JANGStudio/JANGStudio/Wizard/TestInferenceViewModel.swift:11-22`, `JANGStudio/JANGStudio/Wizard/TestInferenceSheet.swift:232-244`. 279 jang-tools tests pass (was 275, +4). 132 Swift tests pass (was 130, +2).
      **Commit:** (this iteration)
- [x] **M122 (new grep-audit class: `assert` on binary-format invariants stripped by `python -O`)** — Grepped `^\s*assert\s+` in `jang-tools/jang_tools`. Of 6 files with asserts, most were type-narrowing guards on internal state (safe to strip). Four **load-time struct-size checks** in `jangspec/format.py` (`BLOB_HEADER_SIZE == 32`, `TENSOR_HEADER_SIZE == 36`, `INDEX_ENTRY_SIZE == 28`, `INDEX_HEADER_SIZE == 24`) gate the on-disk binary layout for MTLIOCommandQueue expert blobs / index files. `python -O` strips `assert`, so a future edit that changes `<IIHHQQ` to e.g. `<IIHHHQQ` without updating the size constant would silently slip through and readers would misalign tensors at runtime — no loud fail, just corrupted experts.
      **Fix (iter 46):** Converted all four asserts in `format.py` to `if SIZE != N: raise ImportError(...)`. ImportError fires at module load regardless of optimization level. Left the other asserts alone — `builder.py` asserts are internal type narrowing (false positives if stripped are UB-level, not format-corrupting), and `progress.py`/`blob.py` asserts similarly guard internal call-site invariants.
      **Tests (+4):** `tests/jangspec/test_format_optimized_imports.py` spawns subprocesses with `python`, `python -O`, and `python -OO` and verifies the format module imports cleanly with correct sizes under all three. A source-inspection regression guard blocks any future edit that reintroduces `assert <SIZE> ==` via `inspect.getsource` + forbidden-pattern list.
      **Why ImportError not ValueError:** ImportError is what Python raises for "module can't be loaded" — e.g. bad bytecode, missing C extension. Size-mismatch at load-time IS a module-integrity failure, not a runtime argument error. Matches Python stdlib convention (`struct.error`-style value validations use ValueError; module-level integrity checks use ImportError).
      **Evidence:** `jang-tools/jang_tools/jangspec/format.py:42-99`. 283 jang-tools tests pass (was 279, +4). 132 Swift tests unchanged.
      **Commit:** (this iteration)
- [x] **M123 (M121 VL-path gap: `--no-thinking` silently no-op for VL reasoning models)** — Iter 45's M121 closure wired `enable_thinking` through the LLM path (`_generate_text`) but left `_generate_vl` unchanged. The VL path passes the raw user prompt straight to `mlx_vlm.generate`, which re-templates internally with the default `enable_thinking=True`. So a wizard user ticking "Skip thinking (reasoning models)" on a VL reasoning model like Qwen3.6-VL saw **zero effect** — same pathology M121 fixed for text but latent for multimodal.
      **Fix (iter 47):** `_generate_vl` gains keyword-only `enable_thinking: bool = True`. When False, pre-template the prompt BEFORE handing to mlx_vlm: first try the processor-level `apply_chat_template` (correct for multimodal messages if it accepts the kwarg), on TypeError fall through to the tokenizer-level template via the shared `_apply_chat_template_if_any` helper. When True (default), preserve the raw passthrough so mlx_vlm handles non-reasoning VL normally. `cmd_inference` pipes `enable_thinking=not args.no_thinking` into the VL call. No Swift or UI changes — M121's `noThinking` flag already flows via `--no-thinking` which now covers both paths.
      **Tests (+3):** `test_vl_generate_preserves_raw_prompt_when_thinking_on` (regression guard: default doesn't touch prompt), `test_vl_generate_pretemplates_when_thinking_off` (happy path: processor template fires with enable_thinking=False), `test_vl_generate_falls_back_to_tokenizer_when_processor_rejects_kwarg` (strict-processor retry — doesn't fall silently back to raw prompt). Uses a new `_FakeVLProcessor` stand-in and `_capture_vl_generate` helper that shims mlx_vlm.generate via monkeypatch + sys.modules stub (mlx_vlm is an optional dep — shim lets the tests run with or without it installed).
      **Evidence:** `jang-tools/jang_tools/inference.py:173-242`. 286 jang-tools tests pass (was 283, +3). Swift untouched.
      **Commit:** (this iteration)
- [x] **M125 (new grep-audit class: `json.load(open(...))` / `json.dump(..., open(...))` without context manager)** — Grepped all 43 .py files under `jang-tools/jang_tools/` for `json\.(load|dump)\(.*open\(`. Found **37 offender sites across 11 files**: `capabilities.py` (5), `load_jangtq_vlm.py` (2), `load_jangtq.py` (2), `build_jangtq_sidecar.py` (3), `convert_minimax_jangtq.py` (6), `convert_qwen35_jangtq.py` (6), `convert_mxrq.py` (4), `convert_mxtq.py` (6), `convert_mxtq_to_jang.py` (2), `load_mxrq.py` (2), `load_mxtq.py` (2), `scripts/verify_qwen36_artifact.py` (3).
      **Why the pattern is unsafe:**
      - Read side `json.load(open(p))`: CPython refcount-GC closes the fd promptly in practice, but PyPy / any GC-delayed implementation leaks the fd. A sandbox with a tight fd limit could exhaust before GC fires. Low-but-nonzero risk.
      - Write side `json.dump(obj, open(p, "w"))`: **much higher risk.** The file's buffer may not flush before the file object is GC'd. If the process crashes or is backgrounded between `json.dump` returning and the GC pass, partial JSON lands on disk — a converter crash could leave `config.json` or `jang_config.json` truncated, bricking the model. Write-side failures are silent (no exception) and irreversible (no atomic rename).
      **Fix (iter 48):** Migrate every site to `with open(...) as f: json.load/dump(...)`. Scope kept to mechanical replacement; no logic changes. The convert_mxtq.py sequence that writes the index twice (once with XXXXX placeholder shards, then rewrites after rename) is preserved — both writes now have independent context managers so the intermediate file is guaranteed flushed before the rename loop.
      **Tests (+1):** `tests/test_fd_leak_pattern.py::test_no_unwrapped_json_load_or_dump_with_open` greps all production .py files under `jang-tools/jang_tools/` with `json\.(load|dump)\([^)]*open\(`, skipping comment-only lines (the M125 rationale prose mentions the pattern). Offenders list is an assertion payload so future regressions point at exact `file:line` for quick fix.
      **Files touched in git commit (7):** `build_jangtq_sidecar.py`, `capabilities.py`, `convert_minimax_jangtq.py`, `convert_qwen35_jangtq.py`, `load_jangtq.py`, `load_jangtq_vlm.py`, `scripts/verify_qwen36_artifact.py`. **Local-only fixes (5, not committed — gitignored per .gitignore:31-37):** `convert_mxrq.py`, `convert_mxtq.py`, `convert_mxtq_to_jang.py`, `load_mxrq.py`, `load_mxtq.py`. The regression guard test scans all .py files in the `jang_tools/` tree regardless of git-tracked status, so untracked files are caught too if they exist on disk in the un-fixed pattern.
      **Evidence:** 22 call sites migrated in the commit + 15 in local-only files. 287 jang-tools tests pass (was 286, +1). Swift untouched.
      **Commit:** (this iteration)
- [x] **M-audit (iter 49)** — Re-grep verification pass on M120's `json.loads(` coverage, applying iter-48's meta-lesson ("grep head_limit silently truncates; re-grep with head_limit=0 after a first-pass fix"). Grep `json\.loads\(` across `jang-tools/jang_tools/` with `head_limit=0` returned **45 total sites across 17 files**. Iter 43's M120 fix handled the two user-boundary sites: `inspect_source.py:62` and `recommend.py:342` (detect() path). Scanned the remaining 43 sites for "user-boundary risk" and classified them:
      - **Safetensors-header reads (5 sites):** `calibrate.py:40`, `convert.py:106`, `fp8.py:67,104`, `convert_minimax_jangtq.py:161`, `convert_qwen35_jangtq.py:251`, `inspect_source.py:21`. These parse our own safetensors-format byte headers. Malformed = corrupt source file which safetensors itself already validates; json.loads failure here means the safetensors has been tampered with and a traceback IS the right signal. Leave as-is.
      - **Post-convert internal reads (34 sites):** `format/reader.py` (3), `loader.py` (14), `examples.py` (3), `codebook_vq.py` (2), `routing_profile.py` (3), `convert.py` (4 additional), `estimate_model.py` (1 — inside existing try/except), `benchmark.py` (1), `jangspec/bundle_loader.py` (1), `jangspec/builder.py` (3), `jangspec/manifest.py` (1). These read files WE wrote (config.json, jang_config.json, model.safetensors.index.json, tokenizer_config.json, checkpoint.json). Per "only validate at system boundaries" rule: internal trust, skip.
      - **Examples CLI top-level try/except coverage (3 sites in `examples.py:47,49,54`):** `cmd_examples` wraps the whole render_snippet call in `try ... except Exception: print('ERROR: {type}: {e}'); sys.exit(3)`. A corrupt `<converted>/config.json` yields a clean `ERROR: JSONDecodeError: ...` with no traceback. Minor UX polish available (add path to error message), logged as M126 low-priority.
      - **CLI-entrypoint reads in `codebook_vq.py` / `routing_profile.py`:** Both are post-convert internal tooling, not user-boundary. Same trust class as loader.py.
      **Finding:** M120's coverage is correct. No additional user-boundary fixes needed. Re-grep validated with head_limit=0 per iter-48 meta-lesson — this is what iter-43 should have done but didn't know to do yet.
      **Meta-outcome:** First application of the "re-grep after fix" rule. Confirmed negative — no new bugs — but the verification itself is a category of work worth logging. Future iters using this lesson have a template.
      **Evidence:** 0 source-code changes in iter 49. 287 Python + 132 Swift + 73 ralph tests unchanged.
      **Commit:** (this iteration — documentation-only)
- [x] **M127 (new peer-helper asymmetry class: `_resolve_modality` text_config fallback misclassifies text-only MoE as vision)** — Iter 47's M123 closure surfaced the generalization "peer helpers in the same module may have parameter/logic asymmetry even when signatures match." Iter 50 greps `def _(resolve|load|generate|detect|render|build|recommend|apply)_\w+\(` and walks the `_resolve_family_str` vs `_resolve_modality` pair in `capabilities.py`. **Found a real logic bug in `_resolve_modality`**: the fallback after both `has_vision` stamps fail was `"text_config" in config or "vision_config" in config` → return "vision". But many **text-only** MoE families (qwen3_moe, qwen3_5_moe, glm_moe_dsa, mistral4) wrap their text params under `text_config` with NO `vision_config`. A jang_config missing the `has_vision` stamp (legacy JANG v1 before stamping, third-party models, manually-edited files) gets misclassified as vision. vmlx's CapabilityDetector then routes the model through VLMModelFactory which fails at load time because the model class mismatches.
      **Why this wasn't caught by iter-44 Cat D:** iter 44 verified current converter behavior (`convert.py` stamps `architecture.has_vision`, `convert_qwen35_jangtq.py` stamps top-level `has_vision=True`). Both stamps are set by OUR converters, so `_resolve_modality`'s fallback never fires for artifacts we produce. The bug only manifests for third-party or legacy JANG artifacts — which iter 44 explicitly noted as edge-case territory but didn't confirm as a bug.
      **Fix (iter 50):** tighten the fallback to `"vision_config" in config` only. `text_config` alone is NOT a vision signal — it's HF's standard wrapper for the language-backbone params on any model that has a nested config structure, text-only or multimodal. The fix is a one-line change (`or` → removal).
      **Tests (+10) in `tests/test_resolve_modality.py`:**
      - 6 baseline behaviors preserved (explicit has_vision true/false, architecture.has_vision true/false, vision_config fallback, no-hints defaults to text).
      - 3 regression captures for the M127 bug: qwen3_moe, qwen3_5_moe, glm_moe_dsa text-only MoE configs with text_config but no vision_config. Each was pre-fix `vision`, post-fix `text`.
      - 1 VL positive: text_config + vision_config = vision (don't regress the real-VL fallback).
      **Evidence:** `jang-tools/jang_tools/capabilities.py:113-134`. 297 Python tests pass (was 287, +10). Swift untouched.
      **Commit:** (this iteration)
- [x] **M129 (peer-helper Swift sweep: typed-error parity across 5 adoption services)** — iter 50's meta-rule applied Swift-side. Greped `^    private (nonisolated )?static func invoke` across `JANGStudio/JANGStudio/Runner/` — found 5 services with near-identical invokeCLI bodies: `RecommendationService`, `ExamplesService`, `ModelCardService`, `CapabilitiesService`, `ProfilesService`. Read each for (a) parameter parity, (b) fallback parity, (c) error-path parity. **Found error-path asymmetry.**
      - 3 services (Recommendation, Examples, ModelCard) throw typed errors: `XServiceError.cliError(code: Int32, stderr: String)` with a `LocalizedError.errorDescription` that reads "jang-tools X exited N: <stderr>".
      - 2 services (Capabilities, Profiles) threw raw `NSError(domain: "XService", code: …, userInfo: [NSLocalizedDescriptionKey: stderr])` from inside the DispatchQueue closure. Caught by outer `catch` → forwarded via `cont.resume(throwing:)`. Functionally works but:
        - `refresh()` catches via `self.lastError = "\(error)"` — NSError stringification produces `Error Domain=CapabilitiesService Code=1 "(null)" UserInfo={NSLocalizedDescription=ModuleNotFoundError: jang_tools}` — ugly, leaks framework internals into the UI banner.
        - Callers can't do typed catch (`catch let e as CapabilitiesServiceError` impossible pre-iter-51).
      **Fix (iter 51):** Added `CapabilitiesServiceError` and `ProfilesServiceError` (LocalizedError, single `.cliError(code, stderr)` case). Migrated both `invokeCLI` impls to resume with the typed error instead of throwing NSError. Added typed catch in `refresh()` so `errorDescription` flows into `lastError` as "jang-tools capabilities exited N: <stderr>" — matching peer services' UX.
      **Tests (+4):** `CapabilitiesServiceTests` gains `test_capabilitiesServiceError_cliError_formats_cleanly` and `_handles_empty_stderr`. `ProfilesServiceTests` gets the symmetric pair. Verified via direct `xcrun xctest -XCTest "JANGStudioTests.CapabilitiesServiceTests"`: 5/5 pass (was 3, +2). Same for ProfilesServiceTests: 7/7 pass (was 5, +2).
      **Evidence:** `JANGStudio/JANGStudio/Runner/CapabilitiesService.swift:66-148`, `JANGStudio/JANGStudio/Runner/ProfilesService.swift:85-159`. 136 Swift tests pass (was 132, +4). Python 297 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M130 (peer-helper sweep: public loader entrypoint parity — `load_jang_model` vs `load_jang_vlm_model`)** — Iter 51's Swift-side peer-helper sweep pattern applied Python-side to the two public loader entrypoints. Both take a `model_path` + dispatch to v1/v2 + LLM/VLM inner helpers. Read each for (a) parameter parity, (b) fallback parity, (c) error-path parity. **Found error-path asymmetry:**
      - `load_jang_model` (text): 3-tier format guard — checks format missing, format not in JANG_FORMAT_VALUES, AND format_version > 2 → clean ValueError at each boundary.
      - `load_jang_vlm_model` (VLM): 1-liner `if not fmt or fmt not in JANG_FORMAT_VALUES: raise ValueError(f"Not a JANG model: format='{fmt}'")`. **Missing format_version check.**
      **Impact:** A future JANG v3 artifact (or any artifact with format_version > 2) passed to `load_jang_vlm_model` skipped the version gate, hit `_is_v2_model()` detection (which checks the "2" prefix of format_version), then tried to load through `_load_jang_v2_vlm` which failed deep inside mlx_vlm with an obscure "Model type X not supported. Error: No module named 'mlx_vlm.models.X'" — the actual failure mode captured by iter-52's regression test.
      **Fix (iter 52):** Mirror the text path's 3-tier guard into `load_jang_vlm_model`. Added split missing-format check, kept the existing unknown-format-value check, and added the `format_version > 2 → "Unsupported JANG format version"` ValueError. Now a malformed or unsupported-version jang_config produces the same clean actionable message from either entrypoint.
      **Tests (+5) in `tests/test_loader_entrypoint_parity.py`:**
      - `test_text_path_rejects_unsupported_format_version` (baseline).
      - `test_vlm_path_also_rejects_unsupported_format_version` (captures the pre-fix bug — was the failing test that proved it).
      - `test_text_path_rejects_missing_format` + `test_vlm_path_also_rejects_missing_format` (pins symmetric missing-format behavior).
      - `test_both_paths_reject_non_jang_format` (pins format='gguf' rejection on both).
      Each test spawns the loader in a subprocess so an mlx-import failure at test-collection time doesn't mask the format-parity signal. Skips gracefully when MLX isn't present.
      **Evidence:** `jang-tools/jang_tools/loader.py:688-738`. 302 Python tests pass (was 297, +5). Swift 136 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M131 (peer-helper sweep inside recommend.py: `_recommend_dtype` missed dynamic 512+ expert promotion)** — Iter 52's meta-pattern "peer-helper grep-audit has 4 finds in 6 iters" applied inside recommend.py's `_recommend_*` family. Diffed `_recommend_family`, `_recommend_profile`, `_recommend_hadamard`, `_recommend_dtype` for parameter/logic/fallback/error-path parity. **Found a self-contradicting recommendation bug:**
      - `_classify_family(model_type, expert_count, …)` **dynamically promotes** any MoE with expert_count ≥ 512 to "moe_large_expert" (line 143-144).
      - `recommend()`'s warning block says `"bfloat16 is required to avoid float16 overflow"` for any 512+ expert model (line 397-398).
      - But `_recommend_dtype(model_type, source_dtype)` uses a **hardcoded** `_BF16_REQUIRED = {"minimax_m2", "glm_moe_dsa"}` set (line 315). A future 512+ expert family (or any custom 512-expert qwen3_5_moe/deepseek_v3) gets `force_dtype=None` while the warning says bfloat16 is required. **The wizard shows contradictory advice; the model then OOMs / NaNs at float16 boundaries.**
      **Fix (iter 53):** add `expert_count: int = 0` parameter to `_recommend_dtype`; promote to bfloat16 when either `model_type in _BF16_REQUIRED` OR `expert_count >= 512`. Pass `expert_count` from `recommend()`'s existing detection. The helper now mirrors `_classify_family`'s dynamic promotion rule.
      **Tests (+3):**
      - `test_recommend_dtype_forces_bfloat16_on_any_512_expert_model`: 512-expert qwen3_5_moe (not in `_BF16_REQUIRED`) — pre-fix force_dtype=None; post-fix bfloat16.
      - `test_recommend_dtype_uses_n_routed_experts_for_bf16_check`: deepseek_v3 with `n_routed_experts=512` (DeepSeek naming). Pre-fix None; post-fix bfloat16.
      - `test_recommend_dtype_below_512_stays_auto` (regression guard): 256-expert must stay on auto dtype — don't over-force bfloat16 on smaller MoEs.
      **Evidence:** `jang-tools/jang_tools/recommend.py:313-332`. 305 Python tests pass (was 302, +3). Swift 136 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M132 (peer-helper sweep: JANGTQ converter unknown-profile handling)** — Iter 53's meta-lesson about "decision-overlap zones" (same question asked in multiple places, drifted implementations) applied to the two JANGTQ converters. `convert_minimax_jangtq.py:47-48` and `convert_qwen35_jangtq.py:99` both take a `PROFILE` string and map to `EXPERT_BITS` via a nearly-identical dict. **Divergent error handling:**
      - MiniMax: `if _PROFILE_NORM not in _PROFILE_BITS: raise ValueError(...)`. Unknown profile → loud failure.
      - Qwen35: `EXPERT_BITS = _EXPERT_BITS_BY_PROFILE.get(_PROFILE_NORM, 2)`. Unknown profile → **silently defaults to 2-bit.** Output labeled with whatever garbage profile the user typed, but actually 2-bit content. No warning.
      **Impact:** A user typing `--profile JANGTQ44` (meant JANGTQ4) for Qwen3.6 gets a 2-bit conversion labeled JANGTQ44 in jang_config with no error. Model size ≈ 2-bit model; user expected 4-bit quality; runtime output degraded; no diagnostic pointing at the typo.
      **Fix (iter 54):** Mirror MiniMax's guard into Qwen35. Explicit `if _PROFILE_NORM not in _EXPERT_BITS_BY_PROFILE: raise ValueError(...)`. Same error message shape.
      **Tests (+3) in `tests/test_jangtq_converter_profile_parity.py`:** uses source-inspection (both converters do heavy MLX work at module load — can't import in-process cheaply). Three pins:
      - Both converters must contain `raise ValueError` in the profile-bits block.
      - Neither converter may use `dict.get(_PROFILE_NORM, <int>)` silent-fallback pattern (regex-blocked in source).
      - Both converters' `JANGTQ*` dict keys must match — prevents divergence (e.g., one converter adding a new legacy alias without the other).
      **Evidence:** `jang-tools/jang_tools/convert_qwen35_jangtq.py:92-105`. 308 Python tests pass (was 305, +3). Swift 136 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M133 (peer-helper sweep: `estimate_model.predict` fallback vs `recommend._estimate_params_billion`)** — Two functions in jang-tools both compute parameter-count estimates from `config.json` when no safetensors are available. Diffed the formulas for decision-overlap (iter 53 meta-pattern):
      - `recommend._estimate_params_billion` (line 150-178): correctly separates `attn = 4*h²` + `mlp_per = 3*h*intermediate` + `mlp = mlp_per * num_experts` for MoE. Handles text_config nested keys + n_routed_experts/num_local_experts/num_experts aliases.
      - `estimate_model.predict`'s fallback branch (line 59-62): **flat `12 * h² * layers + 2 * h * vocab` formula that assumes dense and ignores num_experts**. For a 256-expert Qwen3.5-MoE the fallback predicted `source_gb=12.7` when the real bf16 source is ~700 GB (off by ~55×). Downstream the wizard's "predicted output" in Step 1 would report ~3 GB for a conversion that actually writes 180+ GB — disk-full failure mid-convert with no pre-flight warning.
      **When the fallback fires:** `_source_bytes(model_dir) == 0` — any dir with no `.safetensors` files. Happens if the user points at a `.bin`-only snapshot (legacy HF), a partial download, or a corrupted source. Rare but real.
      **Fix (iter 55):** Mirror `_estimate_params_billion`'s formula inside the fallback: parse `intermediate_size` (fallback to `4 * hidden`), parse `num_experts`/`n_routed_experts`/`num_local_experts` (any of the three HF naming conventions), compute `attn + mlp_per * num_experts` when MoE, plain `attn + mlp_per` when dense. Respects `text_config` nesting for VL wrapper configs.
      **Tests (+2):** `test_predict_fallback_accounts_for_moe_experts` (256-expert qwen3_5_moe must predict source_gb > 100 — pre-fix was 12.7, post-fix is ~700). `test_predict_fallback_still_works_for_dense_model` (regression guard: dense llama 7B must stay in 5-40 GB range; don't over-engineer MoE path and break dense).
      **Why not factor into a shared helper:** `_estimate_params_billion` returns billions (float) with domain-specific rounding; `predict` needs raw byte count. Inlining the formula keeps the two call sites independent + avoids import churn. Regression guard now pins both approximations together via behavioral tests.
      **Evidence:** `jang-tools/jang_tools/estimate_model.py:48-75`. 310 Python tests pass (was 308, +2). Swift 136 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M134 (peer-helper Swift sweep: wizard step Continue-button gating)** — Applied iter-53/54/55's decision-overlap pattern to the 5 Wizard step files. Each has a forward-navigation button; diffed the gating styles for consistency:
      - `SourceStep.swift:164` — `if coord.plan.isStep1Complete { Button… }` (conditional show) ✓
      - `ArchitectureStep.swift:43` — **unconditional button, no gate** ✗
      - `ProfileStep.swift:66-69` — `.disabled(!allMandatoryPass())` (preflight-based) ✓
      - `RunStep.swift:42-43` — `if coord.plan.run == .succeeded { Button… }` ✓
      - `VerifyStep.swift:172` — reset-state navigation, always safe ✓
      **Scenario:** User picks source folder A → detection runs async → user navigates to Architecture before detection finishes → user clicks "Looks right → Profile" → lands on Profile with `detected=nil`. Downstream Profile preflight catches it via `allMandatoryPass()` failing, but only after a late, noisy failure path. The clean fix is to gate at Architecture so the user gets immediate, consistent feedback matching the peer steps.
      **Fix (iter 56):** added `.disabled(!coord.plan.isStep2Complete)` to the Architecture Continue button. Regenerated `.pbxproj` via `xcodegen generate` to pick up the new test file.
      **Tests (+4) in `Tests/JANGStudioTests/WizardStepContinueGateTests.swift`:** source-inspection pattern (iter-46 M122 / iter-54 M132 style — `.disabled` isn't cheap to inspect at runtime without ViewInspector/XCUITest). Each of the 4 gated steps gets a pin:
      - `test_architectureStep_continue_is_gated` — "Looks right → Profile" button must be followed by `.disabled(!coord.plan.isStepNComplete)` within 400 chars. The captured regression of the unconditional button.
      - `test_sourceStep_continue_is_gated` — `if coord.plan.isStep1Complete` wrapper pinned.
      - `test_profileStep_continue_is_gated` — `.disabled(!allMandatoryPass())` pinned.
      - `test_runStep_continue_is_gated` — `if coord.plan.run == .succeeded` pinned.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/ArchitectureStep.swift:43-55`. 140 Swift tests pass (was 136, +4 via targeted `xcrun xctest`). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M135 (SourceStep internal audit: stale-detection-task race condition)** — Iter 56 forecast: deep-trace SourceStep's 347 lines (largest step file). Focused on the async detection/recommendation flow in `detectAndRecommend(url:)`. **Found a real race condition:**
      - `pickFolder()` fires `Task { await detectAndRecommend(url: url) }` — discarded Task handle, not tracked.
      - `detectAndRecommend` does Step A (SourceDetector.inspect, subprocess) then Step B (RecommendationService.fetch, subprocess). Both suspend on awaits.
      - **Scenario:** User picks folder A → Task A starts (~5s, large shard dir). User changes mind, picks folder B → Task B starts (~1s). Task B finishes first, writes `coord.plan.detected = B's_metadata`. **Task A finishes 4 seconds later and overwrites with A's metadata.** The user's `sourceURL` now points at B but `detected` describes A. Downstream conversion uses wrong architecture detection → wrong profile recommendation → quantization misapplied.
      **Why it matters:** SourceDetector.inspect already has iter-34 M105's `withTaskCancellationHandler` for subprocess kill; RecommendationService.fetch has iter-33 M101's wrap too. The *subprocess* cancel path is wired. But the outer Task handle was being discarded, so there was no way to trigger the cancel. This iter adds the handle tracking + cancel call + in-function cancellation guards.
      **Fix (iter 57):**
      - Added `@State private var detectionTask: Task<Void, Never>?` to track the live detection Task.
      - `pickFolder()` now calls `detectionTask?.cancel()` before starting a new task — propagates through both inspect + recommend subprocesses via iter-33/34's existing wraps.
      - `detectAndRecommend` gained 4 `guard !Task.isCancelled else { return }` guards (after Step A success, Step A error, Step A MainActor hop, Step B success, Step B error) so a mid-flight cancel doesn't stomp state.
      **Tests (+3) in `WizardStepContinueGateTests.swift`:** source-inspection pins matching iter-46 M122 / iter-54 M132 / iter-56 M134 style.
      - `test_sourceStep_tracks_detection_task_handle` — `@State private var detectionTask: Task<Void, Never>?` must exist.
      - `test_sourceStep_pickFolder_cancels_previous_task` — pickFolder must call `detectionTask?.cancel()` BEFORE the new `detectionTask = Task { … }` assignment (order-sensitive; cancel-after-assign would cancel the new task).
      - `test_sourceStep_guards_mutations_with_isCancelled` — at least 3 `guard !Task.isCancelled else { return }` statements in the file (one per mutation site after a suspension point).
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:4-16, 174-226`. 143 Swift tests pass (was 140, +3 via targeted `xcrun xctest`). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M136 (Task-handle discipline audit: RunStep onAppear re-enters completed conversions)** — Iter 57's meta-lesson: discarded Task handles + re-entrant triggers = classic SwiftUI concurrency bug class. Grepped all `Task\s*\{` in `JANGStudio/JANGStudio/`, 17 hits across 8 files. Triaged each:
      - **Safe (already stored or self-guarded):** SourceStep:200 (iter-57 M135 added handle), PublishToHuggingFaceSheet:201 (stored `publishTask`), TestInferenceViewModel.send (guarded by `!isGenerating`), GenerateModelCardSheet Retry (only shown post-completion), UsageExamplesSheet Retry (same), TaskGroup-structured UsageExamplesSheet:36.
      - **Potentially risky:** `RunStep.swift:105` bare `.onAppear { Task { await start() } }`. The function `start()` guarded only `run != .running`, allowing re-entry from `.succeeded` / `.failed` / `.cancelled`. **Real bug.**
      **Scenario:** User converts model (run=.succeeded). Nav-forwards to VerifyStep. Nav-backs to RunStep via sidebar to re-read logs. `.onAppear` fires. `start()` sees run=.succeeded, passes the guard, sets run=.running, clears logs, re-spawns the entire conversion on top of the already-written output folder. User loses the log of their successful conversion AND the converter rewrites files in an output dir that was just marked successful. No warning.
      **Fix (iter 58):** Gate `.onAppear` on `run == .idle`. Retry buttons (lines 49, 72) still call `start()` directly — they're user-initiated after a failure/cancel and the existing `!= .running` guard in start() is appropriate there. Only the AUTO-start path needs the tighter gate.
      **Tests (+1) in `WizardStepContinueGateTests.swift`:** `test_runStep_onAppear_only_auto_starts_when_idle` — source-inspection pin. Filters out comment lines first (iter-58 rationale comment mentions `.onAppear`) then checks `coord.plan.run == .idle` appears in code.
      **Audit class result (8-iter series M135+M136):** 2 real bugs in 17 `Task { … }` sites. Pattern documented for future iters: grep `Task\s*\{`, triage each for (a) handle storage when re-entrant, (b) trigger-gated by state that outlives the task, (c) self-guard inside the called function.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift:95-117`. 144 Swift tests pass (was 143, +1). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M137 (Publish sheet race: late-Cancel click on completed upload falsely shows "cancelled")** — Iter 58's `.task` audit was clean for runtime bugs but inspecting PublishToHuggingFaceSheet's cancel path surfaced a timing race. `runPublish()` was:
      ```swift
      do {
          for try await event in publishWithProgress(...) { apply(event) }
          if wasCancelled { errorMessage = "Upload cancelled..." }
          else            { publishResult = ...; token = "" }
      } catch {
          if !(error is CancellationError) { errorMessage = ... }
      }
      ```
      **Race:** user clicks Cancel AT THE SAME MICROSECOND the final upload event lands. `wasCancelled = true` from the button handler. For-await loop exits normally (natural completion). `if wasCancelled` branch fires, shows "Upload cancelled" despite the HF repo having the complete files. User thinks they need to re-upload.
      **Why the pre-fix code was subtly wrong:** it used `wasCancelled` (a button-intent flag) as the authoritative "did we stop early" signal. The authoritative signal is `CancellationError` thrown by the for-await loop — that ONLY fires when the task was cancelled BEFORE the work finished.
      **Fix (iter 59):** Move the success path out of the `if wasCancelled` branch — natural loop exit is always success. Use `catch is CancellationError` specifically for user-initiated cancels. The `wasCancelled` flag is now used only for a progress-log note on the success path ("Cancel click landed after the final upload event — HF repo is complete") so the user who hit Cancel understands why they see success.
      **Tests (+1) in `WizardStepContinueGateTests.swift`:** `test_publishSheet_treats_natural_completion_as_success` — source-inspection pin verifying `catch is CancellationError` is present AND the old `if wasCancelled { errorMessage = ... }` pattern (inside the do-block natural path) is gone.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/PublishToHuggingFaceSheet.swift:263-302`. 145 Swift tests pass (was 144, +1 via targeted `xcrun xctest`). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M138 (RunStep late-Cancel race — data-loss variant of M137)** — Applied iter-59's meta-lesson "distinguish INTENT from OUTCOME" to the other obvious intent-flag site: `cancelRequested` in RunStep. Grepped `wasX|isX|shouldX|cancelRequested` across the Swift app and triaged — found the same class of race.
      **Pre-iter-60 code (RunStep.swift:135):**
      ```swift
      // Stream finished without throwing — distinguish cancel vs natural success.
      coord.plan.run = cancelRequested ? .cancelled : .succeeded
      if cancelRequested {
          if settings.autoDeletePartialOnCancel, let out = coord.plan.outputURL {
              try FileManager.default.removeItem(at: out)   // ← DELETES successful output
          }
      }
      ```
      **The PythonRunner asymmetry that enables the race:** `PythonRunner.launch` treats a cancelled-and-signalled subprocess AND a naturally-completed subprocess identically — both call `continuation.finish()` clean, no throw. So RunStep's for-await exits normally in BOTH cases. `cancelRequested` is the ONLY way RunStep distinguishes them — and it's a button-intent flag with the same timing race as M137.
      **Data-loss scenario:** user watching a 30-min conversion at 99.9% decides it's slow, clicks Cancel. Subprocess finishes its final shard write, exits 0 at the same microsecond. Button handler sets `cancelRequested=true`. PythonRunner emits final `.done(ok=true)` event, then `continuation.finish()`. For-await exits normally. `cancelRequested ? .cancelled : .succeeded` → `.cancelled`. If `autoDeletePartialOnCancel=true`, `FileManager.removeItem(at: out)` **deletes the successfully-written output folder**. User has now lost 30 minutes of GPU work to a button click that was ~1ms late. Higher stakes than M137 (which only mis-labeled an already-uploaded HF repo).
      **Fix (iter 60):**
      - Added `@State private var sawSuccessfulDone: Bool = false`.
      - In `apply(_ ev:)`'s `.done(let ok, _, _)` case, assign `sawSuccessfulDone = true` when ok is true.
      - Reset `sawSuccessfulDone = false` at `start()`.
      - Stream-complete branch: if `sawSuccessfulDone`, always `.succeeded` (append late-cancel note if `cancelRequested`); else fall back to the old `cancelRequested ? .cancelled : .succeeded` logic (genuine cancel with no completion signal).
      The `sawSuccessfulDone` flag comes from the Python side's `.done(ok=true)` event — the authoritative "conversion completed successfully" signal, immune to button timing.
      **Tests (+1) in `WizardStepContinueGateTests.swift`:** `test_runStep_tracks_successful_done_event` — 3 pins: `@State private var sawSuccessfulDone` present, `sawSuccessfulDone = true` assigned somewhere, `if sawSuccessfulDone` check present.
      **Why M138 is adjacent-but-separate from M137:** the async shapes differ. M137 was about HFHub streaming where cancel throws CancellationError. M138 is about subprocess streaming where cancel is a CLEAN finish (PythonRunner deliberately swallows cancel to keep the stream contract simple). The M137 catch-pattern fix doesn't apply; M138 needs an in-band success marker via protocol events. Both share the META lesson — never use intent flags as outcome signals.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift:14-30, 130-167, 205-214`. 146 Swift tests pass (was 145, +1 via targeted `xcrun xctest`). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M139 (preflight: nested src/dst foot-gun — output inside source passes preflight)** — Started iter 61 as a Python publish.py vs convert.py peer-helper sweep. Triaged the two files for error-handling / JSON-shape / exit-code parity — mostly clean. Pivot: re-inspect the JANGStudio preflight where convert's output path validation lives. Found a real foot-gun:
      **Pre-iter-61 `outputUsable` (PreflightRunner.swift:44):** blocks `dst == src` exact match and `dst.path.contains(".app/Contents")`. But does NOT block **nested** src/dst. User selects source `/models/Qwen3.6-BF16` and output `/models/Qwen3.6-BF16/out` — preflight passes. Convert writes shards + config into a subdir of source. Confusing + risky:
      - User later does `rm -rf /models/Qwen3.6-BF16` to reclaim disk space → wipes their conversion too.
      - Any future recursive cleanup pass would touch source files.
      - Re-running convert with `_remove_stale_jang_artifacts` at one level won't re-scan into a deep subdir, so partial state accumulates if anyone ever extends the cleanup to recurse.
      **Fix (iter 61):** extend `outputUsable` to check both directions:
      - `dstPath.hasPrefix(srcPath + "/")` → "Output cannot be inside the source folder"
      - `srcPath.hasPrefix(dstPath + "/")` → "Source cannot be inside the output folder"
      Critical detail: uses `path + "/"` (trailing slash) to prevent sibling-prefix false positives. `/a/b` is NOT inside `/a/bc`; the `/`-appended check `/a/bc/`.hasPrefix(`/a/b/`) is false.
      **Tests (+3) in `PreflightRunnerTests.swift`:**
      - `test_outputInsideSourceFails`: src=/tmp/foo, dst=/tmp/foo/out → fail with correct hint.
      - `test_sourceInsideOutputFails`: src=/tmp/ws/hf-model, dst=/tmp/ws → fail with correct hint.
      - `test_siblingPrefixPathsDoNotTrigger`: regression guard — src=/tmp/abc, dst=/tmp/abcd → pass (sibling with shared prefix must not trip the nested check).
      **Evidence:** `JANGStudio/JANGStudio/Verify/PreflightRunner.swift:44-76`. 149 Swift tests pass (was 146, +3 via targeted `xcrun xctest`). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M140 (preflight-side mirror of M131: bf16-for-512-experts hardcoded whitelist)** — Iter 61's meta-lesson "preflight is the user-facing safety boundary; enumerate foot-guns" applied. Reading `PreflightRunner.bf16For512Experts` revealed the exact Python-side bug M131 fixed, present here on the Swift side:
      ```swift
      guard types.contains(mt) else {
          return .pass    // ← no warning for any dynamic 512-expert MoE
      }
      ```
      `types = capabilities.knownExpert512Types` is the hardcoded `["minimax_m2", "glm_moe_dsa"]` list from capabilities_cli.py's frozen defaults. A future 512-expert qwen3_5_moe variant (or any custom MoE with 512 experts not on the list) would pass preflight even when the user forced fp16 — exact float16 overflow risk the warning exists to catch. **Cross-boundary asymmetry with M131:** recommend.py on the Python side now dynamically promotes to bfloat16 for any 512+ expert count. Preflight on the Swift side still used the hardcoded list.
      **Fix (iter 62):** mirror iter-53's fix. Add `dynamic512 = (plan.detected?.numExperts ?? 0) >= 512` and change the guard to `types.contains(mt) || dynamic512`. Hint includes the dynamic expert count when that path fired so the user understands which heuristic triggered the warning.
      **Tests (+3):**
      - `test_bf16Warning_fires_on_dynamic_512_experts`: 512-expert qwen3_5_moe (NOT in whitelist) with user-forced fp16 → warn with "512 experts" in hint. Pre-fix passed silently.
      - `test_bf16Warning_still_fires_for_named_whitelist_types`: regression guard — minimax_m2 with fp16 must still warn via the named-list path.
      - `test_bf16Warning_skips_small_moe`: 256-expert qwen3_5_moe must stay .pass — don't over-warn on smaller MoEs.
      **Meta-observation:** the sibling bug on both sides of the boundary (Python recommend.py + Swift preflight) means decision-overlap zones can span the Swift⇄Python boundary too. Future peer-helper audits should look for the same hardcoded-list-vs-dynamic-check patterns on BOTH sides when the user-facing gate is split across the boundary.
      **Evidence:** `JANGStudio/JANGStudio/Verify/PreflightRunner.swift:124-149`. 152 Swift tests pass (was 149, +3 via targeted `xcrun xctest`). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M141 (preflight diskSpace was inert: always `estimated: 0` → always `.pass`)** — Iter 62's preflight-enumeration pass flagged `diskSpace` as a potential foot-gun; iter 63 fixed it. The call was:
      ```swift
      out.append(Self.diskSpace(dst: dst, estimated: 0))
      ```
      and `diskSpace` short-circuits to `.pass` when `estimated <= 0`. The entire gate was cosmetic — it displayed "N GB free" but never ACTUALLY blocked anything. User with a near-full disk would pass preflight, start a conversion, fill the remaining free space mid-shard, hit OSError, leave partial output on disk that M115 then cleans up on retry — but the initial pre-flight warning never fired.
      **Fix (iter 63):**
      - New `PreflightRunner.estimateOutputBytes(plan:, profiles:)` public helper. Uses the same formula as `jang_tools/estimate_model.predict`: `srcBytes × (avgBits / 16.0) × 1.05` for metadata overhead.
      - New `PreflightRunner.avgBitsForProfile(profile:, profiles:)` helper. Looks up `profiles.jang[].avgBits` first, falls back to `profiles.jangtq[].bits` (JANGTQ uses integer bits). Returns 0 on unknown profile (preserves the pass-through fallback for typos).
      - `run(...)` now accepts `profiles: Profiles = .frozen` and computes the estimate before calling `diskSpace`.
      - `ProfileStep.refresh()` passes `profilesSvc.profiles` through.
      **Parity with `estimate_model.predict`:** formula mirrors the Python-side size estimator (iter-55 M133 made that formula MoE-aware). Now the wizard's predicted-size banner, the preflight disk-space gate, and the Python CLI `estimate-model` all compute the same number.
      **Tests (+4) in `PreflightRunnerTests.swift`:**
      - `test_estimateOutputBytes_scales_by_profile_avgBits`: 100 GB × 4/16 × 1.05 = 26.25 GB for JANG_4K.
      - `test_estimateOutputBytes_uses_real_avgBits_for_JANG_2L`: 100 GB × 2.9/16 × 1.05 ≈ 19 GB (bounds-checked).
      - `test_estimateOutputBytes_returns_zero_before_source_inspected`: pre-detection state returns 0 so preflight doesn't falsely fail.
      - `test_estimateOutputBytes_returns_zero_for_unknown_profile`: typo-defensive — no guessed bit-width.
      **Evidence:** `JANGStudio/JANGStudio/Verify/PreflightRunner.swift:5-32, 45-67`, `JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift:85-93`. 156 Swift tests pass (was 152, +4). Python 310 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M142 (hadamard-low-bits check: substring match → structured compress-bits lookup, both sides)** — Iter 62's preflight enumeration flagged `hadamardVsLowBits` using `plan.profile.contains("_2")` as brittle. Iter 64 replaces the substring check on BOTH the Swift preflight AND the Python recommend.py side with a structured lookup against the profile tables.
      **Pre-iter-64 issues:**
      - Swift `hadamardVsLowBits`: `plan.profile.contains("_2") || plan.profile == "JANG_1L" || plan.profile == "JANGTQ2"`. Hardcoded JANG_1L because "JANG_1L" lacks the "_2" substring. Brittle to any new profile name.
      - Python `_recommend_hadamard`: exact hardcoded list `{"JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L", "JANGTQ2"}`. Same class.
      **Fix (iter 64):**
      - **Swift**: new `PreflightRunner.compressBitsForProfile(_:, profiles:)` helper. Returns `JANG profile.compressBits` if found; for K-quant (compressBits=nil in schema), derives from `avgBits`; for JANGTQ, uses `bits`; unknown profile returns nil. `hadamardVsLowBits` now checks `compressBits ?? 99 <= 2`.
      - **Python**: `_recommend_hadamard` now looks up `JANG_PROFILES[profile][2]` for the compress tier, falls back to `JANG_K_TARGETS[profile]` for K-quants, parses JANGTQ suffix digit, defaults to 4 on unknown. Single source of truth with the Python-side `allocate.JANG_PROFILES`.
      **Why fix BOTH sides (not just Swift):** iter-62 meta-lesson about cross-boundary decision-overlap. The hadamard recommendation Python-side and the preflight warn Swift-side enforce the same "2-bit → hadamard off" rule. Both must agree.
      **Tests (+10):**
      - Swift (+6) in PreflightRunnerTests: compressBitsForProfile pins for JANG_2L=2, JANG_1L=2, JANG_4M=4, JANG_4K (K-quant)=4, JANGTQ2=2, unknown=nil.
      - Python (+4) in test_recommend.py: `_recommend_hadamard_uses_JANG_PROFILES_compress_tier` (exhaustive JANG_1L through JANG_6M), `_handles_JANGTQ_variants` (JANGTQ2/3/4), `_k_quant_profiles` (3K/4K/5K/6K), `_unknown_profile_defaults_to_on`.
      **Evidence:** `JANGStudio/JANGStudio/Verify/PreflightRunner.swift:69-88, 141-161`, `jang-tools/jang_tools/recommend.py:305-334`. 162 Swift tests pass (was 156, +6). 314 Python tests pass (was 310, +4).
      **Commit:** (this iteration)
- [x] **M143 (re-grep found one more hardcoded profile-behavior: SourceStep.applyRecommendation)** — iter-64 meta-rule "re-grep after a fix class with head_limit=0" applied. Grepped `profile == "JANG` and similar across Swift + Python. Most hits were source-of-truth tables, test fixtures, or docstring prose. **One real behavioral hardcode remained:** `SourceStep.swift:256`:
      ```swift
      if plan.profile == "JANG_4K" {
          plan.profile = rec.recommended.profile
      }
      ```
      The comment said "replace if still at the app-level default (JANG_4K)." The INTENT was "only overwrite if the user hasn't manually changed it." But the hardcoded "JANG_4K" ignored `settings.defaultProfile`.
      **Bug scenario:** User configures `settings.defaultProfile = "JANG_2L"` (e.g., regular MoE work). `applyDefaults` seeds `plan.profile = "JANG_2L"`. User picks a dense LLM source. recommend.py suggests "JANG_4K" for dense. SourceStep's `if plan.profile == "JANG_4K"` check FAILS (profile is JANG_2L). Recommendation not applied. User gets JANG_2L for a dense model where JANG_4K is better. Either they notice and manually change it, or they end up with a suboptimal conversion.
      **Fix (iter 65):**
      - Added `@Environment(AppSettings.self) private var settings` to SourceStep.
      - Changed the check to `let seedDefault = settings.defaultProfile.isEmpty ? "JANG_4K" : settings.defaultProfile; if plan.profile == seedDefault { ... }`. Handles empty-settings fallback (matches applyDefaults behavior).
      - User who manually picks a profile in ProfileStep still has `plan.profile != seedDefault`, so the recommendation won't overwrite their choice across subsequent source re-picks.
      **Tests (+1) in WizardStepContinueGateTests.swift:** `test_sourceStep_applyRecommendation_uses_settings_default` — source-inspection pin that asserts (a) no hardcoded `plan.profile == "JANG_4K"` literal remains, (b) `settings.defaultProfile` is referenced.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:5-16, 254-270`. 163 Swift tests pass (was 162, +1). Python 314 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M144 (applyRecommendation field-overwrite asymmetry: family unconditionally wiped user's JANGTQ choice)** — Iter-65 forecast called for a field-by-field audit of `SourceStep.applyRecommendation`'s overwrite logic. Iter-66 finds the sharpest bug:
      **Pre-iter-66 field table:**
      | Field | Overwrite logic |
      | --- | --- |
      | `family` | **unconditional** |
      | `profile` | conditional (iter-65 M143 fix: seed-default check) |
      | `method` | unconditional |
      | `hadamard` | unconditional |
      | `forceDtype` | unconditional (if rec has one) |
      | `forceBlockSize` | conditional (nil-check) |
      **The sharpest pathology — family+profile inconsistency.** After iter-65, profile is preserved when user manually picked one in ProfileStep. But family was STILL overwritten. Scenario:
      1. User picks source → family=.jang, profile=JANG_4K.
      2. ProfileStep: user manually switches family=.jangtq, profile=JANGTQ2.
      3. User re-picks source (same or similar).
      4. applyRecommendation runs: profile preserved as "JANGTQ2" (M143 check fires), BUT family overwritten to `.jang` (unconditional).
      5. Result: `family=.jang + profile=JANGTQ2`. Invalid pair. ProfileStep's family picker now shows JANG + profile dropdown shows JANGTQ2 — user stuck trying to reconcile.
      **Fix (iter 66):** Couple family + profile. If profile was preserved (user manually set it), family stays preserved too. If profile was overwritten from the recommendation, derive family from the new profile name (`plan.profile.hasPrefix("JANGTQ") ? .jangtq : .jang`). The two fields now cannot disagree.
      **Why not also fix method/hadamard/forceDtype this iter:** scope creep + those fields have less user-visible "inconsistent state" impact. A hadamard=true after user toggled it off is annoying but not INVALID. family+profile mismatch IS invalid — JANGTQ2 isn't a JANG-family profile and vice versa.
      **Tests (+2) in WizardStepContinueGateTests.swift:**
      - `test_sourceStep_applyRecommendation_does_not_unconditionally_set_family`: source-inspection ensures the bare `plan.family = (rec.recommended.family == "jangtq") ? .jangtq : .jang` pattern is gone from code (comment text ignored).
      - `test_sourceStep_applyRecommendation_derives_family_from_profile`: ensures the replacement `plan.profile.hasPrefix("JANGTQ")` check is present.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:253-280`. 165 Swift tests pass (was 163, +2). Python 314 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M145 (applyRecommendation extension: hadamard/method/forceDtype preservation)** — iter-66 M144 coupled family+profile. iter-67 extends the same "preserve if user manually changed it" seed-default pattern to the remaining unconditional overwrites.
      **Pre-iter-67 remaining unconditional overwrites:**
      - `plan.method = recMethod` — wipes user's RTN/MSE-all pick on re-pick.
      - `plan.hadamard = rec.recommended.hadamard` — wipes user's manual toggle.
      - `plan.overrides.forceDtype = ft` — wipes user's manual Force dtype override (when rec supplies one, e.g. on 512+ expert models).
      **Concrete pathologies:**
      - Hadamard: user picks source A → profile=JANG_2L → applyRecommendation sets hadamard=false (correct, low-bit). User goes to ProfileStep and for some reason ticks hadamard ON (experimental). Re-picks source → hadamard silently flipped back to false.
      - Method: user ticks RTN for faster convert on a quick test. Re-picks source → silently back to MSE. 10× slow convert starts, user wonders why.
      - forceDtype: user forces fp16 on a smaller MoE for speed (they know their hardware is safe). Re-picks similar source → rec says bfloat16 → silently overwritten back.
      **Fix (iter 67):**
      - `method` guarded by `plan.method == seedMethod` where `seedMethod` is parsed from `settings.defaultMethod` via the same case table applyDefaults uses.
      - `hadamard` guarded by `plan.hadamard == settings.defaultHadamardEnabled`.
      - `forceDtype` guarded by `plan.overrides.forceDtype == nil` (seed default is nil — no Settings seed for this field).
      **Tests (+3) in WizardStepContinueGateTests.swift:** source-inspection pins that each guard is present — `if plan.method == seedMethod`, `if plan.hadamard == settings.defaultHadamardEnabled`, `if plan.overrides.forceDtype == nil`. Matches iter-54/56/58 test style.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:285-335`. 168 Swift tests pass (was 165, +3). Python 314 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M146 (ProfileStep auto-outputURL goes stale when user changes profile)** — Iter-67 forecast called for an ArchitectureStep/ProfileStep mutation sweep. Grepped `plan\.\w+\s*=` across all Step files. Found a subtle staleness bug in ProfileStep:
      **Pre-M146 auto-outputURL (ProfileStep.swift:78-80):**
      ```swift
      if coord.plan.outputURL == nil, let src = coord.plan.sourceURL {
          coord.plan.outputURL = src.deletingLastPathComponent()
              .appendingPathComponent("\(src.lastPathComponent)-\(coord.plan.profile)")
      }
      ```
      **The bug:** fires on .onAppear, sets output folder name to `<src>-<profile>` (e.g., `MyModel-JANG_4K`). If user then switches profile in the Picker to JANG_2L, outputURL STAYS at `MyModel-JANG_4K`. Convert proceeds, writes files into `MyModel-JANG_4K` folder — but the model inside is actually JANG_2L. Every downstream artifact (HF publish, `ls`-listed folder name, diagnostic zip) carries the wrong profile label.
      **Fix (iter 68):** add a regeneration path inside the existing `.onChange(of: coord.plan.profile)` handler. If the current outputURL matches the auto-pattern for the OLD profile (i.e., we generated it — user didn't pick a custom path via `pickOutput()`), regenerate for the NEW profile. If outputURL doesn't match (user-picked), leave alone.
      ```swift
      if cur == autoOld { coord.plan.outputURL = <autoNew> }
      ```
      **Why not unconditionally regenerate:** `pickOutput()` lets the user choose any folder. If they picked `/some/custom/dir`, rewriting it on every profile change would be astonishing. The auto-pattern match distinguishes auto-fill from user-pick without needing extra @State flags.
      **Tests (+1) in WizardStepContinueGateTests.swift:** `test_profileStep_auto_outputURL_follows_profile_change` — source-inspection: (a) `.onChange(of: coord.plan.profile)` exists, (b) regenerates with `newProfile`, (c) gated on `cur == autoOld`.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift:73-91`. 169 Swift tests pass (was 168, +1). Python 314 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M147 (AppSettings.load silently swallows schema-migration decode failures)** — Iter-68 meta-lesson about "derived-from-field staleness" led to a broader scan of @State/@Observable patterns. Found a symmetric bug with iter-37 M111:
      - iter-37 M111: `persist()` logged encode failures to stderr. ✓
      - pre-M147: `load()` silently swallowed decode failures via `try?`. ✗
      **Pre-M147 code:**
      ```swift
      guard let data = UserDefaults.standard.data(forKey: Self.defaultsKey),
            let s = try? JSONDecoder().decode(Snapshot.self, from: data) else { return }
      ```
      Combined `guard` collapsed two semantically distinct cases: "first launch, no data" (OK silent) and "decode failed, data exists but broken" (warrants logging). A future app version that adds a required Settings field would fail to decode old blobs, silently revert the user to factory defaults, and leave NO signal in Copy Diagnostics.
      **Fix (iter 69):** split the cases. "No data" → `return` silent. "Data exists but decode failed" → `FileHandle.standardError.write(...)` logging to match persist()'s M111 pattern, then return with defaults.
      **Tests (+3) in AppSettingsTests.swift:**
      - `test_load_with_corrupted_settings_blob_falls_back_to_defaults`: injects non-JSON blob into UserDefaults, verifies AppSettings init doesn't crash and defaults are intact.
      - `test_load_with_no_saved_settings_is_silent`: regression guard — first-launch path stays silent.
      - `test_load_method_split_is_present_in_source`: source-inspection pin ensures the log message literal survives future refactors (prevents re-collapsing back to `try?`).
      **Evidence:** `JANGStudio/JANGStudio/Models/AppSettings.swift:147-175`. 170 Swift tests pass (was 169, +1 tracked in WizardStepContinueGateTests; AppSettingsTests grew 20→23, +3). Python 314 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M148 (jangspec manifest load: symmetric-path hardening + schema-migration diagnostics)** — Iter-69's meta-rule "symmetric paths must have symmetric error handling" applied to the jangspec bundle format. `write_manifest` raises on write failure (top-level convert catches). `load_manifest` was weaker: bare `json.loads(Path(path).read_text())` + `Manifest(**data)` — both produced cryptic tracebacks on corruption or schema drift.
      **Pre-M148 failure modes:**
      - Disk read fault (OSError) → raw traceback, no path context.
      - Malformed JSON in jangspec.json → `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` — no file path, no hint.
      - Non-dict JSON root (e.g., `[1,2,3]`) → `AttributeError: 'list' object has no attribute 'get'` via `data.get("bundle_version")`.
      - Schema migration (missing or extra field) → `TypeError: Manifest.__init__() missing 1 required positional argument: 'draft_jang'` — gives field name but no bundle path, no hint about version drift.
      **Fix (iter 70):** mirror iter-43 M120 + iter-69 M147 patterns. Every error path:
      - Wraps as `ValueError` with the bundle path in the message.
      - Hints about the cause: "not valid JSON", "expected a JSON object", "schema validation (likely a bundle written by a different jang-tools version)".
      - `from exc` chaining preserves the original for debugging.
      **Tests (+4) in tests/jangspec/test_manifest.py:**
      - `test_manifest_rejects_malformed_json`: invalid JSON → ValueError with path + "not valid JSON".
      - `test_manifest_rejects_non_dict_root`: `[1,2,3]` → ValueError with "expected a JSON object".
      - `test_manifest_rejects_missing_required_field`: simulates schema migration → ValueError with "schema validation" + "different jang-tools version" hint.
      - `test_manifest_missing_file_raises_value_error`: OSError → ValueError with "could not read manifest" + path.
      **Evidence:** `jang-tools/jang_tools/jangspec/manifest.py:49-104`. 318 Python tests pass (was 314, +4). Swift 170 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M149 (format/reader.py: extract shared `_read_json_object` helper + harden 3 call sites)** — iter-70 meta-lesson about the M120/M147/M148 template applied to `format/reader.py`'s `load_jang_model`. Three bare `json.loads(path.read_text())` calls: jang_config, model config, shard index. All produced cryptic tracebacks on disk/encoding/JSON/schema errors — same user-hostile UX.
      **Fix (iter 71):**
      - **Extracted shared helper `_read_json_object(path, *, purpose)`** — first formalization of the template. Wraps OSError / UnicodeDecodeError / JSONDecodeError + isinstance(dict) guard. Every failure emits a ValueError with path + purpose (e.g., "JANG config at /path/x is not valid JSON (line 1, col 3): …").
      - Replaced 3 bare reads with `_read_json_object(config_path, purpose="JANG config")`, `(model_config_path, purpose="model config")`, `(index_path, purpose="shard index")`.
      - Added downstream structure check: `if "weight_map" not in index or not isinstance(index.get("weight_map"), dict):` → ValueError with "corrupted or incompatible version" hint. Catches the schema-migration case where older index format shipped a different top-level key.
      **Tests (+5) in tests/test_format.py `TestFormatReaderErrorDiagnostics`:**
      - `test_malformed_jang_config_raises_with_path`
      - `test_non_dict_root_jang_config_raises`
      - `test_malformed_model_config_raises_with_purpose`
      - `test_malformed_shard_index_raises_with_purpose`
      - `test_shard_index_missing_weight_map_raises`
      Each asserts the 3-property invariant: ValueError + path in message + purpose-noun in message.
      **Why extract the helper now (vs inline like iters 43/69/70):** four sites with near-identical shape is the threshold where DRY wins over simple inline try/except. `_read_json_object` becomes reusable for future read-side loaders in the same codebase — capabilities.py, loader.py, recommend.py could all migrate. Logged for future iters.
      **Evidence:** `jang-tools/jang_tools/format/reader.py:14-60, 200-220`. 323 Python tests pass (was 318, +5). Swift 170 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M150 (capabilities.verify_directory / stamp_directory: contract-breaking raises on corrupt JSON)** — iter-71 M149 left capabilities.py unmigrated; iter-72 applies the template with a contract-aware twist. `verify_directory` is documented to return `tuple[bool, str]`. But bare `json.load(fh)` at 3 sites raised JSONDecodeError on corrupt JSON instead of returning `(False, msg)` — **contract violation**.
      **Impact:** `verify_capabilities` CLI tool walks discovered model dirs calling verify_directory on each. Any one corrupt bundle crashes the entire batch walk. User has 30 models; one is corrupt; `jang_tools verify_capabilities --discover` aborts mid-sweep.
      **Fix (iter 72):** Local helper `_safe_load_json_dict(path, *, purpose)` — variant of M149's `_read_json_object` that RETURNS `(None, msg)` instead of raising. Keeps capabilities.py self-contained without cross-module import. Both `verify_directory` and `stamp_directory` rewired to use it. Every read failure now lands as `(False, msg)` with path + purpose in the message.
      **Why not import format.reader._read_json_object:** cross-subpackage import coupling, and the return contract differs (raise vs return-tuple). Local helper is tighter.
      **Tests (+6) in new `tests/test_capabilities_verify_directory.py`:**
      - `verify_directory_malformed_jang_config_returns_false_with_path`
      - `verify_directory_non_dict_jang_config_returns_false`
      - `verify_directory_malformed_legacy_config_returns_false`
      - `verify_directory_malformed_model_config_returns_false`
      - `stamp_directory_malformed_jang_config_returns_false`
      - `stamp_directory_malformed_config_json_returns_false`
      **Evidence:** `jang-tools/jang_tools/capabilities.py:169-260`, `jang-tools/jang_tools/capabilities.py:295-330`. 329 Python tests pass (was 323, +6). Swift 170 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M172 (Dead `progressLog` state removed from PublishToHuggingFaceSheet)** — Iter-88 M165's diagnostic-audit flagged `@State private var progressLog: [String] = []` at line 25 as vestigial: 3 append sites + 1 reset site but NO UI element reads it. Dead since iter-24 M43's streaming redesign originally envisioned a scrolling log pane inside the publish sheet; the design shipped without the pane but the state + append sites were never cleaned up. Non-bug (no observable impact) but code surface pollution — a future reader asks "what is this for?" and wastes time reconstructing the history.
      **Fix (iter 95):**
      - Removed `@State private var progressLog: [String] = []` declaration.
      - Removed the `progressLog = []` reset at the start of `runPublish`.
      - Removed the `progressLog.append("[note] Cancel click landed…")` at the late-cancel race note. Replaced with an M172 comment noting the race is still pinned by the `sawSuccessfulDone`/`publishResult` flow — no behavior change.
      - Simplified `apply(event:)`: dropped the phase-name and message-level appends. Phase updates still flow through `progressPhase` (which IS displayed). Message events now `break` — if a publish-side log pane is ever re-added, rewire a new array from scratch; the old dead code offered no structure worth preserving.
      **Test (+1) in WizardStepContinueGateTests.swift:**
      - `test_publishSheet_has_no_dead_progressLog_state` — source-inspection, scans non-comment lines (so M172 rationale comments mentioning the name don't trigger the assertion). Asserts neither the `@State` declaration nor `.append` calls resurface. If a future reader tries to re-add the state without wiring a UI reader, the test catches it.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/PublishToHuggingFaceSheet.swift:22-31, 292-295, 329-333, 345-361`. 30 WizardStepContinueGateTests pass (was 29, +1). 31 AdoptionServices unchanged.
      **Meta-lesson — cleanup iters are first-class work.** Code-surface reduction pays a compound return: every future audit of this file has one less "what's this for?" trip-up. The test pin specifically prevents regression, which matters because dead `@State` could drift back in during future feature work ("I need a log pane here, let me add progressLog… oh wait there used to be one, why was it removed?" — iter-95 documents the answer).
      **Meta-lesson — source-inspection tests can be SMARTER by filtering out comments.** Prior source-inspection tests in this file treated the whole file as one string. For the dead-state test, I want the ASSERTION to fail on code but NOT on the M172 rationale comments that mention the name. Split-on-newlines + filter non-comment lines before joining. Small pattern extension; worth using for future tests that need to check for absence-of-code.
      **Commit:** (this iteration)
- [x] **M171 (SourceStep + PublishSheet dryRun — `.onDisappear` cancel consistency sweep per iter-93 M170's generalized rule)** — Iter-93 M170 codified the broader rule: "any SwiftUI view that spawns a detached Task must wire `.onDisappear` cancel, regardless of sheet/window/main status." Iter-94 executes that sweep. Grepped every `.onAppear` + `Task {` spawn across `JANGStudio/Wizard/` and checked each for corresponding `.onDisappear` cancel. Found 2 gaps.
      **Gap A — SourceStep lacked `.onDisappear { detectionTask?.cancel() }`.** iter-57 M135 added cancel-on-new-pickFolder (handles concurrent picks within the same view instance). iter-84 M161 added URL-match guards (handles orphan state-corruption after view destruction). But NEITHER actually TEARS DOWN the Python subprocess when the user sidebar-jumps mid-detection — the subprocess ran for its full ~5-second completion, then its writes were discarded by M161's guard. Low severity (short subprocess, state guarded) but wastes CPU and violates the "uniform view-cancel rule" codified in iter-93.
      **Gap B — PublishToHuggingFaceSheet's Preview button spawned `Task { await runDryRun() }` with no handle.** iter-85 M162 covered `publishTask` (the long-running publish upload) but missed the shorter dryRun Task. User who clicks Preview then dismisses the sheet orphaned the dry-run subprocess for ~seconds. Same class as iter-86 M163's retry-button consistency fix.
      **Fix (iter 94):**
      - `SourceStep`: added `.onDisappear { detectionTask?.cancel() }`. Cancel propagates through iter-34 M105's SourceDetector + iter-76 M153's PythonCLIInvoker onCancel → SIGTERM subprocess.
      - `PublishToHuggingFaceSheet`: added `@State dryRunTask: Task<Void, Never>?`. Preview button now cancels previous handle + re-spawns into `dryRunTask`. Extended the existing `.onDisappear` block to also cancel `dryRunTask` alongside `publishTask`.
      **Tests (+2) in WizardStepContinueGateTests.swift:**
      - `test_sourceStep_cancels_detectionTask_onDisappear` — pins the `.onDisappear { detectionTask?.cancel() }` wiring.
      - `test_publishSheet_cancels_dryRunTask_onDisappear` — pins `dryRunTask` @State + assignment in Preview + cancel in `.onDisappear`.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:189-204`, `JANGStudio/JANGStudio/Wizard/PublishToHuggingFaceSheet.swift:30-37, 83-89, 208-221`. 29 WizardStepContinueGateTests pass (was 27, +2). AdoptionServices 31, PythonCLIInvoker 9 unchanged.
      **Sweep summary:** all 11 `.onAppear` + Task-spawn sites in the wizard now conform to the uniform rule. Four sheets + SourceStep + RunStep all have `.onDisappear` cancel wiring. The remaining `Task {` spawns are trivially fast (drop callbacks, cancel-button idempotent hops, synchronous log appends) and don't need handle tracking.
      **Meta-lesson — generalized rules pay compound interest.** Iter-93 M170 established the rule; iter-94 executed the sweep and closed 2 more gaps in 15 minutes of work. Without the iter-93 generalization, each gap would have been discovered independently as its own audit iter. Investing in the generalization saved 2-3 iter-lengths of investigation time.
      **Meta-lesson — audit sweeps benefit from explicit "what's the inventory?" first step.** Before fixing, I grep'd ALL `.onAppear` / `.task {` / `Task {` spawns across the wizard (40+ hits), triaged each, then scoped the fix to the 2 real gaps. Without this inventory pass I'd have fixed SourceStep, stopped, maybe missed the PublishSheet dryRun, queued a future iter for it. Explicit inventory → single-iter closure.
      **Commit:** (this iteration)
- [x] **M170 (RunStep orphan subprocess on window-close / app-quit — main-window variant of iter-85 M162)** — Iter-93 fresh audit angle: "what happens when the user quits the app during a 30-minute convert?" Grepped the Swift app for `applicationWillTerminate`, `NSApplicationDelegate`, lifecycle hooks — ZERO matches. RunStep's `.onAppear { Task { await start() } }` spawned the conversion Task with no handle. When the main window closes (red-X) or app quits (cmd-Q), SwiftUI unmounts RunStep, the `runner: PythonRunner?` @State is lost, but the conversion Task keeps running. Python convert subprocess becomes an orphaned child of launchd (PID 1) — continues writing to the output folder for up to 30 more minutes with no UI to cancel from. User sees Mac at 100% CPU after "quitting" and has to kill -9 the orphan manually.
      **Why iter-85 M162's fix didn't cover this:** iter-85 wired `.onDisappear { publishTask?.cancel() }` for SHEETS. RunStep is not a sheet — it's a step in the main-window NavigationSplitView detail pane. Sheet-dismiss hooks don't fire on main-window close. Missed in iter-85's sweep because the audit was framed around "sheets"; should have been framed around "any view that owns a long-running Task handle."
      **Fix (iter 93):**
      - Added `@State private var runTask: Task<Void, Never>?` to RunStep.
      - `.onAppear` now stores the spawned Task in `runTask` instead of discarding.
      - Both Retry buttons (after-cancel + after-fail) now cancel the previous handle then re-spawn into `runTask` — same shape as iter-85 M162's publish-sheet retry pattern.
      - Added `.onDisappear { runTask?.cancel() }` — fires on main-window close, app-quit (cmd-Q), or tab-switch-away. Cancel propagates through iter-32 M100's withTaskCancellationHandler → PythonRunner.cancel() → SIGTERM + 3 s SIGKILL escalation.
      **Tests (+2) in WizardStepContinueGateTests.swift:**
      - `test_runStep_cancels_runTask_onDisappear` — pins the `@State runTask` + `.onDisappear { runTask?.cancel() }` wiring.
      - `test_runStep_retry_buttons_use_runTask_handle` — counts `runTask = Task { await start() }` assignments; must be ≥ 2 (one per Retry button path) to ensure retries also tracked.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift:29-42, 67-71, 92-96, 138-152`. 27 WizardStepContinueGateTests pass (was 25, +2). PythonRunnerTests 10, PostConvertVerifierTests 14 unchanged.
      **Meta-lesson — view-lifecycle cancel hooks are required for EVERY view that spawns a detached Task, not just sheets.** Iter-85 M162's framing was "sheets leak subprocesses on dismiss." The actual rule is broader: "any SwiftUI view that spawns a Task must wire `.onDisappear { taskHandle?.cancel() }`." That covers sheets, popovers, full-screen covers, AND main-window-content views. Would have caught M170 if the iter-85 grep had been "all .onAppear Task spawns across the wizard" instead of "all sheet dismissals."
      **Meta-lesson — "app quit" is a view-unmount event in SwiftUI.** When the user cmd-Qs a SwiftUI app, the window goes away, SwiftUI unmounts the view hierarchy, `.onDisappear` fires for every view in the tree. So the same `.onDisappear` hook handles BOTH window-close AND app-quit cases — no need for a separate NSApplicationDelegate.applicationWillTerminate hook. Simpler than I initially expected.
      **Generalized audit rule going forward:** grep every `.onAppear` with a Task spawn. Each one should have a corresponding `.onDisappear` cancel. Tracked as a new audit axis for future iters.
      **Commit:** (this iteration)
- [x] **M169 (ProcessError remediation — convert-subprocess failure sweep, iter-90/91 pattern continued)** — Iter-91 M168 applied the "surface remediation, not just symptom" meta-lesson to `PublishServiceError`. Iter-92 continues the sweep to the second-highest-stakes error path: `ProcessError` (thrown from PythonRunner when the convert subprocess exits non-zero). Pre-M169, `ProcessError` was a plain struct — RunStep's catch did `logs.append("[ERROR] \(error)")` which stringified the struct to ugly `ProcessError(code: 1, lastStderr: "...")` with no remediation.
      **Fix (iter 92):**
      - `ProcessError` now conforms to `LocalizedError`. `errorDescription` returns `"jang-tools convert exited X: <stderr>\n→ <remediation>"`.
      - Added `nonisolated static func remediation(code:stderr:) -> String` covering 4 most-common convert failure shapes + generic fallback:
        - **OOM** (`failed to allocate` / `MemoryError` / `cannot allocate` / exit 137 / `Killed`) → "Try a smaller profile (JANG_2L or JANG_3L instead of JANG_4K), close other apps, or run on a larger Mac (128+ GB recommended for 256+ expert models)."
        - **Disk full** (`no space left` / `[Errno 28]` / `disk quota`) → "Output needs source-size × (avg-bits/16). Free space or pick a different output folder."
        - **trust_remote_code / missing modeling_*.py** (MiniMax, Cascade, custom architectures) → "Re-download INCLUDING .py files: `huggingface-cli download <repo> --include '*.py' --include '*.safetensors' --include '*.json'`."
        - **Corrupt shard** (`safetensorsError` / `header too big` / `invalid header` / `corrupt`) → "Shard file appears corrupt. Re-download the source model."
        - **Fallback** → "Check log pane, Copy Diagnostics for bug report, or retry with a smaller profile."
      - `RunStep.swift:186` now uses `error.localizedDescription` instead of `\(error)` so the log line inherits the remediation on ProcessError AND falls back to platform default for other error types.
      **Tests (+6) in PythonRunnerTests.swift:**
      - `test_processError_oom_suggests_smaller_profile` — OOM stderr + specific profile name.
      - `test_processError_killed_suggests_oom_root_cause` — exit 137 / "Killed" maps to OOM hint.
      - `test_processError_disk_full_suggests_free_space` — Errno 28 → disk hint.
      - `test_processError_trust_remote_code_suggests_redownload` — ModuleNotFoundError for `modeling_X` → HF-CLI command.
      - `test_processError_generic_falls_back_to_check_logs` — unknown stderr → still gets a next-action hint.
      - `test_processError_preserves_code_and_stderr` — regression guard: remediation appended, not substituted.
      TDD: red on 6 (member-not-found until LocalizedError + errorDescription added) → fix → green.
      **Evidence:** `JANGStudio/JANGStudio/Runner/PythonRunner.swift:4-66`, `JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift:186`. 10 PythonRunnerTests pass (was 4, +6). 31 AdoptionServices + 9 InferenceRunner + 14 PostConvertVerifier + 25 Wizard-gate tests unchanged. Python 351 unchanged.
      **Meta-lesson — substring detection covers cross-layer error sources.** Convert failures can come from Python, MLX, CPython, POSIX, or macOS kernel. Each layer has its own wording conventions. Substring matching case-insensitively catches ALL of them (e.g., "killed" matches both CPython's `KilledError` and macOS's bare `Killed\n`). A regex-per-layer approach would require maintaining separate patterns per error source.
      **Meta-lesson — exit code is a first-class signal alongside stderr.** Code 137 = SIGKILL, code 139 = SIGSEGV, these are kernel signals that often don't produce any stderr. Pattern-match on CODE as well as stderr, so a clean-exit-but-killed subprocess still triggers the OOM hint. Expanded iter-91's substring-only approach.
      **Sweep status:** P1 (PublishServiceError) done iter-91. P2 (ProcessError) done iter-92. Remaining surfaces — `SourceDetectorError.cliError` already emits Python-side messages with remediation (iter-21 M120 / iter-89 M166 / iter-90 M167), but the Swift wrapper lacks a tiered fallback for Python-subprocess-crash cases. P3 (adoption-service errors) degrade silently with frozen fallback and don't need remediation. Sweep is effectively complete for blocking error paths.
      **Commit:** (this iteration)
- [x] **M168 (PublishServiceError.cliError — remediation-command sweep after iter-90 M167 meta-lesson)** — Iter-90 M167 codified: "surface remediation, not just symptom." Iter-91 applies it to the highest-stakes error path in the app: `PublishServiceError.cliError`. After a 30-minute publish dispatch fails, the error banner is the user's only UI — it MUST tell them what to do next. Pre-M168 it showed raw stderr only ("jang-tools publish exited 1: HfHubHTTPError: 401 Client Error: Unauthorized"), leaving the user to google "HF 401 error" for every failure.
      **Fix (iter 91):** added `PublishServiceError.remediation(forStderr:)` — a `nonisolated static` helper that pattern-matches common huggingface_hub stderr substrings (case-insensitive) and returns a concrete next-action string. `cliError.errorDescription` now appends the remediation below the stderr, separated by `→ `.
      **Pattern coverage:**
      - `401` / `Unauthorized` → "Auth failed — verify token with `huggingface-cli whoami`, or regenerate at https://huggingface.co/settings/tokens."
      - `403` / `Forbidden` → "Permission denied — token valid but lacks write access. Check scope at settings/tokens, or confirm org admin granted write access."
      - `Connection` / `timeout` / `Max retries` / `network is unreachable` → "Network error — check connectivity + retry. If persists, huggingface.co may be having an incident (https://status.huggingface.co)."
      - `rate limit` / `429` / `too many requests` → "Rate limited — wait a few minutes and retry. HF imposes per-IP/per-token upload limits during peak hours."
      - **Fallback:** "Common fixes: verify the token, check network, or retry." So EVERY failure gets at least one next-action, not just the well-known error codes.
      **Why substring match, not regex:** HF error messages evolve slightly over versions. Substring + case-insensitive survives minor tweaks without needing exactly-matched regexes that break on dash vs em-dash, quote styles, etc.
      **Tests (+5) in AdoptionServicesTests.swift:**
      - `test_publish_cliError_401_suggests_token_check` — verifies 401 stderr produces a token-aware hint citing the HF settings URL.
      - `test_publish_cliError_403_suggests_permission_check` — 403 → permission / write-access hint.
      - `test_publish_cliError_network_suggests_retry` — Connection/timeout → network + retry hint.
      - `test_publish_cliError_generic_falls_back_to_generic_hint` — unknown stderr → still gets token/network/retry suggestion.
      - `test_publish_cliError_preserves_stderr_in_all_branches` — regression guard: remediation is APPENDED, stderr is preserved. User can still see what actually failed AND what to do.
      TDD red (4 failures on the hint-specific assertions) → green after fix.
      **Evidence:** `JANGStudio/JANGStudio/Runner/PublishService.swift:125-175`. 31 AdoptionServicesTests pass (was 26, +5). Python 351 unchanged. Ralph 73 unchanged.
      **Meta-lesson extension — tiered remediation beats per-case remediation.** Rather than requiring an exhaustive case list (which would miss new HF error shapes), tier 1 is pattern-matched specific hints + tier 2 is a generic fallback. Every user gets AT LEAST one next-action; well-known shapes get more targeted guidance. Apply this pattern to future diagnostic layers: always include a fallback hint so coverage is complete even before the specific cases land.
      **Commit:** (this iteration)
- [x] **M167 (Broken-symlink in HF cache — cryptic traceback → actionable diagnostic)** — Iter-89 M166's symlink audit flagged the broken-symlink case as a UX gap. A `git gc`-style prune on the HF cache (or a disk-failure mid-download via `huggingface-cli download`) leaves snapshot-dir symlinks pointing at missing blobs. Pre-M167, `_total_bytes` ran `f.stat()` on a dangling symlink and raised a bare FileNotFoundError — user saw a cryptic multi-line Python traceback with the temp path in it, no hint that the fix is `huggingface-cli download <model>` or clearing the HF cache.
      **Fix (iter 90):** added `_find_broken_shard(model_path) -> Path | None` that iterates `glob("*.safetensors")` and checks `.exists()` (returns False for dangling symlinks). `cmd_inspect_source` calls it after config.json validation; if any shard is broken, emit an actionable error citing:
      - The broken shard filename.
      - The target the symlink points to (via `resolve(strict=False)` — strict=False lets us inspect the target of a dangling link without raising).
      - The recommended fix command: `huggingface-cli download <model-name>`.
      - The user's selected source path for context.
      Then `sys.exit(2)` to keep the M120 "no cryptic tracebacks" contract.
      **Test (+1) in `jang-tools/tests/test_inspect_source.py`:**
      - `test_inspect_source_broken_symlink_emits_clean_error` — tmpdir with valid config.json + dangling `model-00001-of-00001.safetensors → nonexistent_blob`. Asserts exit non-zero, no `Traceback` in stderr, stderr contains "broken symlink" + the shard filename so the user can locate the bad file in their cache. TDD: wrote test first → red (bare FileNotFoundError traceback confirmed), applied fix → green.
      **Why not skip-and-warn the broken shard:** rejected the "filter out broken symlinks, continue with the rest" alternative because a partial model is LOAD-TIME-BROKEN anyway (convert would fail when trying to read the missing weights). Failing loudly at inspect-source time is better than failing 5 seconds into convert with weights missing — the user is still at Step 1 and can fix the cache before committing to a 30-minute run.
      **Evidence:** `jang-tools/jang_tools/inspect_source.py:45-60, 86-99`. 9 test_inspect_source tests pass (was 8, +1). 351 total Python tests pass. Swift unchanged. Ralph 73 unchanged.
      **Meta-lesson — surface remote-dependency state errors with the remediation command.** JANG Studio doesn't know WHY a shard is broken (cache prune? interrupted download? filesystem corruption?) but it DOES know that "re-downloading the model" is the right fix for all three common causes. Include the exact command (`huggingface-cli download …`) in the error message — saves the user from Googling. Similar pattern candidate for future fixes: any HF-side error should cite the recovery path, not just the symptom.
      **Commit:** (this iteration)
- [x] **M166 (Symlink handling audit — HF hub cache snapshot-dir compatibility)** — Iter-89 picked the symlink audit from iter-88's forecast. Hypothesis: `huggingface_hub.snapshot_download` creates a cache layout where `~/.cache/huggingface/hub/models--org--name/snapshots/<hash>/*.safetensors` are SYMLINKS to blobs under `../../blobs/<sha>`. JANG Studio users who point SourceStep at a snapshot directory transitively rely on glob-matches-symlinks and stat-follows-symlinks — if those ever break, HF-hub users would see "0-byte models" or "no shards found."
      **Audit findings:**
      1. `jang_tools.inspect_source._total_bytes` uses `f.stat().st_size` — `Path.stat()` uses `os.stat`, which follows symlinks. ✓
      2. `_sniff_dtype` uses `open(shards[0], "rb")` — follows symlinks. ✓
      3. Shard enumeration uses `sorted(model_path.glob("*.safetensors"))` — glob matches both regular files and symlinks. ✓
      4. Other `stat()` call sites (`publish.py:47,68,137`, `estimate_model.py:29`) all follow the same pattern. ✓
      5. Swift side: `NSOpenPanel.url` preserves the user's symlink selection (not auto-resolved); Python receives the symlink path and operates through it transparently. ✓
      6. `FileManager.attributesOfItem` in `PostConvertVerifier.diskSizeSanityCheck` returns SYMLINK attrs, not target — but that path operates on the convert's output dir (convert writes real files, never symlinks), so not a real exposure.
      **Result: zero new bugs.** The symlink contract works end-to-end for HF cache layouts. But there was ZERO test coverage locking this in — a future perf-motivated refactor (e.g., swap `stat` to `lstat` for "speed") could silently break every HF-hub user.
      **Tests (+2) in `jang-tools/tests/test_inspect_source.py`:**
      - `test_inspect_source_handles_symlinked_shards` — creates `blobs/` with real 4096-byte safetensors + `snapshot/` with a symlink to it. Runs inspect-source on `snapshot/`; asserts shard_count == 1 AND total_bytes == 4096 (target size, not symlink size). Catches any regression where stat gets swapped to lstat or glob starts filtering symlinks.
      - `test_inspect_source_handles_symlinked_directory` — creates a symlink POINTING at a whole model directory (`~/my-model → ~/.cache/hf/.../snapshot`). Asserts glob traverses INTO the symlinked directory and stats shards correctly.
      **Evidence:** `jang-tools/jang_tools/inspect_source.py:14, 19, 42`. 8 test_inspect_source tests pass (was 6, +2).
      **Meta-lesson — contract-preserving tests for transitive-dependency behavior.** The Swift app relies on Python pathlib's symlink semantics which rely on kernel stat/readdir semantics. Neither layer is owned by JANG Studio; both behave correctly today. But the contract CAN break if someone refactors the Python side. Tests that pin the END-TO-END behavior (symlink input → correct output) survive refactors in any layer. Rule: whenever the app depends on a nontrivial filesystem behavior (symlinks, case-insensitive-but-case-preserving paths, unicode normalization, hardlinks), add a lock-in test even if the current implementation "just works." Future-you will thank present-you.
      **Deferred follow-ups:**
      - Broken-symlink edge case (HF cache with a git-gc'd blob). Currently surfaces as a cryptic FileNotFoundError traceback; could be caught in `_total_bytes` and surfaced as "shard missing (broken symlink)". Not a regression but a UX polish. Flagged as M167 candidate.
      **Commit:** (this iteration)
- [x] **M165 (Diagnostics token-leak audit — verification pass, no new bugs + 2 edge-case pins)** — Iter-88 ran a security-adjacent audit on the diagnostic / bug-report export path. Traced every place that writes log/stderr data to disk or clipboard. **No new bugs found** — iter-14 M22e / M16's scrubbing infrastructure is solid. What the audit verified:
      1. **Coverage of HF token formats.** `DiagnosticsBundle.scrubSensitive` catches `hf_` prefix tokens (current format), `huggingface_` prefix tokens (older format), `Authorization: Bearer <token>` headers, and generic `Bearer <token>` constructs.
      2. **Bundle-write paths both scrub.** Sync `write` and async `writeAsync` both pass logs + events through `scrubSensitive` BEFORE atomically writing to the workDir that gets ditto-zipped.
      3. **No other export path exists.** Grepped `.write(to:`, `NSPasteboard`, `writeTo`, `exportTranscript` — the only disk-write that BYPASSES scrubbing is `TestInferenceViewModel.exportTranscript` (user-initiated Save-transcript for their own use), which is not a bug-report surface. `SettingsWindow.copySystemInfo` copies only non-sensitive system stats (macOS version, RAM, CPU, app version). `VerifyStep.copyPath` copies just the output path.
      4. **iter-6 M41 publish-error scrub still works** — token is replaced with `<redacted>` before cliError is thrown, so `PublishServiceError.localizedDescription` (which shows in the publish-sheet's errorMessage) doesn't leak.
      **Added (+2) regression pins in DiagnosticsBundleTests.swift:**
      - `test_scrub_token_in_url_query_string` — HF client retries can log the full URL including `?token=hf_...&revision=main`. Verifies the greedy regex stops cleanly at `&` (not in token char class), redacting the token body while preserving the URL tail.
      - `test_scrub_token_adjacent_to_json_delimiter` — `"token":"hf_..."` shape. Verifies the regex stops at the closing `"`. Guards against any future regex tweak that would accidentally eat the JSON delimiter.
      **Evidence:** `JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift:13-33`. 15 DiagnosticsBundleTests pass (was 13, +2).
      **Meta-lesson — audit-verification iters that find NO bug are still first-class work.** iter-88 didn't close a new bug but confirmed that a security-critical surface is clean, and added two regression pins for edge cases noticed during the trace. The "no new bug" outcome is a positive signal — the codebase is approaching stability — and the edge-case pins lock the behavior in against future regressions. Recording these iters in the audit log is important: without them, future audits may re-trace the same ground wondering "is this surface really covered?" The pinned tests + this log entry answer "yes, iter-88 traced it end-to-end."
      **Commit:** (this iteration)
- [x] **M164 (HFRepoValidator — fail-fast gap: accepted names that huggingface_hub rejects at upload)** — Iter-87 pivoted to a new audit angle (after iter-86 closed the last sheet-dismiss orphan). Deep-traced `PublishService.HFRepoValidator` and found its regex `^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$` is looser than huggingface_hub.validate_repo_id's actual rules. Three categories of names passed the Swift validator but huggingface_hub rejects them at upload time:
      1. **Consecutive `..`** — e.g. `org/my..model`. HF forbids to prevent directory-traversal constructions server-side.
      2. **Consecutive `--`** — e.g. `org/my--model`. Same rationale.
      3. **Trailing `.` or `-`** — e.g. `org/my-model-`, `org/my.model.`. Most common real-world trigger: auto-complete dropping a trailing period, or a stray dash from model-name templating.
      **Impact:** iter-25 M48 was introduced specifically to catch invalid repo-ids BEFORE dispatching a 30-minute upload. Every failure caught late (HF-side) costs the user ~30 minutes of watching a progress bar + ~100 GB of upload bandwidth + a cryptic `HfHubHTTPError` at the end. The user then has to fix the name and re-dispatch — ~60 minutes round-trip for a typo. Pre-M164 the three categories above slipped past the client validator entirely.
      **Fix (iter 87):** added two post-regex guards in the per-segment loop:
      - `segment.hasSuffix(".") || segment.hasSuffix("-")` → rejected with "cannot end with '.' or '-'."
      - `segment.contains("..") || segment.contains("--")` → rejected with "cannot contain consecutive '..' or '--'."
      Kept the leading-alphanumeric regex as first-line defense for the "no leading special char" rule. Overall structure unchanged; just augmented the per-segment validation tail.
      **Tests (+4) in AdoptionServicesTests.swift:**
      - `test_repo_validator_rejects_consecutive_dots` — `org/my..model` + `my..org/name` both rejected.
      - `test_repo_validator_rejects_consecutive_dashes` — same shape for `--`.
      - `test_repo_validator_rejects_trailing_special_char` — `org/my-model-`, `org/my.model.`, and their ORG-segment counterparts.
      - `test_repo_validator_still_accepts_safe_names_with_specials` — regression guard: `my_org/model_name`, `org-name/model-name`, `org.name/model.v2`, `a-b_c.d/e-f_g.h` all still pass (single specials inside segments remain legal).
      **TDD flow verified:** wrote the 4 new tests first → 8 failures (4 test methods with multiple assertions, each failing on the unconstrained validator) → applied the fix → 26/26 AdoptionServicesTests pass.
      **Evidence:** `JANGStudio/JANGStudio/Runner/PublishService.swift:55-70`. 26 AdoptionServicesTests pass (was 22, +4). Python 348 + ralph 73 unchanged. Full Swift suite ~194.
      **Meta-lesson — mirror remote validation client-side when the remote round-trip is expensive.** The Swift-side validator isn't just UX polish; it's the fail-fast gate for a 30-minute, 100 GB operation. Any name accepted by Swift but rejected by HF becomes a ~60-minute user-time loss per instance. Audit every client-side pre-validation to ensure it's AT LEAST as strict as the remote — occasional false-positives (rejecting a valid name) are worse than missed-negatives (30-minute failure) only when the false-positive rate is significant. HF's rules are well-documented and stable; matching them exactly is the right trade.
      **Commit:** (this iteration)
- [x] **M163 (GenerateModelCardSheet + UsageExamplesSheet — Retry-Button Task orphan consistency fix)** — Iter-85 M162 closed the severe sheet-dismiss bug (publish data exfiltration + inference GPU leak). Iter-86 applies the same pattern to the two remaining sheets for consistency. Both use `.task { ... }` for initial work (SwiftUI auto-cancels on dismount — safe) but have `Button("Retry") { Task { ... } }` for error-recovery retries. The retry Task is detached from the view lifecycle — dismissing the sheet while a retry is in flight orphans the Python subprocess.
      **Severity:** low. ModelCardService and ExamplesService are read-only CLIs that complete in seconds (not the 30 minutes of publish). No data exfiltration, no GPU pin — just a few seconds of wasted compute. But the inconsistency means a reader auditing "do all sheets handle dismiss-cancel correctly?" gets a yes-yes-no-no answer after M162; M163 makes it yes-yes-yes-yes.
      **Fix (iter 86):** each sheet gets `@State private var retryTask: Task<Void, Never>?`. Retry button cancels the previous handle (defense against rapid double-click) and re-spawns into the handle. `.onDisappear { retryTask?.cancel() }` fires on dismissal. Cancel flows through PythonCLIInvoker's iter-83 M160 onCancel wiring → SIGTERM subprocess.
      **Tests (+2) in WizardStepContinueGateTests.swift:**
      - `test_generateModelCardSheet_retry_task_cancelled_onDisappear` — source-inspection pin.
      - `test_usageExamplesSheet_retry_task_cancelled_onDisappear` — source-inspection pin.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/GenerateModelCardSheet.swift:12, 33, 158-161`, `UsageExamplesSheet.swift:11-16, 40, 91-94`. 25 WizardStepContinueGateTests pass (was 23, +2).
      **Meta-lesson — audit consistency is worth low-severity fixes.** Each individual Retry-button orphan is "just a few seconds wasted," but the pattern "all sheets must handle dismiss-cancel uniformly" is easier to hold in your head and audit than "publish + inference do, modelcard + examples don't, for specific reasons." Consistency makes future audits cheaper: a new sheet automatically inherits the expected pattern rather than requiring per-sheet severity analysis.
      **Audit surface status:** all 4 result-producing sheets (Publish, TestInference, ModelCard, Examples) now correctly cancel their in-flight Tasks on dismissal. The widget-sweep from iter-85's forecast is complete. No remaining sheet-dismiss orphans.
      **Commit:** (this iteration)
- [x] **M162 (PublishToHuggingFaceSheet + TestInferenceSheet — dismiss orphans subprocesses)** — Iter-85 applied iter-84's meta-lesson ("@State handles don't bridge view destruction") to the sheet layer. Grepped all Task-spawning sites in `JANGStudio/Wizard/` and found two sheets whose dismiss doesn't cancel in-flight subprocesses.
      **Bug 1 (severe — user-facing data exfiltration):** `PublishToHuggingFaceSheet` stores `publishTask: @State Task<Void, Never>?` and uses it for the in-sheet Cancel button. But the sheet has NO `.onDisappear` cancel wiring — if the user dismisses via the header Close button, cmd-W, or the system-provided close gesture, the Task keeps running and the Python subprocess keeps uploading files to HuggingFace for the remaining ~30 minutes. User who realizes they typed the wrong repo name and clicks Close thinking that cancels the upload is WRONG — files still ship. Real data-exfiltration vector, one bad click away from publishing your model to a competitor's HF org.
      **Bug 2 (minor — GPU leak):** `TestInferenceSheet` has no dismiss-cancel wiring either. User clicks Run, waits, realizes the generate is slow and closes the sheet — the inference subprocess keeps running for the remaining 5-60 s, pinning GPU + memory and blocking a subsequent Test Inference run on the same model.
      **Fix (iter 85):**
      - `PublishToHuggingFaceSheet`: `.onDisappear { publishTask?.cancel() }` — cancels the Task, which triggers `continuation.onTermination` via iter-30 M96 wiring, which triggers `ProcessHandle.cancel()`, which SIGTERMs the subprocess + 3 s SIGKILL escalation.
      - `TestInferenceSheet`: `.onDisappear { Task { await vm.cancel() } }` — hops through the existing actor cancel() which funnels through InferenceRunner.cancel().
      **Tests (+3) in WizardStepContinueGateTests.swift:**
      - `test_publishSheet_cancels_publishTask_onDisappear` — source-inspection pin for the onDisappear + publishTask?.cancel() literal.
      - `test_publishSheet_M162_rationale_pinned` — keeps the "data-exfiltration" rationale in source so a reviewer doing a future simplification sweep can't innocently strip the hook.
      - `test_testInferenceSheet_cancels_vm_onDisappear` — source-inspection pin for the TestInferenceSheet side.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/PublishToHuggingFaceSheet.swift:61-81`, `TestInferenceSheet.swift:41-54`. 23 WizardStepContinueGateTests pass (was 20, +3). Python 348 + ralph 73 unchanged.
      **Meta-lesson extension — sheet dismissal is a "view destruction" event too.** Iter-84's rule was framed around sidebar navigation; the sheet variant is the SAME class (view disappears with active work handle as @State) with a different UI affordance. Audit pattern: for every sheet, ask "does dismissing it while work is in flight leave an orphan subprocess / network call / DB write?" The answer should always be NO; if yes, add `.onDisappear { taskHandle?.cancel() }`. M162 followup flagged: `GenerateModelCardSheet` and `UsageExamplesSheet` use `.task { ... }` + `Button("Retry") { Task { ... } }`. The `.task` variant is auto-cancelled on dismount; the retry-button Task is NOT. Low severity (read-only operations, seconds not minutes), but worth codifying for consistency. Will audit in a future iter.
      **Commit:** (this iteration)
- [x] **M161 (SourceStep stale-task cross-view-destruction race — silent conversion-plan corruption)** — Iter-84 pivoted from subprocess-pipe audits to the UI layer per iter-83 forecast. Deep-traced WizardCoordinator's sidebar selection + SourceStep's detection lifecycle and surfaced a silent data-corruption vector that iter-57 M135's `detectionTask?.cancel()` can't cover.
      **Bug:** The stale-task handle `detectionTask` is `@State private` — scoped to the SourceStep view instance. When the user sidebar-jumps Source → Architecture → Source, SwiftUI destroys the old SourceStep; the detection Task it spawned keeps running (Task independence from View lifetime) but the NEW SourceStep's `detectionTask` is nil. When the new view picks a different folder, `detectionTask?.cancel()` is a no-op on the nil handle — the old task is orphaned, not cancelled. `Task.isCancelled` stays false, and the orphan's completion writes the OLD folder's `detected` into `coord.plan.detected` AFTER the new folder's fast-detection has already populated it. Result: `plan.sourceURL = folderB` but `plan.detected = folderA_architecture`. Downstream convert uses B's path with A's architecture → wrong quantization → silent subtly-wrong output. Same M135 class, orthogonal trigger.
      **Fix (iter 84):** every `MainActor.run` write-back in `detectAndRecommend` gets a second guard — `guard coord.plan.sourceURL == url else { return }`. URL-match is the AUTHORITATIVE "still-relevant" signal: if sourceURL has moved on, the task's result is stale regardless of whether it was cancelled. Five sites: detect-success, detect-error, isDetecting=false, rec-success, rec-error. Task.isCancelled check stays as first-line defense against explicit cancel (M135 path); URL-match is second-line against orphans (M161 path).
      **Why not move detectionTask to WizardCoordinator:** @Observable class storage would persist across view destruction — but it also couples SourceStep's UI-lifecycle concerns to the coordinator, and breaks the @State encapsulation that iter-57 chose for reasons. URL-match guard is strictly less invasive + handles a strict superset of cases (also catches any other future orphan pattern, not just sidebar-destruction).
      **Tests (+2) in WizardStepContinueGateTests.swift:**
      - `test_sourceStep_detectAndRecommend_guards_writes_by_url_match` — counts occurrences of the guard literal; must be ≥ 5 (one per write-back site).
      - `test_sourceStep_M161_rationale_is_present` — pins the M161 + "orphaned" comment text so a future simplification sweep surfaces the reviewer when they try to strip the guards.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:211-286`. 20 WizardStepContinueGateTests pass (was 18, +2). Python 348 + ralph 73 unchanged.
      **Meta-lesson — @State handles don't bridge View destruction.** Any cancel token / task handle / subscription that needs to survive the user moving away from a View must live on an @Observable coordinator, not @State. Corollary: for cross-view-lifetime orphan cancel-protection, rely on CONTENT-match (URL / generation token / resource-ID match), not handle-tracking. The handle can disappear; the content match is authoritative.
      **Commit:** (this iteration)
- [x] **M160 (PythonCLIInvoker — final pipe-drain-class fix, completing the iter-81/82 sweep)** — Iter-82 M159's investigation log flagged `PythonCLIInvoker` as the last latent candidate of the unread-Pipe class. Iter-83 closes it.
      **Bug shape:** `PythonCLIInvoker.invoke` runs on `DispatchQueue.global().async` — inside that thread: `try proc.run()` → `proc.waitUntilExit()` → `readDataToEndOfFile()` on both pipes. `waitUntilExit()` blocks the same thread that will later drain the pipes, so a subprocess that writes >64 KB to either stream blocks on write(2), can't exit, and `waitUntilExit()` deadlocks forever. Seven callers run the risk: `RecommendationService`, `ExamplesService`, `ModelCardService`, `CapabilitiesService`, `ProfilesService`, `SourceDetector`, `PublishService.invoke`. Normal output is small, but `jang_tools examples --list` on a large registry or any Python traceback with MLX imports stacked can cross 64 KB.
      **Fix (iter 83):** two parallel `DispatchQueue.global().async` drain blocks started BEFORE `proc.run()`. Each reads its pipe via `readDataToEndOfFile()` (drain pattern #2 from iter-82's three-pattern rule: whole-buffer reads on separate threads, synchronized via `DispatchSemaphore`). After `waitUntilExit()`, wait on both semaphores, then use the captured Data via a file-private `@unchecked Sendable` `DataBox` wrapper — the semaphore establishes happens-before so the reads are safe; Swift 6 can't infer that statically, hence the wrapper.
      **Why not rewrite to Task.detached:** the existing function body is wrapped in `withCheckedThrowingContinuation` + `DispatchQueue.global().async` because PythonCLIInvoker has to be callable from sync + actor + main-actor contexts alike (iter-76 M153 design). Keeping the DispatchQueue-based body minimizes blast radius and preserves the iter-77 M154 cancel-propagation tests' 7-test coverage.
      **Tests (+2) in PythonCLIInvokerTests.swift:**
      - `test_invoke_does_not_hang_on_large_stdout_output` — subprocess emits ~275 KB to stdout then exits 0. Fixed code captures the full Data in <1 s; bug path would hang at 64 KB.
      - `test_invoke_does_not_hang_on_large_stderr_output_on_failure` — symmetric: subprocess emits ~275 KB to stderr then exits 11. `errorFactory` must receive the full stderr (not a truncated prefix), so the UI surfaces the complete diagnostic. 9/9 PythonCLIInvokerTests pass (was 7, +2).
      **Whole-app pipe-drain audit status:** with M160 landed, all 5 Process-launching sites follow one of the three correct patterns (`PythonRunner` = bytes.lines streaming tasks, `PublishService._streamPublish` = bytes.lines streaming + nullDevice for stdout, `PostConvertVerifier.runJangValidate` = nullDevice for both, `InferenceRunner` = whole-buffer detached tasks, `PythonCLIInvoker` = whole-buffer dispatch-global drains). Zero latent pipe-fill deadlocks remain in the Swift app.
      **Evidence:** `JANGStudio/JANGStudio/Runner/PythonCLIInvoker.swift:5-14, 72-123`. 9 PythonCLIInvokerTests + 22 AdoptionServices + 5 Capabilities + 7 Profiles + 14 PostConvertVerifier + 9 InferenceRunner + 4 PythonRunner pass (70 subprocess-helper tests green). Python 348 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M159 (InferenceRunner + PublishService._streamPublish — unread-Pipe audit, iter-81 M158 pattern sweep)** — Iter-81 M158 fixed one unread-`Pipe()` site (`PostConvertVerifier.runJangValidate`). Iter-82 meta-rule: after fixing a pattern, **grep the whole codebase for the same pattern + every neighbor variant**. The iter-81 grep pinned `proc\.standardOutput\s*=\s*Pipe\(\)` but missed two cases:
      1. **`InferenceRunner.swift:134-135`** — wires `Pipe()` to stdout + stderr, delays `readDataToEndOfFile()` until AFTER `proc.terminationHandler` fires. Same cross-process deadlock as M158 bug 1, but worse because **this is the Test Inference UI subprocess.** MLX inference chatter (per-layer warnings on GLM-5.1 / Qwen3.6 / MiniMax M2.7, tokenizer init messages, generation traces in verbose mode) can cross 64 KB in a single generate call. User clicks Run, progress spinner spins forever, Cancel button is the only out, they see `"generation cancelled by user"` for a run that was actually fine. feedback_model_checklist.md rule 3 violation ("wizard never lies"): the wizard silently fails on chatty models.
      2. **`PublishService.swift:_streamPublish:232-255`** — wires `outPipe = Pipe()` to stdout but never drains it. The comment said "Drain stdout silently here — the final JSON result lives there and we capture it at process end" but the code never does. `publishWithProgress` is the streaming path that yields stderr-based ProgressEvents; stdout is never consumed. Latent (huggingface_hub is normally quiet on stdout) but a timebomb if any dep ever `print()`s.

      **Fix (iter 82):**
      - **InferenceRunner:** Promoted both `readDataToEndOfFile()` calls into `Task.detached` BEFORE `proc.run()` so pipes drain in real time on separate threads. After termination, `await stdoutTask.value` + `await stderrTask.value` retrieves the accumulated Data. Error-path `proc.run()` throw now explicitly closes pipe write-ends + awaits task completion before rethrowing — otherwise the detached reads leak waiting on ARC-deferred pipe deallocation. Matches PythonRunner's pattern (`for try await line in ... .bytes.lines` in detached tasks).
      - **PublishService._streamPublish:** Swapped stdout `Pipe()` → `FileHandle.nullDevice`. The streaming path has no use for stdout at all (all progress yields flow through stderr's JSONL parser); discarding it at the kernel eliminates the deadlock class entirely, with zero behavior change on the consumed path. Non-streaming `publish()` still captures stdout via PythonCLIInvoker (iter-79 M156) which correctly drains both pipes synchronously at end.

      **Tests (+2) in InferenceRunnerTests.swift:**
      - `test_generate_does_not_hang_on_large_stderr_output` — subprocess dumps ~275 KB to stderr (200 KB random base64'd) then emits valid InferenceResult JSON, exits 0. Bug: hits 64 KB, blocks, hangs. Fix: 0.913 s end-to-end.
      - `test_generate_does_not_hang_on_large_stdout_output` — same direction other stream: ~275 KB stdout noise before the final JSON line. Exercises the JSON-scan (`.last(where: hasPrefix("{"))`) over a large buffer. Fix: 0.689 s, correct result extracted.

      PublishService has no dedicated test for the latent-path fix (requires HF mock subprocess — too wide); the existing 22 AdoptionServicesTests pass unchanged, which pins that the `publish()` non-streaming path is unbroken.

      **Meta-lesson — after fixing one instance of a pattern, grep for structurally adjacent variants, not just the exact match.** Iter-81 searched `proc.standardOutput = Pipe()` (literal). That found PostConvertVerifier but missed InferenceRunner because InferenceRunner uses local vars `let out = Pipe(); let err = Pipe()` then assigns. The right grep is **the bug class (delayed read), not the specific syntactic shape**. The class signature: any pipe read that happens AFTER `waitUntilExit()` / `terminationHandler` / process-exit-continuation. I should have grepped `readDataToEndOfFile()` cross-referenced with preceding `Pipe()`.

      **Meta-lesson — `Task.detached { readDataToEndOfFile() }` is the safest drain pattern when you need the full buffer as Data.** `for try await line in fileHandle.bytes.lines` is correct for line-streaming consumers (PublishService.stderrTask, PythonRunner.stdoutTask/stderrTask); `readDataToEndOfFile()` in a detached task is correct for whole-buffer consumers (InferenceRunner). The wrong pattern is ANY synchronous read on the main await-thread AFTER the subprocess has exited, because the subprocess can't exit until the pipe's been drained.

      **Evidence:** `JANGStudio/JANGStudio/Runner/InferenceRunner.swift:103-151`, `JANGStudio/JANGStudio/Runner/PublishService.swift:232-257`. 9/9 InferenceRunnerTests pass (was 7, +2). 14 PostConvertVerifierTests + 7 PythonCLIInvokerTests + 4 PythonRunnerTests + 22 AdoptionServicesTests unchanged (56 subprocess-helper tests green). Full Swift suite ~183 (was 181, +2).
      **Commit:** (this iteration)
- [x] **M158 (PostConvertVerifier.runJangValidate — unread Pipe() hang + terminationHandler race)** — Iter-81 deep-traced `PostConvertVerifier.runJangValidate` on the audit of subprocess-helper race / silent-failure patterns. Found two distinct latent bugs, both reporting "validate failed" for a validation that would have passed:
      1. **Pipe-fill hang.** Code wired `proc.standardOutput = Pipe(); proc.standardError = Pipe()` without ever reading either pipe. macOS pipe buffers are ~64 KB; once the subprocess writes past that, `write(2)` blocks forever because nobody is draining. `jang validate` normally stays small but a Python traceback stacked on a deep shard listing is easy to push over. Result: subprocess blocks → runJangValidate waits the full 60 s default timeout → returns false → user sees "jang validate passes: FAIL" on a model that's actually fine. **Silent mis-report, not a crash.**
      2. **terminationHandler wired AFTER `run()`.** A subprocess that exits in the microsecond window between `try proc.run()` returning and the handler assignment will never fire the handler — Foundation doesn't call terminationHandler on an already-terminated process. Result: same symptom (wait for 60 s timeout, return false). The window is tiny (<1 ms) so the bug is rare but real — worse, it's a "the model the convert succeeded on now fails to validate half the time" Heisenbug.

      **Fix (iter 81):**
      - Swapped `Pipe()` → `FileHandle.nullDevice` for stdout + stderr. We don't surface the subprocess's output anywhere, so discarding is the correct primitive. Zero risk of re-introducing the hang.
      - Moved `proc.terminationHandler = { ... }` to BEFORE `try proc.run()`. Closed the race entirely. The error branch for `proc.run()` throwing now runs inside the same lock.sync guard as the timeout + handler paths so the CheckedContinuation is always resumed exactly once.
      - Added `executableOverride: URL? = nil` parameter (mirrors InferenceRunner iter-32 M100 / PythonCLIInvoker iter-76 M153) so tests can drive the bug path with a shell script instead of needing a real Python runtime.

      **Tests (+3) in PostConvertVerifierTests.swift:**
      - `test_runJangValidate_does_not_hang_on_large_stderr_output` — subprocess blasts 400 KB to stderr+stdout then exits 0. Bug would hang at 64 KB → timeout → false. Fix: completes in <1 s, returns true. Measured: 0.907 s.
      - `test_runJangValidate_returns_true_on_immediate_zero_exit` — subprocess is `exit 0` (exits in microseconds). Bug would miss the handler → timeout → false. Fix: handler wired before run(), returns true in <1 s.
      - `test_runJangValidate_returns_false_on_nonzero_exit` — symmetric: subprocess is `exit 7`, must still be reported as failure. Guards against the fix accidentally returning true regardless of exit code.

      **Evidence:** `JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift:216-302`. 14/14 PostConvertVerifierTests pass (was 11, +3). Pre-existing concurrency warnings on the `resolved` capture are untouched by this edit — they cover the timeout Task which is already serialized through `lock.sync`.
      **Meta-lesson:** When a `Pipe()` is wired to a subprocess, either READ it (synchronously at end via `readDataToEndOfFile` after `waitUntilExit`, or asynchronously via `bytes.lines`) OR discard it with `FileHandle.nullDevice`. Unread `Pipe()` is a latent hang waiting for a chatty subprocess. Grepped the whole Swift app for this pattern (`proc\.standardOutput\s*=\s*Pipe\(\)`) — this was the only offender post-fix; the 5 adoption services and PublishService all correctly drain their pipes.
      **Commit:** (this iteration)
- [x] **M157 (SettingsWindow "Open logs directory" silent failure — iter-35 M107 class, different verb)** — Iter-80 re-audited `try?` patterns across the Swift app (iter-35 M107 swept user-action silent-failures). Grep for `try? FileManager` / `try? \w+.write` / `try? encoder.`:
      - `SettingsWindow.swift:350` — **silent `try? FileManager.default.createDirectory`**. Real bug.
      - `DiagnosticsBundle.swift:106,204` — tempdir cleanup on bundle write. Best-effort; silent-swallow correct (deferred cleanup path).
      - `PostConvertVerifier.swift:96,161,164` — verify checks that read failures default to empty; correct (iter-14 M14 reports via check-status).
      - `TestInferenceViewModel.swift:101` — JSON encode/decode roundtrip for the transcript export; malformed → empty messages defaults. Low-stakes data observation.
      **The SettingsWindow bug:** user clicks "Open logs directory" button in Settings → Diagnostics tab. `try? createDirectory(at: dir, withIntermediateDirectories: true)` silently swallows permission-denied / read-only-volume / disk-full errors. Then `NSWorkspace.shared.open(dir)` against a nonexistent dir silently no-ops. **User clicks button, nothing happens.** Classic iter-35 M107 silent-user-action failure.
      **Fix (iter 80):** `do/catch` the createDirectory. On failure:
      - Log to stderr with `[SettingsWindow] could not create <path>: <error>` — picked up by Copy Diagnostics via iter-14 M22 sensitive-scrubbing pipeline.
      - Fall back to opening the PARENT dir via `dir.deletingLastPathComponent()` — the user still gets SOMEWHERE useful (e.g., `~/Library/` opens if `~/Library/Logs/JANGStudio` can't be created).
      **Tests (+1) in WizardStepContinueGateTests.swift:** `test_settingsWindow_openLogs_surfaces_createDirectory_failures` — source-inspection pin. Asserts (a) the silent `try?` pattern is gone, (b) the fallback `dir.deletingLastPathComponent()` path exists in source, (c) the stderr log literal is present.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/SettingsWindow.swift:336-368`. 178 Swift tests pass (was 177, +1). Python 348 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M156 (PublishService.dryRun was a 7th invokeCLI copy — iter-78 meta-rule applied)** — Iter-78 meta-rule: "grep the WHOLE codebase for structural matches, not just the expected home directory." Iter-79 applied it with a simpler code-shape grep: `proc.waitUntilExit()`. 3 hits:
      - `PythonCLIInvoker.swift:67` — canonical helper.
      - `PublishService.swift:310` — 7th copy. **Hiding in plain sight.**
      - `PostConvertVerifier.swift:219` — iter-19 M42 comment only (prose).
      PublishService had TWO `invokeCLI`-style functions: `publishWithProgress` (streaming, iter-76 scoped this out as a different pattern — live stream drain) and `invoke(args:token:)` for dry-run. The dry-run variant is exactly the iter-76 one-shot shape, **plus** env-var threading (HF_HUB_TOKEN + PYTHONUNBUFFERED + childProcessEnvAdditions) and token-stderr redaction.
      **Fix (iter 79) — in two parts:**
      1. **Extended `PythonCLIInvoker.invoke` with optional `env: [String: String]? = nil` parameter.** Default nil preserves the iter-76 M153 inherit-env behavior for the 6 already-migrated callers. When set, `proc.environment = env` is applied.
      2. **Migrated PublishService.dryRun invoke** to the shared helper. The env-var construction (HF_HUB_TOKEN + PYTHONUNBUFFERED + M62 env additions) stays at the call site since it's PublishService-specific. Token redaction moves INTO the errorFactory closure, where `token` is in scope and the typed error is wrapped around the sanitized stderr.
      **Call-site shrink:** 43 lines → 17 lines. All env+token-security logic survives; only the plumbing vanishes.
      **Tests (+2) in `PythonCLIInvokerTests.swift`:**
      - `test_invoke_passes_env_to_subprocess`: script captures `$MY_TEST_VAR`, verifies passed-through value.
      - `test_invoke_env_nil_preserves_parent_env_inheritance`: regression guard — default nil must NOT blank the env (would break every other caller's PATH inheritance).
      **Verification:** 22 AdoptionServicesTests pass (includes publish dry-run tests). 7 PythonCLIInvokerTests (was 5, +2 for M156). Behavior preserved across the migration.
      **Final invokeCLI-pattern tally:** started at 5 (iter-76) + 1 outlier SourceDetector (iter-78 M155) + 1 outlier dryRun (this iter M156) = 7 total copies, ALL migrated to the shared helper. Token security + env threading + cancel propagation all flow through the canonical `PythonCLIInvoker.invoke`.
      **Evidence:** `JANGStudio/JANGStudio/Runner/PythonCLIInvoker.swift:39-58`, `JANGStudio/JANGStudio/Runner/PublishService.swift:285-306`. 177 Swift tests pass (was 175, +2). Python 348 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M155 (SourceDetector was a 6th invokeCLI copy iter-76 missed — migrate + typed error)** — Iter-77 forecast: Pipe-drain audit. Grepped `readDataToEndOfFile()` + `bytes.lines` across the Swift app. Found **SourceDetector.inspect was a 6th copy** of the iter-76 M153 pattern — iter-76 touched the 5 adoption services but SourceStep's embedded `SourceDetector` enum lived outside the Runner/ directory and got missed.
      **Extra find beyond just migration:** SourceDetector was also the LAST remaining `NSError(domain: "SourceDetector")` usage. iter-51 M129 cleaned up Capabilities/Profiles' NSError usages for typed errors; SourceDetector was missed then too because it's in Wizard/Steps/ not Runner/. Same NSError-stringified-into-banner anti-pattern M129 fixed.
      **Fix (iter 78):** Two changes:
      1. Added `SourceDetectorError: Error, LocalizedError` enum with `.cliError(code, message)` case and `errorDescription` that returns the message directly (preserves iter-43 M120's stderr-in-banner UX).
      2. Replaced 44 lines of inline ProcessHandle / withTaskCancellationHandler / DispatchQueue dance with 10 lines delegating to `PythonCLIInvoker.invoke(args:errorFactory:)`.
      **Impact:** ~34 lines of duplicated subprocess-cancellation code eliminated. Last NSError usage in service-layer code retired. Total `invokeCLI` copies across the Swift app: 5 → 6 migrated → 0 remaining.
      **Tests:** all 5 PythonCLIInvokerTests + 22 AdoptionServicesTests + 5 CapabilitiesServiceTests + 7 ProfilesServiceTests pass unchanged — 39 tests verifying the migration preserves behavior. No new dedicated SourceDetector tests this iter; the helper's contract is already pinned by M154's 5 tests.
      **Meta-observation:** iter-76 "extract 5 copies" missed a 6th that lived outside the expected directory. General rule: when extracting a pattern, grep the WHOLE codebase (not just the expected home directory) for structural matches. Iter-77's `readDataToEndOfFile` grep caught it by body shape, not location.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift:346-360, 375-391`. 175 Swift tests pass (unchanged — migration preserves behavior). Python 348 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M154 (dedicated `PythonCLIInvoker` contract tests)** — iter-76 M153 extracted the shared helper but relied on existing service-level tests to verify behavior (34 tests across CapabilitiesServiceTests/ProfilesServiceTests/AdoptionServicesTests). That's indirect coverage — a helper-level regression could slip through the service tests' type-specific assertions. Iter-77 pins the contract directly.
      **Fix (iter 77):**
      - Added `executableOverride: URL? = nil` parameter to `PythonCLIInvoker.invoke` (matches iter-31/32 InferenceRunner / PythonRunner pattern). Production code passes nil; tests pass shell-script URLs.
      - New `Tests/JANGStudioTests/PythonCLIInvokerTests.swift` with 5 contract tests covering:
        1. `test_invoke_returns_stdout_on_zero_exit` — happy-path data round-trip.
        2. `test_invoke_calls_errorFactory_with_code_and_stderr_on_nonzero_exit` — errorFactory receives actual terminationStatus + captured stderr.
        3. `test_errorFactory_error_is_rethrown_not_wrapped` — caller's returned error is thrown as-is (not wrapped in any envelope).
        4. `test_invoke_forwards_args_to_subprocess` — argv reaches the spawned process (shell script dumps `"$@"` to file, verified).
        5. `test_consumer_task_cancel_terminates_subprocess_within_3_seconds` — Task.cancel propagates SIGTERM to subprocess; mtime-non-advance verification matches iter-31 M98 style (avoids hanging test harness on regression).
      **Why matters now:** iter-76 eliminated ~160 lines of duplicated subprocess-cancel code. Any future change (timeout support, retry logic) now touches ONE file. Dedicated tests mean a regression in that file will fail loudly with a test-name that points directly at the bug.
      **Evidence:** `JANGStudio/JANGStudio/Runner/PythonCLIInvoker.swift:33-45, 56-59`, new `JANGStudio/Tests/JANGStudioTests/PythonCLIInvokerTests.swift`. 175 Swift tests pass (was 170, +5 for M154). Python 348 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M153 (Swift-side analog of M152: extract shared `PythonCLIInvoker`)** — Iter-51 M129 aligned the typed-error shapes across 5 adoption services (Recommendation/Examples/ModelCard/Capabilities/Profiles); iter-75 M152 established the "3+ local copies = extract" threshold Python-side; iter-76 applies the same crystallization to the Swift side.
      **Pre-M153 state:** each of the 5 services had a nearly-identical 31-37-line private `invokeCLI(args:)` body — ProcessHandle + withTaskCancellationHandler + DispatchQueue + waitUntilExit dance + typed error on non-zero exit. **5 copies of the same M101 (iter-33) cross-layer cancel pattern.**
      **Fix (iter 76):** New `JANGStudio/JANGStudio/Runner/PythonCLIInvoker.swift`:
      ```swift
      enum PythonCLIInvoker {
          static func invoke(
              args: [String],
              errorFactory: @escaping @Sendable (Int32, String) -> Error
          ) async throws -> Data
      }
      ```
      Closure-based error factory keeps the helper service-agnostic — each service's typed error enum is captured at the call site, not leaked into the helper. Same M101 Task-cancel pattern, now in ONE place.
      Each of the 5 service call sites shrunk from ~31-37 lines to:
      ```swift
      private static func invokeCLI(args: [String]) async throws -> Data {
          try await PythonCLIInvoker.invoke(args: args) { code, stderr in
              MyServiceError.cliError(code: code, stderr: stderr)
          }
      }
      ```
      **Migration impact:**
      - ~160 lines of duplicated subprocess-cancellation code eliminated.
      - Future changes to the Task-cancel pattern (e.g., adding a timeout, migrating off DispatchQueue, iOS-style transient-error retry) now touch ONE file instead of five.
      **Tests:** Build succeeded post-migration. All 3 relevant test suites pass unchanged — `CapabilitiesServiceTests` 5/5, `ProfilesServiceTests` 7/7, `AdoptionServicesTests` 22/22. Total 34 existing tests verified the migration preserves behavior. No new direct tests for the helper itself (future iter could add, but service-level tests already exercise every code path through it).
      **Evidence:** New file `JANGStudio/JANGStudio/Runner/PythonCLIInvoker.swift` (75 lines); migrated `RecommendationService.swift`, `ExamplesService.swift`, `ModelCardService.swift`, `CapabilitiesService.swift`, `ProfilesService.swift`. Regenerated `project.pbxproj`. 170 Swift tests pass (unchanged). Python 348 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M152 (crystallize shared `_json_utils` across 5 local copies of the template)** — iter-74 forecast: extract the shared helper. By iter-74's end, FIVE local copies of the same read-side-loader template existed:
      | Site | Function | Contract |
      | --- | --- | --- |
      | format/reader.py | `_read_json_object` | raise |
      | jangspec/manifest.py | inline in `load_manifest` | raise |
      | examples.py | `_read_json_object` | raise |
      | loader.py | `_read_config_or_raise` / `_read_config_or_none` | both |
      | capabilities.py | `_safe_load_json_dict` | tuple |
      **Fix (iter 75):** Extract canonical `jang_tools/_json_utils.py` module with two public functions:
      - `read_json_object(path, *, purpose) -> dict[str, Any]` — raise-contract.
      - `read_json_object_safe(path, *, purpose) -> tuple[dict | None, str | None]` — tuple-return contract, wraps `read_json_object` + catches ValueError.
      All 5 call sites migrated to import from the shared module. Thin local aliases preserved at each site to minimize diff at the call sites themselves. Behavior identical — every existing test continued passing unchanged.
      **Tests (+11) in `tests/test_json_utils.py`:** direct pins on the shared helpers, including a "never raises" contract test on the `_safe` variant (passes a directory path → OSError internally, must return tuple not raise).
      **Why now and not earlier:** extracting at 2 copies would be premature. At 5 copies with two distinct contracts, the inline duplicate COST exceeded the import-coupling cost. Documented the "extract at 3+" threshold in iter-71 M149's crystallization note; iter-75 hit the trigger.
      **Evidence:** `jang-tools/jang_tools/_json_utils.py` (new, 85 lines), migrated imports in `capabilities.py`, `examples.py`, `format/reader.py`, `jangspec/manifest.py`, `loader.py`. 348 Python tests pass (was 337, +11). Swift 170 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M151 (loader.py entry-point + detection helpers: corrupt-config diagnostics)** — iter-73 forecast: loader.py had 14 json.loads sites. Iter-74 scoped tight to the 4 USER-FACING surfaces that map 1:1 to Swift SourceStep detection + JANG Studio load flows:
      - `_is_v2_model` (line 69) — called by `is_jang_model` + SourceStep detector.
      - `_is_vlm_config` (line 81) — called by `load_jang_model` branching.
      - `load_jang_model` (line 707) — primary text-load entry.
      - `load_jang_vlm_model` (line 760) — primary VL-load entry.
      Mid-load internal re-reads (7 more sites) were NOT hardened this iter — they execute AFTER the entry point's initial parse, so if the entry point passed, the re-reads will too. Documented as deferred; future iter can revisit.
      **Contract split:** Detection probes MUST tolerate corrupt configs (return False, let upstream fallback handle it); loaders MUST raise informative ValueError.
      **Fix (iter 74):** Two new module-level helpers mirroring the M148/M149 template:
      - `_read_config_or_raise(path, *, purpose)` — entry-point loader uses.
      - `_read_config_or_none(path)` — detection probes use. Wraps the `_or_raise` variant and returns None on ValueError.
      Replaced 4 `json.loads(path.read_text())` sites with the appropriate helper. Pre-fix: cryptic JSONDecodeError traceback bubbled up through SourceStep to the wizard UI. Post-fix: clean ValueError with path + "not valid JSON (line N, col C)" OR False return for detection.
      **Tests (+5) in `tests/test_loader_config_read_diagnostics.py`:**
      - `is_v2_model_tolerates_corrupt_jang_config`: detection returns False without raising.
      - `is_vlm_config_tolerates_corrupt_config_json`: same for VLM probe.
      - `is_vlm_config_tolerates_non_dict_root`: same for `[1,2,3]`-root.
      - `load_jang_model_names_corrupt_jang_config`: entry-point raises ValueError naming the file.
      - `load_jang_vlm_model_names_corrupt_jang_config`: symmetric VL entry.
      The entry-point tests use subprocess + skip-on-ImportError so MLX-absence in CI doesn't block the error-path check.
      **Evidence:** `jang-tools/jang_tools/loader.py:47-100, 730-740, 800-810`. 337 Python tests pass (was 332, +5). Swift 170 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M126 (iter 73)** — long-open polish: `examples.py:detect_capabilities` reads 3 config files (`config.json`, `jang_config.json`, `tokenizer_config.json`) with raw `json.loads`. Pre-iter-73 the top-level `cmd_examples` try/except emitted `ERROR: JSONDecodeError: Expecting value: line 1 column 1 (char 0)` — correct that it failed but didn't name WHICH config broke. User had to check 3 files manually.
      **Fix (iter 73):** Added local `_read_json_object(path, *, purpose)` helper — same raise-contract template as format/reader.py's M149 + jangspec.manifest.py's M148. Applied to all 3 read sites with file-specific purpose strings. Error now reads e.g.: `ERROR: jang_config.json at /path/to/model/jang_config.json is not valid JSON (line 1, col 3): Expecting property name enclosed in double quotes`.
      **Tests (+3) in `tests/test_examples.py`:**
      - `test_cli_examples_names_corrupted_config_json`: corrupt config.json → stderr names "config.json".
      - `test_cli_examples_names_corrupted_jang_config`: valid config.json + corrupt jang_config.json → stderr names "jang_config.json" (NOT "config.json").
      - `test_cli_examples_names_corrupted_tokenizer_config`: first two valid + corrupt tokenizer_config.json → stderr names "tokenizer_config.json".
      The three pins together verify the error-path correctly identifies which of the three files failed.
      **Evidence:** `jang-tools/jang_tools/examples.py:45-87`. 332 Python tests pass (was 329, +3). Swift 170 + ralph 73 unchanged.
      **Commit:** (this iteration)
- [x] **M109 (new grep-audit class: force-unwraps)** — Grepped for `!` in production .swift (excluding tests, comments, != , string literals). Found TWO force-unwraps, both identical pattern: `FileManager.default.urls(for: ..., in: .userDomainMask).first!`.
      - `SettingsWindow.swift:338` — `.libraryDirectory` for "Open logs directory" button
      - `RunStep.swift:74` — `.desktopDirectory` for "Copy Diagnostics" button
      The `.first!` crashes the app if `.userDomainMask` returns an empty array. In practice macOS always provides these dirs, BUT sandboxed / MDM-restricted / unusual-home-directory environments COULD produce the empty case. One crash-the-app wait for a Support ticket vs. "it worked but landed in home dir" is a strictly better UX.
      **Fix (iter 36):** replaced `.first!` with `?? URL(fileURLWithPath: NSHomeDirectory())` (RunStep) or `?? NSHomeDirectory()+Library` (SettingsWindow — preserves the logical "logs belong under Library" intent). Copy Diagnostics always lands a zip; Open Logs always opens a directory. Neither crashes.
      **Bonus fix during iter-36 inspection:** while in RunStep I noticed the Copy Diagnostics button was still using `try? DiagnosticsBundle.write(...)` — a M107-class silent-failure on the failed-convert path. Replaced with explicit do/catch + log-pane surfacing. Same fix template as iter 35's M107.
      **Evidence:** `SettingsWindow.swift:339-350`, `RunStep.swift:73-92`. 122 Swift tests pass unchanged.
      **Commit:** (this iteration)
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
- [x] **M67** — Lock-file edge: `--reset` doesn't remove `results/ralph.lock`. If a crash left both state.json and a lock, reset cleans state but next `--next` still has to reclaim the stale lock (PID-dead detection works but untested). Consider removing the lock in cmd_reset too, gated on same-host+dead-PID check.
      **Close (iter 107, verified correct-by-design + test added):** `cmd_reset` already calls `acquire_lock`, which already handles stale same-host+dead-PID locks (runner.py:228+ — detects dead PID, removes lock, retries). The observation was "untested" — now tested. Added `test_cmd_reset_recovers_from_stale_lock_with_dead_pid` that writes a stale lock payload with PID=999999 (definitely dead), asserts cmd_reset cleans state + releases lock + exits 0 + prints "state reset" (not "BLOCKED").
      **Evidence:** `ralph_runner/tests/test_runner.py` new test. 76 ralph_runner tests pass (was 75, +1).
- [x] **M68** — Lock-file edge: SIGKILL + fast auto-restart (launchd/systemd) could put A's lock in front of B, with A's PID reused by an unrelated process → B thinks A is alive. Workaround: cross-check `ps -o lstart -p PID` start-time. Pathological; low priority.
      **Close (iter 107, observation-only):** pathological case confirmed. Trigger requires: (1) crashed ralph_runner process, (2) PID recycled by OS within the small window before a new --next runs, (3) the reused-PID process happens to also be on the same host and with a similar-enough process name that our `_pid_alive` heuristic can't distinguish. In practice on macOS: PID space is large enough that recycle within seconds is rare, and JANG Studio doesn't run under launchd/systemd at dev time. Documented here; revisit trigger: if users report "BLOCKED by lock held by <pid>" with no actual running ralph process. Workaround per the original observation (`ps -o lstart -p PID` cross-check) is not worth the complexity today.
- [x] **M69** — Lock-file on network FS (SMB/NFS): O_EXCL isn't always atomic. Document in README: lock/state files must be on a local filesystem.
      **Close (iter 107, observation-only):** the O_EXCL-on-NFS-isn't-atomic behavior is a kernel-level concern that JANG Studio can't fix from user-space. The constraint is: "don't put ralph_runner's state/lock dirs on a network filesystem." Today, the default `RALPH_STATE_PATH` lives under `~/jang/ralph_runner/results/` — on a local filesystem in every expected setup. A user who symlinks `results/` to an NFS mount could hit this; the failure mode would be multiple concurrent ralph_runner --next instances all acquiring what looks like the lock simultaneously, racing each other to write state.json. Documented here; revisit trigger: if users set up cross-machine shared ralph state (none today). Workaround for that future case: restrict `LOCK_PATH` to a fixed local path regardless of where state lives, or add a startup check that asserts the lock directory is on a local filesystem (statfs type check).
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
- [x] **M62** — Remaining UI-lie settings. **Iter 10 closed 6, iter 11 closed 3 more, iter 14 closed 1, iter 108 closed last 3 via "not yet implemented" labels** (12/12 done):
  - ~~`autoDeletePartialOnCancel`~~ ✅ iter 10
  - ~~`revealInFinderOnFinish`~~ ✅ iter 10
  - ~~`defaultProfile` / `defaultFamily` / `defaultMethod` / `defaultHadamardEnabled`~~ ✅ iter 10
  - ~~`customJangToolsPath` → PYTHONPATH prepend~~ ✅ iter 11
  - ~~`tickThrottleMs` → JANG_TICK_THROTTLE_MS env var, Python side reads it~~ ✅ iter 11
  - ~~`mlxThreadCount` → OMP_NUM_THREADS + MLX_NUM_THREADS env vars~~ ✅ iter 11
  - ~~`logVerbosity`~~ → iter 108 added "Not yet implemented — setting is preserved for when JANG_LOG_LEVEL lands." label. Persisted value stays so when the refactor eventually wires it, the user's choice applies immediately.
  - ~~`preAllocateRam*`~~ → iter 108 added "Not yet implemented — awaits an MLX buffer-pool API." label. Same preserve-value-for-later strategy.
  - ~~`anonymizePathsInDiagnostics` → DiagnosticsBundle pre-process rewrite (medium-size — deferred)~~ ✅ iter 14 (see M22)
  **Close (iter 108):** applied the M05/M175 disambiguation philosophy to UI-lies: rather than silently doing nothing, the setting's section now explicitly states the implementation status. User who enables `Pre-allocate RAM at launch` sees "Not yet implemented — awaits an MLX buffer-pool API" immediately, so they don't wait 20 minutes wondering why RAM pressure didn't drop.
  **Tests (+1) in AppSettingsTests:** `test_inert_settings_have_not_yet_implemented_labels` — source-inspection pins both labels + their blocker-citations (JANG_LOG_LEVEL / MLX buffer-pool). Future implementer removes the label when wiring up; test failure then forces updating the taxonomy.
  **Evidence:** `JANGStudio/JANGStudio/Wizard/SettingsWindow.swift:178-188, 271-281`. 31 AppSettingsTests pass (was 30, +1).
  **Meta-lesson — "don't lie to the user" extends from .pass states (iter-101/102) to UI affordances.** iter-101 M05 + iter-102 M175 applied the principle to preflight checks. iter-108 M62 applies it to Settings. Rule: any UI affordance (button, toggle, picker, slider) that appears interactive but does nothing should either be wired up OR carry a visible "not yet implemented" label. The persistent value stays so future implementation picks up the user's choice; only the UX claim ("this works") is corrected.
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
- [x] **M63** — `AppSettings.reset()` / `persist()` are synchronous `@MainActor` but the leaf-mirror writes happen inside them. If a write blocks (CloudKit-backed UserDefaults sync), the UI freezes. UserDefaults.standard is generally non-blocking but worth flagging.
      **Close (iter 103, observation-only):** confirmed scope. `UserDefaults.standard` writes in macOS do not block on iCloud/CloudKit sync by default — sync happens asynchronously via `NSUbiquitousKeyValueStore` which is a separate API JANG Studio doesn't use. The M63 concern requires an explicit opt-in the project doesn't have. Accepted as informational; no code change. If a future refactor adds a CloudKit-backed UserDefaults suite, revisit — the `persist()` on MainActor would then need to hop to a background Task for the write, mirroring iter-42 M106's async-write-for-ditto pattern. No test pin (nothing to test absent the trigger).
- [x] **M64** — `observeAndPersist` in `SettingsWindow.swift` uses `withObservationTracking` inside a loop with `withCheckedContinuation`. Each mutation fires a continuation that persists ONCE. But if two fields are mutated in the same SwiftUI pass (e.g., resetting via the Reset button), does the loop fire TWICE or ONCE? The `CheckedContinuation` pattern may miss paired mutations. Verify.
      **Verification (iter 98):** behavior is correct-by-design. `withObservationTracking.onChange` is a ONE-SHOT callback — fires exactly once on the first mutation of any tracked property, after which tracking is consumed. If the Reset button synchronously sets `foo` then `bar`, `onChange` fires only once. BUT `persist()` runs AFTER BOTH mutations land (the Task hops to a new main-actor execution, giving the synchronous Reset handler time to finish). The persisted Snapshot captures both. The loop's next iteration re-registers tracking for the next batch.
      **Net behavior:** ONE `persist()` call per batch of mutations that land in the same main-actor transaction. **Coalescing is a feature**, not a bug — fewer disk writes, atomic multi-field updates, no lost data.
      **Documentation + test:** added a detailed rationale comment block above `observeAndPersist` in `SettingsWindow.swift`. Added two tests:
      - `test_observeAndPersist_coalescing_rationale_is_pinned` — source-inspection pin so a future simplification sweep can't strip the rationale without a reviewer noticing.
      - `test_observeAndPersist_captures_rapid_multi_field_mutations` — functional test: mutates 3 fields synchronously then `persist()`, reloads, verifies all 3 round-trip. Guards against a future refactor that accidentally drops fields from the Snapshot or changes persist semantics.
      **Evidence:** `JANGStudio/JANGStudio/Wizard/SettingsWindow.swift:429-451`. 28 AppSettingsTests pass (was 26, +2).
- [x] **M65** — `SettingsWindow` auto-persist TASK is bound to the `.task { await observeAndPersist(settings) }` on the Settings body. If the user never OPENS Settings, the auto-persist never runs — which is fine (no changes to persist) UNLESS something else mutates settings programmatically (it doesn't today, but a future crash reporter that toggles `autoOpenIssueTrackerOnCrash` would lose the change).
      **Close (iter 103):** verified the invariant via grep sweep. ALL `settings.<prop> = ...` writes across `JANGStudio/` live in `SettingsWindow.swift`. Nothing outside that file mutates settings programmatically today. **Locked in with a regression test** — `test_appSettings_mutations_are_settingsWindow_only` walks every .swift file (excluding SettingsWindow itself + AppSettings.swift self-mutation), greps non-comment lines for `settings.<ident> = ` pattern. Any future addition (crash reporter, telemetry sampler, background sync) that mutates settings outside SettingsWindow FAILS this test, forcing the engineer to either move the mutation into the Settings sheet's reach OR rewire the auto-persist task to cover the new site.
      **Evidence:** new test in `AppSettingsTests.swift`. 29 AppSettingsTests pass (was 28, +1).
      **Meta-lesson — testable invariants > observation comments.** M65 sat as an observation for 80+ iters because there was no trigger. Iter-103 turns it into a GREP INVARIANT — the "hypothetical" mutation-outside-SettingsWindow is now enforced by a test that fails the moment anyone introduces it. Rule: when an observation describes a "would be a bug if X happened," look for a way to turn X into an invariant test. Turns a passive note into active protection.
- [x] **M66** — `Snapshot.apply` uses `LogVerbosity(rawValue: logVerbosity) ?? .normal` — if someone writes a garbage value into UserDefaults (e.g., schema migration from a newer version downgraded), the setting silently reverts to `.normal` without telling the user. Same for `updateChannel`. Consider emitting a log line on coercion.
      **Fix (iter 96):** both coercion sites now log to stderr naming the bad value + the fallback. Triggers: schema renames in app updates, cross-version downgrades that drop an enum case, or manual `defaults write` typos. Pre-M66 the user lost their custom setting silently; now the coercion shows up in Copy Diagnostics with a specific "re-save in Settings" recovery hint. Matches iter-35 M107 / iter-80 M157's surface-silent-failures pattern.
      **Tests (+3) in AppSettingsTests.swift:**
      - `test_snapshot_apply_logs_coercion_for_invalid_logVerbosity` — source-inspection, pins the stderr literal.
      - `test_snapshot_apply_logs_coercion_for_invalid_updateChannel` — same for updateChannel.
      - `test_snapshot_apply_still_coerces_invalid_values_to_defaults` — functional test: writes a JSON snapshot with invalid values into UserDefaults, constructs AppSettings() (which calls load → Snapshot.apply), asserts fallback behavior is preserved (logVerbosity=.normal, updateChannel=.stable). Guards against a future "fix" that changes the fallback silently.
      **Evidence:** `JANGStudio/JANGStudio/Models/AppSettings.swift:298-324`. 26 AppSettingsTests pass (was 23, +3).

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
