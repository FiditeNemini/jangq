# Ralph Loop — JANG Studio production audit

You are iterating production-readiness of JANG Studio — a signed/notarized macOS app that converts HuggingFace models to JANG and JANGTQ formats with full adoption surface (Test Inference chat, Usage Examples, Model Card, Publish to HuggingFace, Settings pane, dynamic profile discovery).

## Your job each iteration

1. Read `/Users/eric/jang/ralph_runner/AUDIT_CHECKLIST.md` in full.
2. Find the first `[ ]` item.
3. Investigate against the referenced memory files and codebase. Use read/grep tools; do not re-verify items already marked `[x]`.
4. Determine verdict:
   - **Proper** — already correct. Add one-line evidence note quoting the memory rule + pointing at the relevant file:line or existing commit. Mark `[x]`.
   - **Drift** — fix it, commit to the current branch, add the commit SHA + one-line note. Mark `[x]`.
   - **Blocked** — external dependency (e.g., HF terms acceptance). Mark `[-]` with blocker reason. Not skipped — just paused.
5. Commit the checklist: `audit: <item id> — <short result>`
6. End turn.

Ralph Loop will re-fire this prompt with your previous work visible in the checklist + git history. Next iteration picks the next `[ ]` item and repeats.

## Completion criteria

This loop runs forever (no `--completion-promise` set). Eric cancels manually when the checklist is fully green.

## Hard rules

1. **Never mark `[x]` based on assumption.** Evidence from file inspection or test run, or don't touch.
2. **Never disable an audit check to make it pass.** If something's broken, fix the root cause.
3. **No AI attribution in any commit.** `feedback_no_claude_attribution.md`. No `Co-Authored-By: Claude`, no "Generated with Claude Code".
4. **Author line is Jinho Jang, not Eric Jang.** `feedback_credits.md`.
5. **NEVER write to `research/`.** It's gitignored per `feedback_no_research_public.md`.
6. **NEVER `rm` anything under `/Volumes/EricsLLMDrive/`.** Read-only source models.
7. **NEVER force-push.** `feedback_all_instructions.md`.
8. **For write actions during Ralph iteration:** you have pre-authorization via this prompt. Do NOT stop and ask for confirmation on each fix — make the fix, commit, move on.
9. **If a fix requires running a long job (Ralph convert, MLX inference on a real model, bundle rebuild):** announce it in the commit message, don't block the iteration on it. Ralph cycles regardless.

## Memory index (read when relevant)

Rules live in `~/.claude/projects/-Users-eric-jang/memory/`. The index at `MEMORY.md` lists all files. High-signal memories for this audit:
- `feedback_chat_template_rules.md` — eos_token_id fixes per family, enable_thinking toggle
- `feedback_always_vl.md` — every model must include VL preprocessor files
- `feedback_jang_studio_audit_coverage.md` — exhaustive coverage requirement (image-VL vs video-VL, every config file, every layer type)
- `feedback_model_checklist.md` — 6-item per-model verification
- `feedback_runtime_before_quant.md` — 17-point runtime checklist; always test no-cache greedy before blaming quant
- `feedback_no_bandaid_fixes.md` — fix root causes with data, verify at every layer
- `feedback_naming_convention.md` — JANG_2S not JANG-2bit
- `feedback_jangtq_naming.md` — JANGTQ2/3/4 digit-only
- `project_bfloat16_fix.md` — 512+ expert models need bf16
- `project_mlp_asymmetry.md` — 512+ expert gate_proj=4-bit, down_proj=3-bit floors
- `project_mistral4_architecture.md` — 6 MLA+MoE fixes
- `project_mla_absorb_bug.md` — deepseek_v32 bf16 SDPA cast to fp32 on L==1 branch

## Codebase map (for item investigation)

- **jang-tools (Python)** — `/Users/eric/jang/jang-tools/jang_tools/`
  - `convert.py` — the main conversion entry point
  - `convert_qwen35_jangtq.py`, `convert_minimax_jangtq.py` — JANGTQ paths
  - `allocate.py` — profile → per-tensor bit allocation
  - `capabilities.py` — FAMILY_MAP; must cover every arch
  - `inspect_source.py` — source detection for the wizard
  - `examples.py`, `modelcard.py`, `inference.py`, `publish.py` — adoption CLIs
  - `profiles_cli.py`, `capabilities_cli.py`, `estimate_model.py` — discovery CLIs
- **JANGStudio (Swift app)** — `/Users/eric/jang/JANGStudio/JANGStudio/`
  - `Models/ConversionPlan.swift`, `AppSettings.swift` — observable state
  - `Runner/PythonRunner.swift`, `InferenceRunner.swift`, `BundleResolver.swift`, `ProfilesService.swift`, `CapabilitiesService.swift`, `ExamplesService.swift`, `ModelCardService.swift`, `PublishService.swift`, `DiagnosticsBundle.swift`, `CLIArgsBuilder.swift`
  - `Verify/PreflightRunner.swift`, `PostConvertVerifier.swift`
  - `Wizard/Steps/{SourceStep,ArchitectureStep,ProfileStep,RunStep,VerifyStep}.swift`
  - `Wizard/{TestInferenceSheet,UsageExamplesSheet,GenerateModelCardSheet,PublishToHuggingFaceSheet,SettingsWindow,WizardCoordinator,TestInferenceViewModel}.swift`
- **jang-runtime (Swift SwiftPM)** — `/Users/eric/jang/jang-runtime/Sources/`
  - `JANG/` — low-level loaders + inference engine
  - `JANGKit/` — high-level adopter-facing facade
  - `JANGCore/`, `JANGMetal/`, `JANGCoreMetal/` — format primitives + Metal kernels
- **ralph_runner** — this directory; runner.py drives the harness, audit.py implements A1-A18

## Test commands (for verification)

- Pytest suite: `cd /Users/eric/jang/jang-tools && /Users/eric/jang/.venv/bin/python3 -m pytest`
- Ralph unit tests: `cd /Users/eric/jang && /Users/eric/jang/.venv/bin/python3 -m pytest ralph_runner/tests`
- Swift package tests: `cd /Users/eric/jang/jang-runtime && swift test`
- XCTest: `cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' -only-testing:JANGStudioTests`
- Ralph one-shot: `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --next`

## End-of-turn behavior

When you've closed one checklist item this iteration, end your turn cleanly. Ralph re-fires the same prompt automatically. You'll see the updated checklist in the file + the new commit in git log.
