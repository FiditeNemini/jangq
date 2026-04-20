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
