# Ralph Deep-Trace Investigation Log

Each entry records ONE deep trace. Created by iterations under the new Ralph prompt (see `PROMPT.md`). Oldest first.

## 2026-04-19 iteration 1 — new PROMPT + first deep trace

**Angle:** Category A (user journey) — Beginner picks a folder; what are all the failure modes?

**Deep trace walkthrough:**
1. Subprocess probe on bogus inputs:
   - Nonexistent dir → `inspect-source` clean exit 2 "config.json not found"
   - Malformed `config.json` → `recommend` clean exit 3 "ERROR: JSONDecodeError: ..."
   - Empty `{}` config.json → `inspect-source` succeeds with `unknown` everywhere; `recommend` succeeds with dense_llm fallback + 2 warnings. Clean UX.
2. Swift UI trace of "empty-cfg" folder pick:
   - `SourceStep.pickFolder:172` sets plan.sourceURL → clears detected → fires `detectAndRecommend`.
   - Both subprocess calls return success; UI shows Detected card with "unknown / Dense / UNKNOWN / 0 GB (0 shards)".
   - **BUG FOUND:** `ConversionPlan.isStep1Complete:57` only checked `sourceURL != nil && detected != nil`. With a 0-shard detected, it returned TRUE. User saw Continue button enabled — could progress into Step 2 with a non-model folder.
3. Fix: gate on `shardCount > 0`; add explicit "No .safetensors files found" red hint in Detected card pointing at common causes (wrong folder, incomplete download).
4. Verified 65/65 XCTest still pass after the change.

**Items touched:**
- Added M01 [x] — zero-shard Step 1 bypass (fixed this iteration)
- Added M02-M17 [ ] — 16 new questions spawned from the trace

**Commit:** (this iteration)

**Questions newly added:** 17 (M01-M17). Net +16 open items.

**Next iteration should pick:** Category B (data flow) or C (concern category). Rotating categories per the prompt's policy. Candidates: tokenizer file lifecycle (B), cancellation semantics (C), accessibility (C).

---

## 2026-04-19 iteration 2 — cancellation semantics

**Angle:** Category C (concern) — trace every code path involving cancel, interruption, or state transitions. Target: close M12/M13/M14, spawn new questions.

**Deep trace walkthrough:**
1. Read `PythonRunner.swift` + `RunStep.swift` end-to-end.
2. Normal success path: `for try await ev in r.run()` exits cleanly → `coord.plan.run = .succeeded`.
3. **BUG FOUND:** cancellation path. `PythonRunner.launch` finishes the stream without throwing on cancel (line 84-85 `else if cancelled { continuation.finish() }`). RunStep's for-await loop sees natural stream end, unconditionally sets `coord.plan.run = .succeeded`. User clicks Cancel, sees "Continue → Verify" button as if convert succeeded. Worse: `isStep4Complete` returns true so Step 5 activates — user would run verifier against partial/empty output and see cryptic failures.
4. **Fix:** track `cancelRequested` @State in RunStep. Cancel button sets it before firing runner.cancel(). After for-await exits without throw, branch on `cancelRequested` → `.cancelled` vs `.succeeded`. New cancel branch in UI shows orange "Cancelled" label + Retry + Delete partial output buttons.
5. M13 (stale log on navigation): traced SwiftUI @State lifecycle; `logs.removeAll()` at start of `start()` clears properly; re-entry guard prevents duplicate starts. Not a bug.
6. M14 (double-click Start): guard at start() top rejects re-entry; additionally disabled Cancel button when already cancelled. Benign.
7. PID-recycle concern for SIGKILL-after-3s task: `Task.detached { try? await Task.sleep(for: .seconds(3)); if proc.isRunning { kill(proc.processIdentifier, SIGKILL) } }` captures proc strongly. Even if user starts a new convert on a fresh PythonRunner, the timer-task targets the correct dead process (isRunning=false guards). Not a bug.

**Items touched:**
- M12 [x] — real bug fixed (Cancel→succeeded state leak)
- M13 [x] — verified not a bug, evidence logged
- M14 [x] — verified benign, cancel button hardening added
- M18-M23 [ ] — 6 new cancellation-adjacent questions spawned

**Commit:** (this iteration)

**New questions added:** 6 (M18-M23).

**Next iteration should pick:** Category B (data flow) — haven't done one yet. Candidates: tokenizer file lifecycle (source → convert copy → post-verify → TestInference load) OR settings value lifecycle (AppSettings → UserDefaults → subprocess env → actual behavior).

---

## 2026-04-19 iteration 3 — TestInference lifecycle

**Angle:** Category C (concern) — picked M19 from the open queue; InferenceRunner looked structurally identical to the pre-fix PythonRunner, so the suspicion was the same actor-deadlock.

**Deep trace walkthrough:**
1. Read `InferenceRunner.swift` end-to-end.
2. **BUG CONFIRMED (M19):** `generate()` is `actor`-isolated and blocks on `proc.waitUntilExit()` (pre-fix line 74). `cancel()` is ALSO actor-isolated (pre-fix line 107). User presses Cancel in `TestInferenceSheet` while `generate()` is running → `cancel()` call queues on the actor mailbox behind the in-flight `generate()`. `waitUntilExit()` blocks the actor thread until the subprocess exits naturally. Cancel button literally does nothing until generation finishes on its own. Exact same deadlock pattern as PythonRunner had before commit `6270214`.
3. **Fix applied:** Replaced `proc.waitUntilExit()` with `withCheckedContinuation` + `proc.terminationHandler` so the actor is NOT held during the wait. Added `cancelled: Bool` actor state; reset at the top of each `generate()`; set from `cancel()` before firing SIGTERM. After the continuation resumes, if `cancelled` is true, throw `InferenceError(code: cancelledCode=-2, message: "generation cancelled by user")`. Kept the SIGKILL-after-3s Task.detached with a strong `p` capture (same defense as PythonRunner).
4. **UI follow-through:** `TestInferenceViewModel.send()` catches `InferenceError` and was unconditionally setting `lastError = e.message`. After the fix, a user-initiated cancel would surface "generation cancelled by user" as a red error banner — looks like a failure. Added `e.wasCancelled` filter so deliberate cancels don't show as errors. `cancel()` already flips `isGenerating = false` so the UI returns to the prompt-entry state.
5. Verified 65/65 XCTest still pass.

**Items touched:**
- M19 [x] — real bug fixed (InferenceRunner actor-deadlock + UI error banner on cancel)
- M24-M28 [ ] — 5 new inference-lifecycle questions spawned

**Commit:** (this iteration)

**New questions added:** 5 (M24-M28). Net +4 open items after closing M19.

**Next iteration should pick:** Category B (data flow) — STILL haven't done one. Candidates: tokenizer file lifecycle (source → convert copy → post-verify → TestInference load — 4 handoffs to audit) OR settings value lifecycle (AppSettings → UserDefaults → subprocess env → actual CLI behavior). Rotating away from Category C twice in a row.

---

## 2026-04-19 iteration 4 — tokenizer lifecycle: source → convert → post-verify → inference

**Angle:** Category B (data flow) — finally. Trace a chat template + eos_token_id + temperature setting from the source HF folder through every pipeline stage into the generated text. Four handoffs to audit: source read, convert mutation/copy, post-verifier accept, inference load + use.

**Deep trace walkthrough:**
1. **Source read** (`convert.py:921-948`): Reads 7 tokenizer files by exact filename. JSON files parsed into dict; `tokenizer.model` read as bytes; `merges.txt` etc. as UTF-8 with `surrogateescape`. Coverage looks complete for the currently-supported families (llama/qwen/mistral/gemma/phi/MoE variants/MLA). T5-family `spiece.model` not covered — fine since T5 isn't supported.
2. **Convert mutation** (`convert.py:950-980`): eos_token_id fix for `qwen3_5`/`qwen3_5_moe` maps `248044 → 248046` in text_config + top-level + tokenizer_config.json. Good. Chat template preserved either inline (via the in-memory dict), as `chat_template.jinja`, or as `chat_template.json` (copied via `_safe_copy`). Three-form redundancy matches post-verifier expectations.
3. **Convert write** (`writer.py:80-105`): iterates `tokenizer_files` dict → dispatches on type (dict→JSON, bytes→raw, str→UTF-8 with surrogateescape). The explicit binary-vs-text dispatch was a late fix after mlxstudio#74 (tokenizer.model broke the whole convert at the very last step). Solid.
4. **Post-verify** (`PostConvertVerifier.swift:42-60`): Chat template check accepts ANY of inline/.jinja/.json. Tokenizer check requires (tokenizer.json OR tokenizer.model) AND tokenizer_config.json AND special_tokens_map.json. `runJangValidate` runs `jang_tools validate` as a subprocess off the main actor. No blocking.
5. **Inference load** (`inference.py`): **TWO BUGS FOUND**.
6. **BUG M29a — temperature silently ignored:** `_generate_text(model, tokenizer, prompt, max_tokens, temperature)` accepts `temperature` but never passes it to `mlx_lm.generate()`. Call site on line 56 was `generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)` — no `sampler` kwarg, no `temp` kwarg, nothing. TestInferenceSheet has a temperature slider (min=0.0, max=2.0) — slider is a UI lie. Every single generation was greedy argmax regardless of the user's setting. M17 in the checklist asked whether 0.0 works correctly; answer is "only 0.0 works, all others behave as 0.0".
7. **BUG M29b — chat template not applied:** The raw `prompt` string was passed directly to `generate(prompt=prompt, ...)` — no `apply_chat_template` call. mlx_lm's `generate` tokenizes the prompt as-is. For Qwen3/Llama3/Gemma/Phi chat models that expect the full `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n` structure, the model saw `"Hello"` instead, causing either infinite thinking loops (the exact failure mode `feedback_chat_template_rules.md` was written about) or garbage output. This is the root cause we've fought multiple times at different layers — now fixed at the inference-CLI layer.
8. **Fix:** Added `_apply_chat_template_if_any(tokenizer, prompt)` — checks for `.chat_template` on both the TokenizerWrapper (outer `mlx_lm` object) AND the inner HF tokenizer; falls through to raw prompt if neither has one; graceful fallback on `apply_chat_template` exception so a broken template var doesn't crash generation. Added `_make_sampler(temperature)` → returns `make_sampler(temp=...)` for `temp > 0`, `None` for greedy. Both wired through `_generate_text` with graceful kwarg-drop if mlx_lm is too old to accept `sampler`.
9. **Tests:** 5 new unit tests covering wrapped/unwrapped tokenizer, template-absent fallback, greedy vs non-greedy sampler, negative-temperature defensive behavior. 228/228 jang-tools tests pass. Swift tests not affected (fix is Python-side; no Swift test iterates on this).

**Items touched:**
- M29 [x] — real bugs fixed (temperature ignored + chat template not applied)
- M30-M34 [ ] — 5 new data-flow-adjacent questions spawned (tok/s counting, enable_thinking template var, VL chat template path, `_is_vl` false positives, `_safe_copy` silent-failure log lie)

**Commit:** (this iteration)

**New questions added:** 5 (M30-M34).

**Next iteration should pick:** Category D (memory cross-ref) — never done. Walk `memory/MEMORY.md` entries vs. what the app actually does. For example: does the app enforce `feedback_always_vl.md` (VL preprocessor files always copied)? Does it surface `feedback_no_concurrent_mlx.md` (don't run convert + inference in parallel) in the UI? Does the convert path respect `feedback_jang_must_stay_quantized.md` (never dequant to float16)? Or Category A user journey from the POV of an ADOPTER (not beginner) — someone who converted a model, wants to publish it to HF, test inference on it, export a model card, and write usage snippets into their own app.

---

## 2026-04-19 iteration 5 — memory cross-ref (Category D, never done)

**Angle:** Cross-reference memory entries (`feedback_*.md`, `project_*.md`) against actual app + CLI behavior. Pick one claim from memory, verify whether the app upholds it, fix the drift.

**Deep trace walkthrough:**
1. Read `feedback_no_concurrent_mlx.md` (8 days old): rule is "don't run MLX convert + MLX generate concurrently — 2× wallclock penalty at P8 GPU power". Checked app architecture: `WindowGroup` allows multi-window, no global mutex; user could start convert in window 1 and another in window 2 and saturate the GPU. Flagged as M36 — not fixing this iteration (needs design decision on single-instance enforcement vs queue-based coordinator).
2. Read `feedback_jang_must_stay_quantized.md` (35 days old): rule is "NEVER dequant to float16 at load". Checked `jang_tools/loader.py:126-275`: only GATE weights (small MoE routers) are dequanted, for Nemotron/Mistral4 correctness; outer `model.set_dtype(bfloat16)` sets activation dtype, not quant weights. Rule upheld. Memory still valid.
3. Read `feedback_always_vl.md` (33 days old): rule is "every conversion must copy preprocessor_config.json + video_preprocessor_config.json". Checked `convert.py:1018-1028`: both files are in `extra_configs` list and copied via `_safe_copy`. PostConvertVerifier rows #8 + #8b enforce presence gated on `detected.isVL` / `detected.isVideoVL`. Rule upheld.
4. Read `feedback_jang_studio_audit_coverage.md` (this session): says `tokenizer_class must be concrete (not "TokenizersBackend" — breaks Osaurus)`. **BUG FOUND.**
5. **BUG M35 CONFIRMED:** grep-audit of repo:
   - `convert_qwen35_jangtq.py:483-499` has the osaurus remap (TokenizersBackend → Qwen2Tokenizer).
   - `convert_minimax_jangtq.py` — doesn't have it (MiniMax sources are Qwen-tokenizer family so they were grandfathered in, but not defensively guarded).
   - `convert.py` (the MAIN JANG path, not the JANGTQ-specific ones) — `grep tokenizer_class` returns ZERO matches. **The regular JANG conversion path does not apply the Osaurus remap at all.** A user who picks a Qwen3.5-VL source and runs a regular JANG convert (not JANGTQ) ships a model that Osaurus will reject on load.
   - `PostConvertVerifier.swift:108-110` flags this as `status: .warn, required: false` — just a yellow label, user can click "Continue → Publish" and upload a broken model to HuggingFace.
6. **Fix:** Two layers — source-side remap AND output-side hard fail.
   - Added `_OSAURUS_TOKENIZER_MAP` to `convert.py` covering qwen/llama/mistral/gemma/phi. If `tokenizer_class == "TokenizersBackend"`, remap by source `model_type`. Logs `[osaurus-fix]` line matching the JANGTQ path style for grep-ability.
   - Upgraded `PostConvertVerifier` row #10 from `warn/required=false` → `fail/required=true` with a hint that directs future debuggers to the convert.py remap table. If a model_type slips through unmapped (e.g. minimax, deepseek_v2), the default is still Qwen2Tokenizer — verifier now catches cases where the source is unmapped AND remap defaulted to something incompatible.
   - Updated `CoverageMatrixTests.test_verifier_tokenizerClassConcreteIsWarn` → renamed to `test_verifier_tokenizerClassConcreteIsBlockingFailure` with assertions matching new policy.
7. **Verification:** 225/225 jang-tools tests, 65/65 Swift tests, all pass.

**Items touched:**
- M35 [x] — real bug fixed (missing Osaurus remap in main convert path + warn-only verifier)
- M36 [ ] — multi-window concurrent convert (memory violation; design task, deferred)
- M37 [ ] — Osaurus remap table incomplete (mistral4, deepseek, minimax, nemotron, glm — all fall through to Qwen2Tokenizer default)
- M38 [ ] — recommendation warnings are flat strings, no link to source memory doc
- M39 [ ] — Ralph runner end-to-end trace on macstudio never done
- M40 [ ] — verify-at-every-layer principle: Osaurus fix covers layers 1+3, no layer 2 (mid-convert) assertion

**Commit:** (this iteration)

**New questions added:** 5 (M36-M40). Net +4 open items after closing M35.

**Next iteration should pick:** Category A (adopter user journey, never done) — trace every step an adopter takes AFTER they've converted a model: model card generation, usage examples export, HF publish, and critically WHAT THE ADOPTER'S END-USER EXPERIENCES when they try to load the uploaded model. Or Category E (M-spawned) — pick one of M02/M03/M07 (mid-priority data flows). Rotating away from Cat D now that we've done one.

---

## 2026-04-19 iteration 6 — adopter journey: publish to HuggingFace (Category A, never done)

**Angle:** First Category A trace. Put the adopter hat on: they converted a 200 GB model, clicked "Publish to HuggingFace" in the wizard, entered a repo like `dealignai/MyModel-JANG_4K`, pasted their HF token. Trace the token through Swift → Python → HF API. Audit every place the token could leak.

**Deep trace walkthrough:**
1. Read `PublishService.swift:48-92` (Swift side).
2. **BUG M41 CONFIRMED (SECURITY):** Line 55 builds argv with `"--token", token` as plain-text positional. `proc.arguments` on macOS Process ends up as the actual argv of the child `python3` process. For the entire duration of `upload_folder` (which for a 200 GB model means ~30 minutes to hours), `ps aux | grep jang_tools` would show the token to ANY local user, Activity Monitor displays it under Open Files, and macOS CrashReporter captures it if the child ever crashes. This is a plain credentials-in-argv vulnerability.
3. Read `publish.py:22-34` (Python side). Line 29 `token = args.token   # assume literal token string` — accepts literal tokens on argv unconditionally. No rejection path.
4. **Fix (defense in depth — both sides):**
   - **Swift side:** PublishService now sets `HF_HUB_TOKEN` in the child's environment (env vars are visible only to the process and root, not to `ps aux`). Argv no longer contains `--token`. Stderr from failure paths is additionally scrubbed: `stderrRaw.replacingOccurrences(of: token, with: "<redacted>")` before the error bubbles up to the UI — covers the case where an HF exception embeds the Authorization header.
   - **Python side:** `--token <value>` is now ONLY accepted as a FILE PATH. Any non-file argument value triggers a clean exit 2 with hint pointing at HF_HUB_TOKEN. This is a layer-2 defense: even if something down the line starts passing tokens on argv again, the Python side refuses. Help text updated. `except Exception as e` on the upload path scrubs the token from the exception string before printing.
5. **Tests:** 2 new regression tests in `test_publish.py`:
   - `test_cli_rejects_literal_token_via_argv` — asserts exit 2 AND that the literal token value does NOT appear in stderr (which would defeat the purpose — would end up in shell history, CI logs, etc.).
   - `test_cli_accepts_token_file` — asserts the file-path branch still works (dry-run completes successfully).
   Plus existing `test_cli_rejects_missing_token` and `test_cli_dry_run` tests still pass.
6. **Verification:** 227/227 jang-tools tests pass, 65/65 Swift tests pass.

**Items touched:**
- M41 [x] — real security bug fixed (token in argv → env var only, with scrubbing on both sides)
- M42-M47 [ ] — 6 new publish-adoption-adjacent questions spawned:
  - M42: no cancellation on Python subprocess during verify
  - M43: publish progress never streams — user sees spinner for 30+ min with no ETA
  - M44: commit_url is emitted by Python but not decoded by Swift (most useful field is dropped)
  - M45: modelcard.py generate_card silently catches Exception and hides per-arch failures
  - M46: no repo-name validation before dispatching (cryptic HF error on "my model/repo")
  - M47: private vs public repo handling in modelcard link generation untested

**Commit:** (this iteration)

**New questions added:** 6 (M42-M47). Net +5 open items after closing M41.

**Next iteration should pick:** Continue Category A (more adopter-journey surface area): model card generation for every supported arch (M45), OR the commit_url/progress gap (M43/M44) which together are the "this took 30 minutes and I have no idea what happened" UX failure. Alternatively rotate to Category K (Ralph runner on macstudio — untouched) since we're now 6 iterations deep and have never looked at the test harness that's supposed to be proving the app works.
