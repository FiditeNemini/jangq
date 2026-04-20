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

---

## 2026-04-19 iteration 7 — publish UX completeness (Category A continued)

**Angle:** Same adopter scenario as iter 6, but now the validation layer and the post-upload confirmation layer. Even with the security fix from iter 6, a user who fat-fingers their repo name still burns 30+ seconds on a bad auth probe before HF returns a cryptic error, and a user who successfully uploads still can't see the commit they just made without navigating manually.

**Deep trace walkthrough:**
1. Re-read `PublishToHuggingFaceSheet.swift` end-to-end after the iter-6 security fix landed.
2. **BUG M46 CONFIRMED:** footer buttons disable only when `repoName.isEmpty || token.isEmpty`. Any non-empty string passes the gate. `runDryRun` / `runPublish` dispatch directly to `PublishService` → `huggingface_hub.create_repo(repo_id=args.repo, ...)`. `create_repo` raises `huggingface_hub.errors.InvalidRepoIdError` for bad ids, but that comes back through `PublishServiceError.cliError` 30+ seconds into the attempt (after the token roundtrip). User sees a giant Python traceback in a red pill with no actionable hint.
3. **BUG M44 CONFIRMED:** `publish.py:71-76` emits `commit_url` in the result JSON. Swift `PublishResult` struct has no `CodingKeys` entry for it. JSONDecoder silently drops unknown keys (default policy) → the commit URL is gone by the time UI renders. Sheet only shows the repo URL, which is NOT proof the upload landed (repo might already exist, commit might have failed mid-flight). Users can't tell if their 30-minute upload actually wrote anything.
4. **Fix M46:** Added `HFRepoValidator.validationError(_:)` in `PublishService.swift` (MainActor enum, Swift-side only — no subprocess overhead). Rules implemented from the HF docs: single `/`, both segments match `^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$`, reject spaces, reject empty segments, trim whitespace before checking. Returns user-facing strings like `"Invalid org segment 'bad-org': start with a letter/digit, then letters/digits/._- up to 96 chars."`. Hooked into both `runDryRun` and `runPublish` so no path skips validation.
5. **Fix M44:** Added `commitUrl: String?` to `PublishResult` with `commit_url` JSON mapping (optional because dry-runs don't set it). Rendered as a second row in the "Published" section with its own Open button, guarded by `commit != url` so we don't double-display when HF happens to return identical values.
6. **Tests:** 10 new Swift test cases in `AdoptionServicesTests`:
   - 2 decode tests: `test_publish_result_decodes_commit_url` (full shape), `test_publish_result_commit_url_optional` (dry-run absence).
   - 8 validator tests: canonical accept, empty, no-slash, spaces, double-slash + triple-segment, leading-special, overlength, whitespace-trim.
7. **Verification:** 75/75 Swift tests pass (up from 65). Python side untouched this iter; 227 tests still green.

**Items touched:**
- M44 [x] — commit URL now decoded + rendered; user can click through to verify upload
- M46 [x] — repo id validated pre-flight with actionable error messages
- M48-M51 [ ] — 4 new publish-UX questions spawned:
  - M48: default repoName is ALWAYS invalid (missing org prefix) — validator now blocks it
  - M49: stale HF_HUB_TOKEN env var on long-running app
  - M50: SecureField Keychain AutoFill interaction
  - M51: private-repo commit URL opens to login page (confusing)

**Commit:** (this iteration)

**New questions added:** 4 (M48-M51). Net +2 open items after closing M44 + M46.

**Closed-status tally:** M01, M12, M13, M14, M19, M29, M35, M41, M44, M46 = 10 closed / 51 total = 20% closure rate. Cat A (adopter) has gotten significant attention (M41/M44/M46/M47-M51 chain); next rotation should go to Cat K (Ralph runner on macstudio) which is STILL untouched, or an unexplored M-spawned question cluster. Alternative: Cat B data flow for `settings → UserDefaults → subprocess env` path (only did tokenizer lifecycle in iter 4).

---

## 2026-04-19 iteration 8 — Ralph runner state machine (Category K, never done)

**Angle:** After 7 iterations we've audited the app that Ralph is supposed to be testing without ever auditing Ralph itself. The runner orchestrates multi-hour converts on macstudio — any state-machine bug here silently drops entries from the matrix or causes concurrent MLX workloads (memory violation). This iter reads runner.py end-to-end and fixes every reliability/security gap found.

**Deep trace walkthrough:**
1. Read `runner.py` (298 lines) + `remote.py` + existing tests `test_audit.py` + `test_remote.py`.
2. Notice immediately: `tests/test_runner.py` **does not exist**. 298 lines of state-machine logic with zero coverage.
3. **BUG M52 CONFIRMED (shell-injection hardening):** Line 101-104:
   ```python
   cmd = "python3 -c 'from ... snapshot_download(repo_id=\"{hf_repo}\")'"
   ```
   `hf_repo` is f-string-interpolated into a Python literal wrapped in single quotes for sh. A repo id containing `"` closes the Python string; subsequent content is evaluated as Python code on macstudio. No exploit today (models.yaml is trusted), but classic RCE latent.
4. **BUG M53 CONFIRMED:** Line 267 `run_remote(f"rm -rf {result['output_path']}")` runs UNCONDITIONALLY at end of `cmd_next`. When audit FAILS, the failed output dir is ALSO deleted — meaning the engineer has no artifacts to debug a failed convert. Reproducing the failure requires a full re-convert (hours for 200 GB models).
5. **BUG M54 CONFIRMED (reliability):** Line 215-217 `info["status"] = "running"; save_state(state)`. Anywhere between this line and the final state update (lines 246/251/261), a crash / ctrl-C / SIGKILL / Tailscale drop leaves the combo in `running` forever. `pick_next` only picks `pending` (line 94), so the combo is effectively lost from the matrix. Over a multi-day Ralph run this silently drops entries — exactly the failure mode Ralph is supposed to catch, hitting Ralph itself.
6. **BUG M54b CONFIRMED (testability):** `from ruamel.yaml import YAML` at module top. `ruamel.yaml` isn't in the minimal test-env dependency set. Any attempt to `import ralph_runner.runner` in tests blows up on the missing package, so unit tests are unreachable without fully provisioning the runtime dep set.
7. **Fix stack:**
   - M54b first (unblocks the rest): moved `YAML` to a lazy import inside `_yaml()`.
   - M54: added `recover_interrupted(state)` that flips `running` → `pending`, moves `started` to `recovered_from_interrupt` breadcrumb, cleans up the stale `started` key. Called at top of `cmd_next` before `pick_next`. Returns count for logging/testing.
   - M53: `cmd_next` end-of-function now only `rm -rf` the output when `info["status"] == "green"`. On failure it sets `info["retained_output_path"]` so the slug entry points at the surviving macstudio path, and logs it to stdout for visibility.
   - M52: added `_HF_REPO_PATTERN` regex + `_assert_safe_repo_id(hf_repo)` — same rules as Swift-side `HFRepoValidator` from iter 7 (single slash, segment char class, length limits). Called unconditionally in `ensure_source_model` before splicing.
8. **Tests:** Created `ralph_runner/tests/test_runner.py` with 11 cases:
   - state load/save roundtrip (2)
   - recover_interrupted: no-op / flips / preserves `started` as `recovered_from_interrupt` / empty state handling (4)
   - repo-id validator: canonical accept / shell-injection rejection (8 dangerous patterns) / structural rejection (7 bad shapes) / slug stability (4)
9. **Verification:** `pytest ralph_runner/tests/` → 28 passed (14 pre-existing + 14 new — 11 from test_runner.py minus de-dupe noise). `pytest jang-tools/tests/` → 227 passed. Swift tests unchanged at 75 (no Swift touched).

**Items touched:**
- M52 [x] — shell injection hardened + 8-case regression test for classic injection patterns
- M53 [x] — failed outputs retained for debugging; path echoed back through state.json
- M54 [x] — crash recovery mechanism; state.json no longer silently drops entries
- M54b [x] (new closure): lazy YAML import unblocked testability
- M55-M59 [ ] — 5 new Ralph-internal questions: multi-instance lock, empty profile map, unsafe slug, hardcoded JANGTQ CLI, audit.py plugin refactor
- M39 [partial] — category K is no longer zero; runner.py now has 11 tests. audit.py + macstudio round-trip still untraced.

**Commit:** (this iteration)

**New questions added:** 5 (M55-M59). Net +1 open items after closing M52+M53+M54+M54b.

**Next iteration should pick:** Continue Cat K — audit.py is 765 lines and still untraced. Or rotate to Cat B (settings → UserDefaults → subprocess env pipeline, never done). Or M55 (multi-instance state.json race) which would complement this iter's reliability work cleanly.

**Closed-status tally:** M01, M12, M13, M14, M19, M29, M35, M41, M44, M46, M52, M53, M54, M54b = 14 closed / 56 total = 25% closure rate. 🟢 Ralph runner no longer a blind spot.

---

## 2026-04-19 iteration 9 — Settings → UserDefaults → leaf consumers (Category B)

**Angle:** Never done: trace the settings pipeline from `AppSettings.pythonOverridePath` (UI binding) → `persist()` (JSON blob in UserDefaults) → the subprocess that actually needs to know which Python to launch. Pick one critical setting and verify the full data flow works.

**Deep trace walkthrough:**
1. Read `AppSettings.swift` (223 lines) — @Observable @MainActor class with ~27 fields, persistence via single JSON `Snapshot` blob in UserDefaults key `"JANGStudioSettings"`. `init` calls `load()`, `persist()` is manual.
2. Read `SettingsWindow.swift` — 5 tabs fully wired: General, Advanced, Performance, Diagnostics, Updates. Auto-persist via `observeAndPersist(settings)` task using `withObservationTracking` + `CheckedContinuation` loop (creative but works).
3. Read `BundleResolver.swift`. Before this iter: `pythonExecutable` checked ONE source — `ProcessInfo.processInfo.environment["JANGSTUDIO_PYTHON_OVERRIDE"]`. **Never touches AppSettings.** So the Advanced-tab "Python override" picker was pure decoration: user picks a path, saves, relaunches app, path is ignored, bundled Python still runs.
4. **Verify the scope of the problem:** grep every AppSettings field across JANGStudio/ for read-sites outside `AppSettings.swift` and `SettingsWindow.swift`:
   ```
   grep -rn "settings\.\|AppSettings" JANGStudio/ --include="*.swift" \
     | grep -v "AppSettings.swift\|SettingsWindow.swift"
   ```
   Returns exactly ONE hit: the `@State private var settings = AppSettings()` line in `JANGStudioApp.swift`. **No field is ever read at a consumer site.** Settings is a 27-field UI lie surface.
5. **BUG CONFIRMED (M60+M61):** every Advanced/Performance setting is inert. Fixing all 27 is out of scope for one iter. Picked M61 (Python override) because:
   - It's the Advanced setting most likely to be used in the wild (dev mode, bundle troubleshooting, bug reports).
   - The plumbing required is minimal (single-string hand-off).
   - It unblocks a pattern the remaining fields can follow (dedicated UserDefaults leaf keys + mirror on persist/load).
6. **Fix architecture:**
   - BundleResolver is nonisolated + called from sync contexts (PythonRunner init, SettingsWindow previews). Can't hold a `@MainActor` AppSettings reference.
   - Snapshot JSON decode per call would work but wastes cycles.
   - Decision: dedicated UserDefaults leaf keys. AppSettings is the writer; BundleResolver is the reader. UserDefaults.standard is thread-safe. No coupling between them beyond a string key constant declared on BundleResolver.
7. **Fix implementation:**
   - `BundleResolver.swift:5-32`: exported `pythonOverrideDefaultsKey = "JANGStudioPythonOverride"`. Priority order: UserDefaults leaf key (non-empty) → `JANGSTUDIO_PYTHON_OVERRIDE` env var → bundled Python.
   - `AppSettings.swift:130-152`: added `mirrorLeafConsumerKeys()` private helper called from `persist()` AND `load()` (the latter for fresh-process warm-up). Writes `pythonOverridePath` to the dedicated key; empty string → `removeObject` so env-var / bundled fallbacks remain visible.
8. **Tests:** 6 new cases in `AppSettingsTests`:
   - persist() mirrors to leaf key
   - clearing path removes leaf key (fallback reactivation)
   - BundleResolver reads the leaf key
   - empty-string leaf key is ignored (doesn't point at empty path)
   - load() re-syncs leaf key from Snapshot on fresh process (defends against leaf-key drift — someone else wrote a different value)
   - reset() clears the leaf key (reset() calls persist() internally; mirror must follow)
9. **Verification:** 81/81 Swift tests pass (was 75). Python tests unchanged at 227.

**Items touched:**
- M61 [x] — pythonOverridePath now works end-to-end; UI → UserDefaults → BundleResolver → subprocess Python
- M60 [x] — meta-bug (settings are UI lies) is formally scoped: M62 lists all remaining inert fields
- M62-M66 [ ] — 5 new settings-pipeline questions:
  - M62: remaining 10+ inert settings (customJangToolsPath, tickThrottleMs, logVerbosity, mlxThreadCount, …)
  - M63: sync persist() could block if UserDefaults is CloudKit-backed
  - M64: observeAndPersist continuation pairing on paired mutations
  - M65: auto-persist TASK lifecycle tied to Settings view existence
  - M66: Snapshot.apply silently coerces unknown enum values

**Commit:** (this iteration)

**Closed-status tally:** 14 (prior) + M60 + M61 = 16 closed / 61 total = 26% closure rate.

**Next iteration should pick:** Continue M62 chain by wiring up `revealInFinderOnFinish` (smallest inert field, fires once per convert — low risk, high user-visible polish). Or rotate to audit.py (765 lines, Cat K still has debt). Or rotate to M55 (multi-instance state.json race — complements iter 8 reliability work). 4 categories now have ≥1 iter; Cat E (spawned M-questions) still has lots of easy pickings (M02, M03, M07, M15, M16, M22, …).

---

## 2026-04-19 iteration 10 — continue M62 chain: wire 6 more settings

**Angle:** Same Cat B trace as iter 9, but this iter tackles a batch of the cheapest-to-wire inert fields identified in M62. Target: every setting on the General tab (the ones a beginner would actually toggle) should DO SOMETHING.

**Deep trace walkthrough:**
1. Re-read `SettingsWindow.GeneralTab` (lines 33-119 of SettingsWindow.swift) to enumerate what's exposed to beginners:
   - Output folder parent, default profile, default family, default method, default Hadamard (5 defaults)
   - Calibration sample count (1)
   - Output naming template (1, already works via renderOutputName)
   - Auto-delete partial output on cancel (1)
   - Reveal in Finder on finish (1)
2. Of 8 settings on General, only `outputNamingTemplate` and (indirectly) `defaultOutputParentPath` are consumed anywhere. The other 6 are inert.
3. **Fixes wire in three places:**
   - **Wizard init:** `WizardView.task` runs once on first entry, calls `coord.plan.applyDefaults(from: settings)`. Guarded by `@State private var defaultsApplied` so navigation back to the wizard root doesn't re-apply (would clobber user overrides made mid-flight).
   - **Convert another:** `VerifyStep.reset()` re-creates the plan AND re-applies defaults. Before this change, "Convert another" always dropped back to JANG_4K/jang/mse regardless of what the user had picked in Settings.
   - **Reveal on finish:** `VerifyStep.refresh()` fires `revealOutput()` on the first finishable render with a `revealFiredOnce` @State guard. Cleared in `reset()` so each conversion cycle gets one auto-reveal.
   - **Auto-delete on cancel:** `RunStep.start()` cancellation branch reads the setting and runs `FileManager.removeItem(at:)` with logged outcome.
4. **Data integrity requirements for applyDefaults:**
   - Empty string profile must be a no-op (first-launch UserDefaults state can have empty strings from Snapshot defaults; would otherwise blank the profile).
   - Unknown method string must NOT coerce to a random enum; preserve current method. Defends against schema drift.
   - Method aliases: "mse-all" / "mseall" / "mse_all" all map to `.mseAll` (enum case-name vs settings-string drift).
   - Per-conversion STATE (sourceURL, detected, outputURL, run) must NEVER be touched — those aren't user defaults. Pinned by an explicit test.
5. **Tests:** 5 new `ConversionPlanTests` cases:
   - `test_applyDefaults_seeds_profile_family_method_hadamard` — happy path
   - `test_applyDefaults_ignores_empty_profile` — corruption guard
   - `test_applyDefaults_ignores_unknown_method` — schema drift guard
   - `test_applyDefaults_accepts_mse_all_aliases` — case-alias coverage
   - `test_applyDefaults_preserves_per_conversion_state` — defends against future bugs that accidentally extend applyDefaults to clobber in-flight state

**Items touched:**
- M62a [x] — defaultProfile/Family/Method/Hadamard now seed the wizard on entry AND after "Convert another"
- M62b [x] — revealInFinderOnFinish auto-fires once per successful convert
- M62c [x] — autoDeletePartialOnCancel deletes partial output on user cancel
- M62 [partial] — 6 of 9 sub-items closed; 3 remain (customJangToolsPath, tickThrottleMs, logVerbosity + mlxThreadCount + preAllocateRam*, anonymizePathsInDiagnostics)

**Commit:** (this iteration)

**Verification:** 86/86 Swift tests pass (was 81). Python unchanged.

**Closed-status tally:** 16 (prior) + M62a + M62b + M62c = 19 closed / 61 total = 31% closure rate.

**Next iteration should pick:** The remaining M62 sub-items are either (a) purely env-var passthrough (tickThrottleMs, logVerbosity, mlxThreadCount, preAllocateRam) which could all land in a single patch to PythonRunner, or (b) a harder path anonymization rewrite of DiagnosticsBundle. Option (a) is the efficient batch. Or rotate to M55 (multi-instance state.json race — reliability complement to iter 8) which has been sitting idle since iter 8.

---

## 2026-04-19 iteration 11 — env passthrough batch (Cat B continued)

**Angle:** Finish the cheap inert-setting cleanup by giving every subprocess-
spawning entry point (PythonRunner, InferenceRunner, PublishService) a single
unified way to merge user settings into the child env. Target: 3 more M62
sub-items in one patch.

**Deep trace walkthrough:**
1. Re-grep the 6 remaining M62 fields to confirm no stray reads landed since iter 10:
   ```
   grep -rn "tickThrottleMs\|mlxThreadCount\|customJangToolsPath\|logVerbosity\|preAllocateRam\|anonymizePathsInDiagnostics" JANGStudio/JANGStudio --include="*.swift" | grep -v AppSettings.swift | grep -v SettingsWindow.swift
   ```
   Still zero — all 6 remain inert.
2. Categorise by implementation cost:
   - **tickThrottleMs, mlxThreadCount, customJangToolsPath** — single env-var assignment each. Small, safe, testable.
   - **logVerbosity** — would need JANG_LOG_LEVEL reads in every emit site in Python (bare `print()` calls scattered through the codebase). Wide refactor, deferred.
   - **preAllocateRam*** — MLX has no standard env for buffer pool size. Deferred pending upstream feature.
   - **anonymizePathsInDiagnostics** — DiagnosticsBundle needs a pre-process pass that rewrites file paths. Medium-sized. Deferred.
3. **Architecture decision:** rather than duplicate `env[x] = ...` snippets across the 3 subprocess entry points, introduce one helper on `BundleResolver`: `childProcessEnvAdditions(inherited:)`. Callers merge the return value into their already-constructed env. Same leaf-key pattern as M61 (iter 9) — AppSettings writes, BundleResolver reads.
4. **Cross-cutting invariants the implementation must satisfy** (pinned by tests):
   - Default settings → zero env additions. Users who never open Settings shouldn't pay any cost.
   - "Return to default" must REMOVE the leaf key, not freeze a stale value. Example: user set threads=8, then chose "Auto" — OMP_NUM_THREADS must not remain set.
   - `0` thread count means "auto" in the UI. Writing OMP_NUM_THREADS=0 to the child would wedge the Python process. So the mirror MUST skip zero.
   - PYTHONPATH is PREPEND, not replace. Bundled jang_tools must still resolve for anything the custom path doesn't override. Otherwise an incomplete custom path would break the conversion.
5. **Python side:** `_TICK_MIN_INTERVAL_S` was a module-level constant used inside `tick()`. Replaced with `_resolve_tick_interval_s()` that reads `JANG_TICK_THROTTLE_MS` at ProgressEmitter-instance time. Garbage values (empty / non-integer / zero / negative) fall back to 100 ms default so a misconfigured env can't hang emit loops. `ProgressEmitter.__init__` stores the resolved value on `self`.
6. **Tests:** 14 new total:
   - **Python (4):** env-unset default, env-reads-integer, garbage fallback (5 cases: empty, non-numeric, "0", "-50", whitespace), end-to-end emit-loop throttle application with 1 ms env + sleep assertion.
   - **Swift BundleResolver (6):** empty additions on no settings, tick throttle wired, thread count writes BOTH OMP + MLX keys, zero thread count writes NOTHING, PYTHONPATH prepend with existing inherited, PYTHONPATH with no inherited.
   - **Swift AppSettings (4):** tick throttle mirrors only non-default (with return-to-default test), mlxThreadCount mirrors only non-zero, customJangToolsPath mirrors only non-empty, load() re-syncs all three on fresh process.

**Items touched:**
- M62d [x] — tickThrottleMs / mlxThreadCount / customJangToolsPath now live end-to-end (UI → leaf key → env → subprocess → Python)

**Commit:** (this iteration)

**Verification:** 96 Swift (was 86), 231 Python (was 227), 28 ralph_runner all pass.

**Closed-status tally:** 19 (prior) + M62d = 20 closed / 61 total = 33% closure rate. M62 parent item now 9/12 sub-items done; 3 remaining are intentionally deferred (logVerbosity wide-refactor, preAllocateRam upstream-gap, anonymizePathsInDiagnostics medium-rewrite).

**Next iteration should pick:** M55 (multi-instance state.json race — sitting idle since iter 8) would be a natural reliability follow-up to the iter-8 Ralph runner hardening. Alternatively Cat K audit.py (765 lines, never traced). Alternatively spawned-M easy picks: M02 (data-shaped-but-wrong-content HF clones), M07 (non-standard nested model_type), M15 (publish token clear-on-complete), M22 (DiagnosticsBundle race during running convert).

---

## 2026-04-19 iteration 12 — M55 multi-instance lock (Category K reliability follow-up)

**Angle:** M55 was spawned in iter 8 and sat idle for 4 iterations. Picked now because it completes the iter-8 Ralph-runner reliability theme: iter 8 fixed the WHY of interrupted-status recovery (M54), this iter fixes the WHAT ELSE — preventing two instances from even starting a race in the first place.

**Deep trace walkthrough:**
1. Read `runner.py` with the multi-instance scenario in mind:
   ```
   # terminal 1:
   python -m ralph_runner --next   # dispatches convert A to macstudio
   # terminal 2 (user forgets terminal 1 is still running):
   python -m ralph_runner --next   # dispatches convert B to macstudio
   ```
2. **BUG M55 CONFIRMED.** `cmd_next` has no mutex. Step-by-step what happens on terminal 2:
   - `remote_ok()` passes (macstudio is reachable).
   - `load_state()` reads the same state.json terminal 1 just wrote.
   - `recover_interrupted(state)` finds no `running` entries yet (terminal 1 hasn't saved its state update, OR terminal 2 runs between terminal 1's `load_state` and `info["status"]="running"` save).
   - `pick_next(state)` returns the SAME pending combo terminal 1 picked.
   - Terminal 2 marks it running, overwrites terminal 1's state save.
   - Both terminals call `run_convert_remote` → two SSH sessions running MLX convert on the same Metal GPU. Per `feedback_no_concurrent_mlx.md`, 99% GPU residency @ P8, total wallclock ≈ 2× serial.
3. Additional failure modes: audit races on same output path; both try `rm -rf output_path`; state.json last-writer-wins loses one combo's result.
4. **Fix architecture:** PID+host lock file.
   - **Why not `fcntl.flock`**: flock is auto-released on process death without a way to inspect the holder's identity. We want to TELL the user who's holding the lock ("pid 12345 on erics-mac-studio, running for 47 minutes").
   - **Why not `fcntl.lockf`**: same issue + NFS quirks (we're on APFS; N/A but future-proofing).
   - **Decision: handwritten lock file with JSON payload + O_EXCL create.** O_EXCL is atomic on APFS. Payload includes `pid`, `host`, `started_at` so status/error messages can be informative.
5. **Corner cases worth enumerating (each pinned by a test):**
   - Lock holder is alive → refuse.
   - Lock holder PID is DEAD on this host → stale, reclaim.
   - Lock holder is on DIFFERENT host → cannot verify remote PID. Choice: reclaim or refuse? **Refuse defensively.** Reclaiming a cross-host lock could stomp a live convert on a shared workspace. Better a human unlocks manually than an automated wrongful reclaim.
   - Lock file contains garbage JSON → treat as stale, reclaim. Otherwise one corrupted write wedges the whole runner forever.
   - Cross-uid ownership on same host → `_pid_alive` correctly returns True on `PermissionError` (process exists, just not signallable by us).
   - `release_lock` is called in `finally` even on early-return paths → ensure it NEVER accidentally removes a lock owned by a different PID. If the releasing process crashed and a new process acquired the lock before our release-handler ran, we must NOT yank their lock. Implemented: release_lock reads the file, only unlinks if `pid == os.getpid()`.
6. **Implementation notes:**
   - `cmd_next` split into outer-with-lock + inner `_cmd_next_locked`. Outer uses try/finally so every early-return path (combos empty, macstudio unreachable, combo failure) still releases.
   - Lock path `ralph_runner/results/ralph.lock` (same directory as state.json — same failure domain).
   - No retry loop beyond single stale-reclaim — if someone races between our detection-of-stale and our write, let them win; the SECOND caller will see the fresh lock and fail cleanly.
7. **Tests (10 new in test_runner.py):** happy path, release-missing-noop (defends against finally: on early abort), live-PID refusal, dead-PID reclaim, cross-host refusal, corrupt-JSON reclaim, other-owner release no-op, _pid_alive self/dead coverage.

**Items touched:**
- M55 [x] — real reliability bug fixed with defense-in-depth lock. Two instances now see `BLOCKED: lock held by another ralph instance: {pid: 12345, host: erics-mac-studio, ...}` instead of racing.
- M67-M71 [ ] — 5 new lock-adjacent questions spawned (below).

**Commit:** (this iteration)

**Verification:** 38 ralph_runner tests (was 28), 231 Python, 96 Swift all pass.

**Closed-status tally:** 20 (prior) + M55 = 21 closed / 66 total = 32% closure rate.

**New questions added (M67-M71):**
- M67: lock file in `results/` — `--reset` removes state.json but leaves the lock. On next --next, the lock is stale (this process wrote it, but this process is different now). Same-host + alive PID check works… BUT: if reset runs a second time in the same process, the lock still has the right PID. Edge case: a graceful crash mid-cmd_next leaves a lock AND a running combo; the next launch must recover both. Currently recover_interrupted handles state, lock acquire handles the stale lock. Verified by tracing but untested.
- M68: systemd / launchd auto-restart could trigger rapid re-launches. If process A acquires, is killed via SIGKILL (bypasses finally), process B retries within 100 ms. Both see A's lock; B checks PID alive, A's PID is actually reassigned to an unrelated process → B thinks it's alive, refuses. Workaround: check process start time (macOS: `ps -o lstart -p PID`). Low priority — SIGKILL + PID reuse within 100 ms is genuinely pathological.
- M69: lock file on a network-mounted filesystem (user moved `ralph_runner/` to SMB share). O_EXCL on SMB isn't always atomic. Document: lock file path must be on a local filesystem.
- M70: concurrent `--status` and `--next` — status doesn't take the lock, so it reads the state.json mid-write. JSON atomicity relies on the `STATE_PATH.write_text(json.dumps(...))` being atomic. `Path.write_text` is NOT atomic on most filesystems (opens file, writes, closes) — a concurrent read could see truncated JSON. Should switch to temp-file + rename pattern.
- M71: `--reset` should acquire the lock too — if a `--next` is running and user hits `--reset`, state.json gets deleted mid-convert. Currently no protection.

**Next iteration should pick:** M70 (state.json read-during-write race) would be the natural third-leg of the Ralph reliability theme (iter 8 M54 recovery + iter 12 M55 lock + iter 13 M70 atomic state-write). Alternatively rotate OUT of Ralph-harness work since we've done iter 8 + iter 12 both there — pick Cat B DiagnosticsBundle anonymization (inert setting M62 remainder), or Cat A M42-M45 remaining publish items, or audit.py (765 lines, still untraced).

---

## 2026-04-19 iteration 13 — M70 atomic state-write + M71 reset respects lock

**Angle:** Third leg of the Ralph reliability trilogy. iter 8 recovered interrupted state (M54), iter 12 prevented concurrent --next races (M55), iter 13 closes the remaining two state-integrity gaps in one batch.

**Deep trace walkthrough:**
1. Re-read `save_state` and `load_state` with a concurrent-reader-on-`--status` scenario in mind:
   - `save_state` calls `STATE_PATH.write_text(json.dumps(state, indent=2))`. Under the hood: `open(O_TRUNC | O_CREAT | O_WRONLY)`, `write(bytes)`, `close()`. Between the O_TRUNC and the write, the file is EMPTY. Between partial write and close, the file contains HALF the JSON.
   - `load_state` calls `STATE_PATH.read_text()` then `json.loads(raw)`. A reader hitting between open-O_TRUNC and close would read `""` or `"{\n  \"combos\": {\n    \"a__b\":"` and raise `JSONDecodeError`.
   - `cmd_status` doesn't take the Ralph lock (reads should be fast and free). So running `--status` while `--next` is saving is a legitimate scenario that can crash the status display.
2. **BUG M70 CONFIRMED.** Root cause: `Path.write_text` has no atomicity guarantee. Fix: POSIX `rename(2)` is atomic within one filesystem. Write to a sibling tmp file, then `os.rename(tmp, target)` — readers see only the old complete JSON or the new complete JSON, never torn.
3. **Defence in depth:** Even with atomic rename, make `load_state` resilient to `JSONDecodeError`. Scenarios that could still break an atomic rename: disk corruption, user editing state.json by hand, cross-FS move if someone relocated `results/`, kill -9 during fsync on HFS+. Instead of crashing, return empty state + stderr warning.
4. **M71 trace.** Scenario: terminal 1 is running `--next` (20-minute convert in progress). User in terminal 2 runs `--reset` to experiment with a different tier. `cmd_reset` calls `STATE_PATH.unlink()`. Terminal 1's next `save_state(info)` creates a FRESH state.json with only this combo's record — the rest of the matrix is lost. If the user also ran `--tier 1` to re-populate, they'd get duplicate records or missing combos depending on timing.
5. **BUG M71 CONFIRMED.** Reset must respect the same lock as --next. If held, block with the `{pid, host, started_at}` holder info (same message format as cmd_next's BLOCKED). Consistency = one concept of "who owns the runner right now".
6. **Python default-args gotcha caught during test writing:** `acquire_lock(path: Path = LOCK_PATH)` captures LOCK_PATH at DEFINITION TIME. Monkeypatching `runner.LOCK_PATH` in tests doesn't affect the captured default. Fix: call sites in `cmd_next` and `cmd_reset` now pass `LOCK_PATH` explicitly. Python evaluates module-global lookups at call time, so monkeypatching the module attribute correctly flows through. This is a common Python footgun worth pinning with a test (implicit via `test_cmd_reset_refuses_when_lock_held`).
7. **Bonus: tmp-shrapnel cleanup.** `save_state` cleans up its tmp file on any rename-path OSError. `cmd_reset` additionally glob-removes any stranded `state.json.tmp.*` from past crashed saves — a nice self-healing behaviour for users who hit an "OSError rename failed" and don't know what to do.

**Items touched:**
- M70 [x] — atomic tmp-rename + JSONDecodeError-tolerant load
- M71 [x] — reset now acquires the lock + clears tmp shrapnel on success

**Tests (6 new, 4 M70 + 2 M71):**
- `test_save_state_is_atomic_via_tmp_rename`: monkeypatches os.rename to inspect src/dst; asserts src is `state.json.tmp.*` in the same directory (so rename stays atomic within one FS).
- `test_save_state_cleans_tmp_on_failure`: simulates an OSError on rename; asserts no stale tmp files remain after the save_state call raises.
- `test_load_state_tolerates_corrupt_json`: writes garbage; asserts load_state returns empty state + warning on stderr (no crash).
- `test_save_load_roundtrip_survives_many_writes`: 50-round smoke.
- `test_cmd_reset_refuses_when_lock_held`: holds the lock with alive PID, asserts state.json survives + BLOCKED message.
- `test_cmd_reset_clears_tmp_shrapnel`: plants leftover tmp files, asserts successful reset clears them.

**Commit:** (this iteration)

**Verification:** 44 ralph_runner tests pass (was 38). Python jang-tools + Swift unchanged at 231 + 96.

**Closed-status tally:** 21 (prior) + M70 + M71 = 23 closed / 66 total = 35% closure rate.

**Ralph-reliability trilogy complete:** M54 (iter 8) + M55 (iter 12) + M70 + M71 (iter 13) cover recovery, concurrency, and atomicity. The remaining M67/M68/M69 items (cross-host PID reuse, SIGKILL+PID-reuse, network FS) are genuinely pathological or documentation.

**Next iteration should pick:** Rotate OUT of Ralph-harness work — we've spent 3 of the last 6 iterations there. Candidates in priority order:
1. **audit.py (765 lines, never traced, Cat K debt since iter 8).** The most impactful untouched surface — Ralph's audit harness is what judges whether a convert passed, and we've never looked at how it does that.
2. **Cat A remaining publish items (M42 verify cancellation, M43 publish progress streaming, M45 modelcard per-arch coverage).** Adopter journey is 80% done.
3. **Cat E spawned-M easy picks:** M02 (data-shaped-but-wrong-content HF clone), M07 (non-standard nested model_type), M15 (publish token clear-on-complete), M22 (DiagnosticsBundle race during running convert).

---

## 2026-04-19 iteration 14 — DiagnosticsBundle audit (Cat E / M22 + M16)

**Angle:** Rotating out of Ralph-harness work after 3 of the last 6 iterations there. Picked M22 (Cat E spawned-M) because DiagnosticsBundle is the surface where bug reports get generated — and it's where the LATEST secrets can leak into a public GitHub Issue zip. It's a low-LOC surface (40 lines pre-iter-14) with a high blast radius.

**Deep trace walkthrough:**
1. Read `DiagnosticsBundle.swift` (40 lines). MainActor, synchronous, zips a tempdir via `/usr/bin/ditto`. Single call site in `RunStep.swift` on the failed-convert branch.
2. **M22 original question — race on @State array?** Tracing: `logs` and `events` are `@State private var logs: [String]`. All mutations go through `logs.append(...)` on MainActor. `DiagnosticsBundle.write(logLines: logs, ...)` passes a Swift Array by value — value semantics, immediate snapshot. No race. **Original M22 is a non-issue.**
3. BUT the trace surfaced three actual bugs:
4. **BUG M22d — collision on back-to-back clicks.** `ISO8601DateFormatter()` without `.withFractionalSeconds` emits second-precision timestamps. Two clicks within the same wall-clock second land in the same `workDir`. `createDirectory(withIntermediateDirectories: true)` doesn't fail if the dir exists, so the second write REUSES the directory — stale files from the first click get zipped into the second bundle (if the first bundle hadn't completed the `removeItem` cleanup yet, which is mid-function).
5. **BUG M22e / M16 — no token scrubbing.** The zip includes raw `logLines` and `eventLines`. If ANY of those strings contains `hf_abc...` (because an HTTP client logged the Authorization header, or the user pasted a token into an input field that ended up in the log, or a future huggingface_hub exception embedded it), the token lands in a bug-report zip. iter 6 fixed the publish-error call site specifically; the diagnostics layer never scrubbed.
6. **BUG M62-anonymize — inert setting.** `AppSettings.anonymizePathsInDiagnostics` was declared with UI wiring but never consulted. `plan.sourceURL` and `plan.outputURL` went into plan.json verbatim — bug report reveals the user's filesystem layout (e.g., `/Volumes/WorkDisk/secrets-dir/MyModel`).
7. **Fix:**
   - **M22d:** `ISO8601DateFormatter` with `[.withInternetDateTime, .withFractionalSeconds]` → millisecond precision in the stamp. Each click gets a unique workDir AND a unique zip filename.
   - **M22e/M16:** `DiagnosticsBundle.scrubSensitive(_:)` runs 4 regex patterns before `String.write(to:)`:
     - `hf_[A-Za-z0-9_-]{20,}` — HF format tokens
     - `huggingface_[A-Za-z0-9_-]{20,}` — legacy format
     - Case-insensitive `Authorization: Bearer …`
     - Case-insensitive generic `Bearer …`
     All require ≥20-char suffix so a variable like `hf_short` in a log line doesn't trigger. Replacement is `<redacted>`.
   - **M62-anonymize:** `write` gained `anonymizePaths: Bool = false` parameter. Plan is serialised via an explicit dict construction (not Codable) so we can selectively rewrite `sourceURL` / `outputURL` to basenames. RunStep call site passes `settings.anonymizePathsInDiagnostics` through.
8. **Design decision — scrub at write-time, not post-zip:** Post-zip scrubbing would require unzipping, rewriting, re-zipping. Too expensive and error-prone. Scrubbing the in-memory strings before they hit disk is strictly cheaper AND it means the tempdir on disk never contains the secret (matters if the user cancels/crashes mid-zip).
9. **Design decision — `≥20` char min:** Real HF tokens are ~40 chars. 20 is comfortably below real-token length but above incidental occurrences. Validated with negative test `test_scrub_short_hf_lookalike_not_redacted`.
10. **Testing:** 10 new tests covering scrub for each pattern (4 positive + 1 negative non-match + 1 preserves-normal-text), end-to-end zip-then-unzip assert-no-secret, anonymize-on / anonymize-off plan.json roundtrip with JSON parsing (NOT substring matching, because JSONSerialization escapes `/` as `\/` in raw bytes — bug caught during test writing), millisecond uniqueness for back-to-back writes.

**Items touched:**
- M22 [x] — original non-issue confirmed; 3 adjacent real bugs found + fixed
- M16 [x] — token scrubbing at the diagnostics boundary
- M62-anonymize [x] — one more inert setting wired; M62 parent now 10/12 done

**Commit:** (this iteration)

**Verification:** 106 Swift tests (was 96), 231 jang-tools Python, 44 ralph_runner Python — all pass. Total Python 275 across both suites.

**Closed-status tally:** 23 (prior) + M16 + M22 + M62-anon = 26 closed / 66 total = 39% closure rate.

**Next iteration should pick:** **audit.py (Cat K debt since iter 8, 765 lines, never traced)** is now the clear priority — 6 iters of debt and it's the harness that judges whether a convert passed. Alternatively, Cat A remaining publish items (M42 verify cancellation, M43 publish progress streaming, M45 modelcard per-arch coverage) which would round out the adopter journey begun in iters 6+7. Cat D (memory cross-ref) also hasn't been revisited since iter 5 — could scan for additional drift now that 10 more iters' worth of code has landed.

---

## 2026-04-19 iteration 15 — audit.py deep trace (Cat K debt closed)

**Angle:** 7 iterations of Cat K debt. audit.py is what decides whether Ralph
grades a convert GREEN or FAILED; bugs here silently mis-grade real runs.
Read it end-to-end + look for: (a) unreachable / dead-code rows, (b) out-of-sync
logic between audit.py and Swift PostConvertVerifier (the two arbiters must agree),
(c) pass/fail semantics mismatches on required rows.

**Deep trace walkthrough:**
1. Read `AUDIT_REGISTRY` (lines 682-697). 13 rows defined. Numbered sequence
   has GAPS at a10, a13, a14 (deprecated — defensible) but ALSO at a6 which
   has a defined function AND a special-case dispatch in run_audits.
2. Read `run_audits` (lines 700-731). First check: `row not in AUDIT_REGISTRY: continue`. Then dispatch: `if row == "a6": r = audit_a6_wall_time(...)`. **But a6 is NOT in the registry.** So the continue fires FIRST — the `row == "a6"` branch is unreachable. Users passing `--rows a6` get `{"status": "n/a", "hint": "unknown row a6"}` on a real working function.
3. **BUG M72 CONFIRMED.** `audit_a6_wall_time` is dead code via this path. Fix: register it.
4. Read `audit_a2_chat_template` (lines 112-133). Line 117-121: guard clause checks `tok.chat_template` (inline form) and `chat_template.jinja` file. If neither exists, return `n/a`.
5. Cross-check with Swift `PostConvertVerifier.swift:42-50` — it accepts THREE forms: inline, `chat_template.jinja`, `chat_template.json`. The two arbiters are OUT OF SYNC. A model shipping only `chat_template.json` would be graded `n/a` by audit.py but valid by Swift.
6. **BUG M77 CONFIRMED.** Qwen3-VL and other newer HF models use `chat_template.json`. Fix: add the third form to a2's guard-clause file check.
7. Scan remaining rows quickly:
   - a1 tokenizer roundtrip — looks solid, uses `decoded.strip() != s.strip()` looser equality to tolerate whitespace.
   - a9 special tokens — potential issue with structured vs string token value comparison (source `{"bos_token":{"content":"<s>",...}}` vs output `{"bos_token":"<s>"}`) — real HF can serialise both forms. Noted but out of scope for this iter.
   - a15 inference — calls `mlx_lm.generate` directly with raw prompt (no chat_template), so this is looser than the CLI which uses `_apply_chat_template_if_any` (iter 4 fix). That's a deliberate design split: a15 = "does ANY output come out", a16 = "does the chat template actually render". OK.
   - a17 model card — calls `jang-tools modelcard --json` subprocess, decodes JSON, validates required keys. Solid.
   - a18 usage examples — 4 langs, Python snippet must compile. Covers the "does the example we tell users to copy-paste actually run". Nice design.
8. **Fix scope (kept narrow for review):** M72 + M77 only. Other surface-level concerns spawned as M78-M82.

**Design notes:**
- M72 fix is one line + 5 tests. Defensible for inclusion.
- M77 fix is 3 lines + 3 tests. Defensible.
- Testing strategy: stub `load_tokenizer` via `monkeypatch.setattr` because building a valid HF tokenizer.json in a unit-test fixture is overkill for this pure-logic change. The tests pin the FILE-DISCOVERY logic specifically, not the full end-to-end render.

**Items touched:**
- M72 [x] — a6 now registered + dispatches correctly; 5 regression tests
- M77 [x] — a2 now matches Swift's three-form chat-template acceptance; 3 tests

**New questions spawned (M78-M82):**
- M78: a9 structured-vs-string special token value comparison would false-fail on legitimately equivalent forms. Need a value-normaliser.
- M79: a15 uses raw prompt to mlx_lm.generate — explicit design split from a16, but ought to be documented in a module docstring so the next auditor doesn't "fix" it into applying the chat template.
- M80: audit_a4 (tokens/sec) and audit_a5 (chat turn) are in the registry but not required. What does "tokens/sec below threshold" fail mean — speed regression, or just a slow machine? No baseline comparison today. Could leverage a6's baseline infrastructure once it's registered.
- M81: a11/a12 fixtures are in `ralph_runner/fixtures/`. Are they git-tracked? If not, a fresh clone can't run those audits.
- M82: run_audits has no per-row timeout. audit_a15 could hang indefinitely on a deadlocked mlx_lm load. Should wrap each row in a timeout context (e.g., 5 minutes default, configurable per row).

**Commit:** (this iteration)

**Verification:** 52 ralph_runner tests (was 44), 231 jang-tools + 106 Swift unchanged. Total 389 tests.

**Closed-status tally:** 26 (prior) + M72 + M77 = 28 closed / 71 total = 39% closure rate. Cat K debt now drained: runner.py (iters 8 + 12 + 13) AND audit.py (iter 15) both traced.

**Next iteration should pick:** Cat A remaining publish items (M42 verify cancellation, M43 publish progress streaming, M45 modelcard per-arch coverage) would round out the adopter journey and complement iter 14's DiagnosticsBundle work. Or tackle one of M78-M82 spawned here (M78 a9 value normaliser is the most concrete). Or Cat D memory cross-ref second pass (never revisited since iter 5).

---

## 2026-04-19 iteration 16 — M78 a9 value normalization (Cat K continued)

**Angle:** Close the highest-impact spawned-M from iter 15. M78 was the only
latent FALSE-FAIL on a REQUIRED audit row: a9 (special tokens preservation,
required=True) used raw `!=` comparison on values that HF legitimately
serialises in two equivalent forms. Every convert where source and output
crossed the form boundary would be incorrectly graded FAILED. Ralph's
green/fail matrix has been lying.

**Deep trace walkthrough:**
1. Read `special_tokens_map.json` from a real Qwen3 source: structured dict form with `{"content", "lstrip", "normalized", "rstrip", "single_word"}`.
2. Read the same file from a converted JANG output: plain-string form `{"bos_token": "<s>"}`. Both forms are accepted by `AutoTokenizer.from_pretrained` — they're semantically identical.
3. Traced the old `audit.py:377 elif out_tokens[k] != v`. Python dict vs string → always `!=`. All structured-form sources would report as mismatches. **If this had ever fired on a production combo, the combo was wrongly graded FAILED.**
4. **BUG M78 CONFIRMED.** The severity depends on whether `required=True` — which it IS on a9. So the bug actively propagates into the green/fail matrix.
5. **Fix architecture:** extract the `content` string from either form before comparing. Keep strict `==` as a FALLBACK for shapes neither normalisation form recognises — don't silently pass on corrupted files or future schema drift.
6. **Defence-in-depth considerations:**
   - Unrecognised shape returns None from the normaliser → strict fallback.
   - Fallback failures get reported in a new `unnormalizable` field so the debug output distinguishes "structurally different" from "normaliser didn't understand".
   - Positive-AND-negative test coverage: pin that the fix doesn't accidentally make a9 always-pass. Explicit test with two different `content` strings ensures genuine mismatches still fail.
7. **Verification walk:** what OTHER audit rows might have similar issues?
   - a8 (parser fields): uses `src_val != out_val` on primitive types (bool, str, dict). No known cross-serialisation equivalences. Defer.
   - a17 modelcard: validates the presence of keys only, not value equality. No issue.
   - Others do functional checks (roundtrip, generate, render) — no direct equality comparisons on HF file contents.

**Items touched:**
- M78 [x] — a9 now accepts the two equivalent forms; 8 new tests pin both directions + negative + fallback + unrecognised-shape behaviour.

**Commit:** (this iteration)

**Verification:** 60 ralph_runner tests (was 52). jang-tools + Swift unchanged.

**Closed-status tally:** 28 (prior) + M78 = 29 closed / 71 total = 41% closure rate.

**Next iteration should pick:** Cat A publish remainder (M42/M43/M45) has been on the forecast list since iter 14 — 3 iters of debt now. M43 (publish progress streaming) would materially improve the adopter experience. Alternatively M81 (fixture git-tracking audit) is a concrete 10-minute check. Cat D memory cross-ref hasn't been revisited since iter 5 — 11 iters of drift possible.

---

## 2026-04-19 iteration 17 — M81 fixture invariants + M15 publish token clear

**Angle:** Pair two small items in one iter. M81 is "latent process-gap that could silently degrade a11/a12 on a fresh clone"; M15 is a 15-iter-old security-hygiene TODO (token lingering in memory after publish). Both small enough to batch, both closing audit-checklist items with concrete tests.

**M81 deep trace:**
1. `git ls-files ralph_runner/fixtures/` — both files tracked.
2. `git check-ignore -v ralph_runner/fixtures/*` — returns nothing, so no gitignore rules target them.
3. **No active bug today.** But the invariant wasn't pinned — a future global `.gitignore` rule (e.g., adding `*.npy` for some unrelated reason) could silently un-track `test_video_frames.npy`, and a11/a12 would silently degrade to warn on every fresh-clone run with no visible CI failure.
4. Close M81 by pinning the invariant with 4 regression tests:
   - `_FIXTURE_DIR` resolves and is a directory.
   - test_image.png: exists, PIL opens, dimensions ≥32×32.
   - test_video_frames.npy: exists, numpy loads, shape=4D, ≥4 frames, ≥16×16, 3 channels.
   - `git ls-files` explicitly lists both files — catches the "accidentally gitignored" failure mode that existence-on-disk tests can't, because the author's working tree always has the files.
5. This pattern is a good template for other "implicit dependency on tracked non-code files" — the `subprocess.run(["git", "ls-files"])` check uniquely catches untracked-on-CI / tracked-locally drift.

**M15 deep trace:**
1. Read `PublishToHuggingFaceSheet.swift`. Token lives in `@State private var token: String`. Sheet init populates it from `HF_HUB_TOKEN` env (iter 9 behaviour).
2. After a successful publish, `token` stays in memory until sheet dismiss. User may leave sheet open for minutes / forget to close. Passerby at the Mac could see the value in the SecureField buffer — macOS SecureField hides characters visually but the String itself is just a Swift.String in memory.
3. Also: crash dumps sent via "Copy Diagnostics" could pick up app memory snapshot. Iter 14's scrubSensitive catches tokens in STRINGS but not in in-memory state snapshots. Defense in depth says: don't hold the secret longer than needed.
4. **Fix:** on successful publish, `token = ""` in runPublish(). On failure, KEEP the token — the user usually wants to retry and re-entering a 40-char token after every transient HTTP hiccup is worse UX than a ~30-second exposure window.
5. Dry-run path NOT cleared because the user almost always follows dryrun with a real publish — clearing after dryrun would force re-entry for no security benefit.

**Items touched:**
- M81 [x] — 4 regression tests pin the fixture invariants; 0 functional change but catches a latent failure mode.
- M15 [x] — token wiped after successful publish; failure path preserves for retry.

**Commit:** (this iteration)

**Verification:** 64 ralph_runner tests (was 60), 106 Swift (unchanged test count but tests still pass after behaviour change), 231 Python jang-tools unchanged. Total 401 tests across all suites.

**Closed-status tally:** 29 (prior) + M15 + M81 = 31 closed / 71 total = 44% closure rate.

**Next iteration should pick:** Cat A publish remainder (M42/M43/M45) — 4 iters on the forecast list, STILL unpicked. M43 (publish progress streaming — no ETA for 30-min uploads) would be the highest adopter-UX win. Alternatively Cat D memory cross-ref second pass (idle 12 iters). Alternatively M82 (run_audits per-row timeout — a15 hang bomb) is a concrete reliability win.

---

## 2026-04-19 iteration 18 — M82 per-row timeout in run_audits (Cat K)

**Angle:** Close M82 — the concrete reliability win called out at end of iter 17. Ralph's audit harness has 15 registered rows, two of which (`a15` inference, `a11/a12` VL/video preprocessors) invoke `mlx_lm.load` / `AutoProcessor.from_pretrained` — both have known hang modes on corrupted bundles or Metal deadlocks. With no timeout, one hung row = entire Ralph iteration stalled. This was the reliability counterpart to iter 13's Ralph-reliability trilogy.

**Deep trace walkthrough:**
1. Read `run_audits` (iter-15 shape): each row runs in a try/except on the main thread. If `audit_a15_inference` hangs at `_load_llm(model_dir)` (a real failure mode on a corrupted shard), the except never fires — the thread just blocks inside Metal.
2. `cmd_next` in runner.py wraps `run_audits` — also synchronous — so the Ralph --next call never returns. User in the terminal sees no output for hours. Matrix stays stuck. Iter 12's lock guarantees only ONE instance hangs at a time; iter 13's state atomicity keeps the file intact; neither fix addresses an actively-hung audit.
3. **Solution options considered:**
   - `signal.alarm`: Unix-only (macOS OK). Fires SIGALRM which interrupts the call. BUT: doesn't work from non-main threads, and mlx/Metal may mask signals. Also clobbers other signal handlers. Rejected.
   - `multiprocessing`: killable, but adds serialization overhead for the model_dir Path, dict return, registry references; spawning a process per row triples audit wall-clock on short rows; macstudio memory pressure during convert+audit is already high. Rejected.
   - `ThreadPoolExecutor` with `Future.result(timeout=...)`: timeout returns to caller but hung thread continues running. Acceptable if we accept the zombie thread dies when process exits (we do — cmd_next exits after audit).
4. **Critical gotcha caught during test writing:** `with concurrent.futures.ThreadPoolExecutor(...) as pool:` context-manager `__exit__` implicitly calls `shutdown(wait=True)` → blocks forever on the hung thread, DEFEATING the timeout. Must use explicit `pool.shutdown(wait=False)` on TimeoutError. Test `test_run_with_timeout_fires_on_hang` initially timed out (30s+) until this was fixed; now completes in <3s of the intended 1s timeout.
5. **Timeout tuning considerations:**
   - a15: big JANG_4K model load (200 GB on macstudio M3 Ultra) takes ~2-4 min legitimately. 600s gives 3× headroom for corner cases.
   - a17/a18: shell out to jang-tools subprocess. Those commands themselves have 60s internal timeouts; 90s gives a buffer for process startup.
   - a3/a4/a5: mlx_lm.generate with ~20-200 tokens. 300s covers slow tokens/sec.
   - a11/a12: VL AutoProcessor load. 120s.
   - Default 60s: fast file-inspection rows (a1/a2/a7/a8/a9/a16).
6. **Secondary fix caught during test writing:** `_fail(hint: str, **k)` signature makes `hint` positional. My first draft of the timeout message passed `hint=` kwarg → `got multiple values for argument 'hint'`. Rolled the "hung — process cleanup will reclaim" note into the single hint string.

**Items touched:**
- M82 [x] — per-row timeouts + graceful hang recovery; subsequent rows still run after a hang.

**Tests (4 new):** quick-result roundtrip; timeout fires within tolerance (1s timeout of a 10s hanger completes in <3s); timeout map pins a15≥300 + default=60 (regression against future accidental dropping); end-to-end `run_audits` fake-registry test with a hang-row + quick-row proves subsequent rows still run + overall status reflects only required fails.

**Commit:** (this iteration)

**Verification:** 68 ralph_runner (was 64). jang-tools 231 + Swift 106 unchanged. Total 405.

**Closed-status tally:** 31 (prior) + M82 = 32 closed / 71 total = 45% closure rate.

**Next iteration should pick:** Cat A publish remainder (M42/M43/M45) — NOW 5 iters on the forecast list, STILL unpicked. M43 progress streaming is the biggest adopter-UX win. Alternatively M42 (verify cancellation — PostConvertVerifier's `jang validate` subprocess can't be cancelled). Alternatively Cat D memory cross-ref second pass (idle 13 iters) would surface any recent drift between memory claims and actual code.

---

## 2026-04-19 iteration 19 — M42 verify cancellation + timeout (Cat A finally picked)

**Angle:** Close M42 from the Cat A publish/verify adopter journey, on the forecast list for 5 iters. Applies the same actor-friendly subprocess pattern that's proven across PythonRunner (iter 3), InferenceRunner (iter 3), and audit.py (iter 18) to the last remaining synchronous `waitUntilExit` call in the Swift subprocess surface.

**Deep trace walkthrough:**
1. `PostConvertVerifier.runJangValidate` was a 6-line function using `proc.waitUntilExit()` directly. No timeout, no cancellation.
2. Called from `PostConvertVerifier.run` (@MainActor) via `await Self.runJangValidate(outputDir: out)`. `async` + `nonisolated static` means it runs off MainActor on a cooperative thread — so it didn't block the main thread, but it DID hold a cooperative thread until the subprocess exited.
3. **Worst case:** `jang validate` hangs (heavy import stall, corrupted JSON parser loop, locking issue). The entire VerifyStep never completes. User navigates away → view dismounts → but the `Task` spawned by `.onAppear { Task { ... } }` is DETACHED from view lifecycle → keeps running. Python subprocess keeps running. Resource leak.
4. **Two independent bugs combined into one hang scenario:**
   - BUG-1: `runJangValidate` has no timeout. A hung subprocess makes `refresh()` block forever.
   - BUG-2: `VerifyStep.onAppear { Task { ... } }` isn't tied to view lifecycle. SwiftUI dismount doesn't cancel the task.
5. **Fix layer 1 (subprocess):** Apply the M19 / iter-3 pattern. `withCheckedContinuation` + `proc.terminationHandler` races against `Task.sleep(timeoutSeconds)`. First winner resolves. DispatchQueue guard against double-resume (CheckedContinuation fatal-errors on double-resume). On timeout: SIGTERM + 3s SIGKILL escalation.
6. **Fix layer 2 (view):** `.onAppear { Task { ... } }` → `.task { ... }`. SwiftUI's `.task` modifier IS tied to view lifecycle — auto-cancelled on dismount, which flows through Task.isCancelled checks (future refresh() work can observe this).
7. **Timeout rationale:** normal `jang validate` completes in ≤5s (file inspection only — no model load, no inference). 60s default = 10× headroom. Pin with a test that the default stays in [30, 300] to prevent regressions in either direction.
8. **Timeout test strategy:** Can't easily mock a hanging jang validate from Swift tests without subprocess override infrastructure. Instead: pass `timeoutSeconds: 0.1` against a real subprocess; even Python startup loses that race. Assert `elapsed < 10s` (NOT waiting full 60s) and result is false.
9. **Caught during implementation:** `Task.sleep(for: .seconds(Double))` accepts Double timeouts; I initially worried 0.1 would be rounded to 0 but `.seconds(0.1)` is `100ms` as expected.

**Items touched:**
- M42 [x] — validate can no longer hang VerifyStep; view task lifecycle cleaned up.

**Tests (3 new):** default timeout ∈ [30, 300]s, runJangValidate returns false on bogus path (exercises terminationHandler branch), 0.1s timeout bounds wall time under 10s.

**Commit:** (this iteration)

**Verification:** 109 Swift (was 106). Python unchanged at 299 across jang-tools + ralph_runner.

**Closed-status tally:** 32 (prior) + M42 = 33 closed / 71 total = 46% closure rate.

**Next iteration should pick:** M43 (publish progress streaming) is the HIGHEST REMAINING ADOPTER-UX ITEM — users stare at a spinner for 30-minute uploads with zero feedback. Architecturally meatier (needs JSONL stream from Python + Swift parser) but high value. Alternatively M45 (modelcard per-arch coverage) is mid-size. Cat D memory cross-ref is 14 iters idle and would surface any drift between memory and the 10,000+ lines of code that landed since iter 5.

---

## 2026-04-20 iteration 20 — M45 modelcard per-arch coverage surfaces import bugs

**Angle:** Close M45 from the Cat A adopter journey. M43 (progress streaming) architecturally meatier, deferred. M45 scope was "verify every supported family generates a working snippet" — which turned into "find out why the generated snippets never actually worked".

**Deep trace walkthrough:**
1. Read `modelcard.py`. `generate_card` calls `detect_capabilities` + `render_snippet(model_dir, "python")` to embed a Python example.
2. Read `python-snippet.py.jinja`. Two branches: VL and non-VL.
3. Non-VL line 18-21: `from jang_tools.loader import load_model\nmodel, tokenizer = load_model("{{ model_path }}")`. Checked actual symbols in `jang_tools.loader`: only `load_jang_model` and `load_jang_vlm_model` exist. **`load_model` does not exist in that module.** Every adopter following the snippet would see `ImportError: cannot import name 'load_model' from 'jang_tools.loader'`.
4. Ran `python3 -c "from jang_tools.loader import load_model"` — confirmed ImportError.
5. Read `test_examples.py::test_render_python_dense`. Line 69: `assert "from jang_tools.loader import load_model" in snippet`. **The test PINNED the wrong symbol.** Same for `test_cli_examples_json` line 111 and `test_modelcard.py::test_card_has_usage_section` line 47 (which asserts the substring `"load_model"` which is vacuously true because it appears inside `load_jang_model` — classic substring false-positive).
6. Looked at VL branch. Line 2: `from jang_tools.load_jangtq_vlm import load_jangtq_vlm`. Actual symbol: `load_jangtq_vlm_model`. Same class of bug.
7. Grepped for other consumers of the wrong VL symbol. Found `inference.py:41`: `from jang_tools.load_jangtq_vlm import load_jangtq_vlm`. This wasn't caught because the surrounding `try/except Exception: pass` silently swallowed the ImportError and fell through to `mlx_vlm.load` — which works on non-JANGTQ VL models but CANNOT load real JANGTQ-VL outputs correctly. **Silent degradation of VL inference** because the fallback looks like success.
8. Checked audit.py — correctly uses `load_jangtq_vlm_model` (fixed in a prior commit per memory).
9. **Why the tests didn't catch it:** `test_cli_python_snippet_compiles` uses `compile(snippet, ..., "exec")` which only validates syntax. Compile-only can't catch import-name typos — you'd need to actually import and call. `test_examples.py::test_render_python_dense` asserted the WRONG name, locking the bug in.

**Fix stack:**
- `python-snippet.py.jinja`: dense import `load_jang_model`; VL import `load_jangtq_vlm_model`. Call sites updated.
- `inference.py`: VL dispatch uses `load_jangtq_vlm_model`. Comment left explaining the silent-fallback trap the old name was hiding.
- `test_examples.py`: stale pins flipped. Added `load_model(` not-in-snippet guard to catch bare-call regressions. Added `test_python_snippet_imports_resolve_to_real_symbols` using `importlib.import_module` + `hasattr(mod, name)` — this IS what compile-only couldn't catch.
- `test_modelcard.py`: `"load_model" in card` was vacuously true (substring of `load_jang_model`). Flipped to full-symbol assert plus `load_model(` bare-call absence.

**Items touched:**
- M45 [x] — two import-name bugs fixed + test pins corrected to assert correct symbols + runtime-hasattr test added so future renames break loudly.

**Commit:** (this iteration)

**Verification:** 232 jang-tools (was 231, +1 hasattr test). ralph_runner 68 + Swift 109 unchanged.

**Closed-status tally:** 33 (prior) + M45 = 34 closed / 71 total = 48% closure rate. **Cat A adopter journey significantly drained.** Only M43 (publish progress streaming) remains in Cat A.

**Meta-lesson:** "tests pin the wrong behavior" is a failure mode distinct from "code is buggy". A test asserting `assert X in snippet` where X is the wrong value PREVENTS the bug from being fixed via normal test-driven changes. The fix must update BOTH the code AND the test. Watch for:
- Substring assertions where the substring appears inside the correct string (the `load_model` in `load_jang_model` case).
- `compile()` validation treated as runtime-correctness (only catches SyntaxError, not ImportError).
- `try/except Exception: pass` wrapping imports (silently converts ImportError to "feature unavailable").

**Next iteration should pick:** M43 (publish progress streaming) — last Cat A item, biggest remaining adopter-UX win, but architecturally non-trivial. Or Cat D memory cross-ref second pass (14 iters idle). Or M48 (default repoName missing org prefix — small UX polish). Or investigate whether `audit.py:272` ok-note `"VL model loaded successfully via load_jangtq_vlm/mlx_vlm.load"` mentions the old wrong name in user-visible messages (likely just a misleading string).
