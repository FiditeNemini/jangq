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

---

## 2026-04-20 iteration 21 — Cat D memory cross-ref (second pass, 16 iters since iter 5)

**Angle:** Iter 5 did memory cross-ref and found M35 (Osaurus remap drift). 16 iterations of code changes have landed since; whatever memory claims were fresh then may have drifted now. Cat D hadn't been revisited. Pick one concrete memory file per critical subsystem and grep its claims against the live code.

**Deep trace walkthrough:**
1. Started with `project_mlp_asymmetry.md` (11 days old). Memory claim: "Threshold lowered from 512 to 256 on 2026-04-08". Memory says GLM-5.1 (256 experts) degenerates without the floor → repetition loops after single-token coherence. Memory cites `allocate.py:_apply_mlp_asymmetry_floor()` as the implementation.
2. Read `allocate.py:_apply_mlp_asymmetry_floor`:
   - Docstring (line 320): "Apply MLP asymmetry bit floors for **256+** expert models"
   - Code guard (line 325): `if num_experts < **512**: return bits`
   - **TRIPLICATE DRIFT:** memory says 256, docstring says 256+, code checks <512. Code wins at runtime. Every 256-expert model (GLM-5.1, MiniMax M2.7, Qwen3.6) slipped through the guard without getting the floor.
3. **Severity assessment:** Per memory, lack of floor causes single-token coherence but multi-token repetition loops. This is an OUTPUT QUALITY bug affecting every JANG_2*/1L convert of a 256-expert source model. Users could have been getting broken models for unknown weeks without noticing because the audit harness's a3 ("generation coherence, does output contain 'Paris'") only checks ONE token.
4. **Budget impact of the fix:** Per memory, "Budget impact for 2-bit profiles: (4+2+3)/3 = 3.0 avg for expert MLPs. Raises overall average from ~2.15 to ~3.13 bits for GLM-5.1 JANG_1L." → ~50% larger model on disk and in RAM. Higher memory pressure but correct output.
5. **Fix design:** Don't just flip the number. Extract a named constant `_MLP_ASYMMETRY_MIN_EXPERTS = 256` so future drifts fight a named constant + a regression test, not a magic `512` three lines below its own "256+" docstring. Memory file updated with an iter-21 fix note.
6. **Test design:** 9 tests cover the full behavior matrix: threshold matches memory (pins future drift), floor values pin to 4/3/no-up_proj, below-threshold passthrough, at-256 floor applies, at-512 still applies, shared_expert exempt, non-MLP names pass through, floor never lowers bits (max semantics), Mixtral w1/w2 naming also covered.
7. **Adjacent drift spotted:** `capabilities_cli.py:_KNOWN_512_EXPERT_TYPES = ["minimax_m2", "glm_moe_dsa"]` — list name says 512 but per `project_bfloat16_fix.md` MiniMax has 256 experts + hidden≤3072 and is NOT affected by the 512-hidden=4096 bfloat16 overflow. Swift PreflightRunner uses this list for warnings → MiniMax users get misleading "512+ experts" warning. Spawned as M84.
8. **Adjacent drift spotted:** allocate.py has ~6-8 comments still saying "512+ expert models" in prose while the code is (now) correctly 256+. Sweep-comment task spawned as M85.

**Items touched:**
- M83 [x] — primary drift fixed: `< 512` → `< _MLP_ASYMMETRY_MIN_EXPERTS`; 9 regression tests pin behaviour.

**Commit:** (this iteration)

**Verification:** 241 jang-tools (was 232). ralph_runner 68 + Swift 109 unchanged.

**Closed-status tally:** 34 (prior) + M83 = 35 closed / 73 total = 48% closure rate. **Cat D revisit useful** — spawned 2 new drift items (M84, M85).

**Meta-observation on memory cross-ref cadence:** 16 iters is probably too long between passes. A single drift fix (M35 at iter 5, M83 at iter 21) per long cycle suggests drift accumulates steadily. Future: revisit Cat D every 5-7 iters when code velocity is high. Catching one bit-allocation-quality bug per cycle is worth one iter's attention.

**Next iteration should pick:** M43 (publish progress streaming, last Cat A item — deferred across iters 14/17/18/19 now; architecturally non-trivial but highest remaining adopter-UX win). Or M85 (sweep the 512→256 prose comments — tiny, zero-risk, would avoid repeating this same investigation 10 iters from now). Or continue Cat D by picking another memory file (e.g., `project_mistral4_architecture.md` claims 6 fixes for MLA+MoE — verify each is still in place).

---

## 2026-04-20 iteration 22 — Cat D cross-ref Mistral 4 + M85 prose sweep

**Angle:** Apply iter-21's meta-observation ("catch one drift per 5-7 iter cycle") by doing another Cat D pass on a different memory file, plus the trivial M85 prose cleanup to prevent the same investigation from recurring. Two small checks batched — efficient.

**Deep trace walkthrough (M86 — Mistral 4 architecture claims):**
1. Read `project_mistral4_architecture.md` (28 days old). Claims 7 fixes; MEMORY.md index says 6 — minor inconsistency, not actionable. Walk through each claim against `mistral4_mlx.py`:
2. Fix 1 (FP8 bf16 scale_inv loader) — handled in `fp8.py:_load_bf16_from_header`. Already tested. ✓
3. Fix 2 (`norm_topk_prob=True`) — `mistral4_mlx.py:58` config default True; used at line 382-383. ✓
4. Fix 3 (`llama_4_scaling_beta=0.1`) — line 76 rope_scaling merge, line 246 loaded into attention. ✓
5. Fix 4 (attention scale plain `1/sqrt(128)`, NO mscale²) — line 194: `self.scale = self.q_head_dim ** -0.5` = 1/√128 = 0.0884. Confirmed no mscale multiplied in. Line 9 top-of-file comment explicitly says "mscale == mscale_all_dim == 1.0 → no mscale on attention or rope". ✓
6. Fix 5 (`rope_interleave=True → traditional=True`) — **AMBIGUOUS.** `mistral4_mlx.py:54` config has `rope_interleave: bool = True` (matches HF). Line 151-158 calls `mx.fast.rope(traditional=False)` with comment "interleaved RoPE for Mistral 4". Memory says these should map True↔True. Two possibilities:
   a) MLX-native path has inverted `traditional` semantics vs mlx-vlm (where the memory fix was originally captured).
   b) Our native port has a RoPE bug that would show up as garbage output on a real Mistral 4 convert.
   Too risky to flip without live validation. Flagged as M87 for future verification.
7. Fix 6 (gate dequant uint32→bfloat16) — handled in loader.py:162-205 (verified iter 9). ✓
8. Fix 7 (auto bfloat16 via `model.set_dtype(mx.bfloat16)` for MLA) — loader.py:278-280: `elif _text_cfg.get("model_type") == "mistral4" or _text_cfg.get("kv_lora_rank", 0) > 0: model.set_dtype(mx.bfloat16)`. ✓
9. **6 of 7 memory-claimed fixes confirmed present; 1 needs live validation.** No drift like M83.

**Deep trace walkthrough (M85 — prose sweep):**
1. After iter 21's M83 fix, `allocate.py` had 8 "512+" prose comments — 6 about MLP asymmetry (wrong post-M83) and 2 about bfloat16 overflow (correct — different concern).
2. Separated the two concerns by reading each comment's context. Lines 62, 77 describe 397B NaN / float16 overflow — that threshold is genuinely 512 experts + hidden≥4096 per `project_bfloat16_fix.md` (loader.py:271 uses that exact check). Kept unchanged.
3. Lines 81, 87, 94, 246, 303, 529 describe MLP asymmetry — that's post-M83 at 256. Updated to "256+". Line 81 got a pointer to `_MLP_ASYMMETRY_MIN_EXPERTS` so future auditors can grep the symbol instead of a magic number.

**Items touched:**
- M85 [x] — 6 prose comments swept 512 → 256 in MLP-asymmetry context; bfloat16-context comments correctly left at 512.
- M86 [x] — cross-referenced 7 Mistral 4 fixes; 6/7 confirmed, 1 ambiguous (flagged M87).
- M87 [ ] — new spawn: needs live Mistral 4 validation of rope_traditional semantics.

**Commit:** (this iteration)

**Verification:** 241 jang-tools still pass (no code-path changes this iter, just prose + cross-ref confirmation). ralph_runner 68 + Swift 109 unchanged.

**Closed-status tally:** 35 (prior) + M85 + M86 = 37 closed / 74 total = 50% closure rate. **Half the audit matrix is closed.**

**Meta-observation on Cat D second pass:** iter 21 found a real quality bug (M83 threshold drift). iter 22 found mostly alignment + one ambiguous case (M87). Suggests that after one big drift catch, the marginal yield drops — but the second pass was still cheaper than a real bug to find via user report. Worth doing every 5-7 iters.

**Next iteration should pick:** M43 (publish progress streaming) is now 5 iters deferred — it's the LAST Cat A item and the biggest remaining adopter-UX gap. Architecturally meaty (needs JSONL stream from Python upload_folder + Swift parser) but highly user-visible. Alternatively M48 (default repoName missing org prefix — small polish). Or M87 live Mistral 4 RoPE validation (requires a real convert — can't be done in a unit test).

---

## 2026-04-20 iteration 23 — M43 Python-side progress streaming

**Angle:** After 6 iterations of deferral, finally tackled M43 — the biggest remaining adopter-UX gap. Scoped to the Python side only because the full end-to-end (Python JSONL + Swift stream + UI progress bar) is 2-iter sized work and breaking it in half keeps each commit reviewable. Iter 24 completes with the Swift half.

**Deep trace walkthrough:**
1. Read `publish.py` as it stood after iter 17. Single call: `api.upload_folder(folder_path=..., repo_id=..., token=..., commit_message=...)`. No streaming, no output. From Swift's perspective (`PublishService.swift:invoke`), the subprocess runs in a DispatchQueue + `waitUntilExit()` — stderr is captured only on FAILURE. On success, the 30-minute silence is literally silence.
2. **Solution architecture decision: per-file vs intercepted progress.**
   - Option A: hook into `upload_folder`'s tqdm output and parse it. Fragile; tqdm format changes upstream, progress bars use carriage-return lines which are hard to parse as JSONL.
   - Option B: use `huggingface_hub`'s `create_commit` with `CommitOperation*` primitives + custom progress callback. Possible but API is complex and different between huggingface_hub versions.
   - Option C: iterate files manually, call `HfApi.upload_file` per file, emit a JSONL event between files. Slower (no LFS batching, more commits) but dead-simple progress + reusable with the existing `ProgressEmitter` protocol from convert.
   - **Chose Option C.** Trade-off: for a 40-shard JANG_4K at 5 GB each, per-file serial upload is ~15% slower than bulk (no parallel sha256 hashing) but gives a tick-per-shard which is exactly what the UI needs. For the target user (publishing one model per day), 35min vs 30min is invisible; the progress bar is MASSIVELY visible.
3. **Implementation tightness:**
   - `upload_file` accepted as a callable injection so tests never touch HfApi / the network.
   - File enumeration is sorted, deterministic — gives consistent commit ordering.
   - Empty dir raises RuntimeError immediately (before any HF call) rather than creating an empty repo.
   - Schema reuses convert's 5-phase protocol EXACTLY (v=1, type=phase/tick/info/done, ts field) so Swift's existing `JSONLProgressParser` works unchanged in iter 24.
   - `--progress {none,json}` keeps the fast path (bulk upload_folder) as the default; opt-in `json` for UI integration.
4. **Commit-message details:**
   - Each file's commit message carries `(idx/total: relpath)` so HF's commit history reads like `Upload my-model-JANG_4K via jang-tools (7/42: model-00007-of-00042.safetensors)` instead of 42 identical messages. Useful for post-hoc debugging if an upload fails midway.
5. **Throttling:** `emitter.tick` has 100ms default throttle (overridable via iter-11's `JANG_TICK_THROTTLE_MS`). For 42 files @ 15-45s per 5GB file, ticks are plenty sparse to pass through the throttle unclogged.

**Items touched:**
- M43 [x] — Python side complete. Swift half (publishWithProgress + UI progress bar) is iter-24 work.

**Tests (4 new):**
- `test_upload_with_progress_iterates_every_file` — every file uploaded once, sorted order, commit messages carry progress.
- `test_upload_with_progress_emits_expected_jsonl_shape` — stream contains exactly 3 phase events + ≥1 tick + info, all with v=1 and ts (Swift parser invariants).
- `test_upload_with_progress_raises_on_empty_dir` — empty input is a hard error, not a silent no-op.
- `test_publish_cli_has_progress_flag` — pins the `--progress json` flag in the CLI help so a rename would break CI (once Swift depends on it).

**Commit:** (this iteration)

**Verification:** 245 jang-tools (was 241, +4). ralph_runner 68 + Swift 109 unchanged (Swift half next iter).

**Closed-status tally:** 37 (prior) + M43 = 38 closed / 74 total = 51% closure rate. **Last Cat A Python-side item done.** Swift-side integration work remains but is well-specified now.

**Design note for iter 24's Swift work:**
Swift already has `JSONLProgressParser` + the pattern from `PythonRunner.run() -> AsyncThrowingStream<ProgressEvent, Error>`. Copy that shape:
```swift
extension PublishService {
    static func publishWithProgress(modelPath: URL, repo: String, isPrivate: Bool,
                                    token: String) -> AsyncThrowingStream<ProgressEvent, Error>
}
```
Sheet wiring: `isPublishing: Bool` → `progress: (done: Int64, total: Int64, label: String)?`. Render `ProgressView(value:, total:)` + bytes-uploaded label. Total upload time unchanged; user perception vastly improved.

**Next iteration should pick:** M43 Swift half (pairs naturally with this iter, architecture pre-specified). OR a different rotation — M48 (default repoName polish), M87 (Mistral 4 live validation — requires real convert, probably not doable in a unit test iter), or next Cat D pass (following iter-21/22's 5-7 iter cadence — we're at iter-22 + 1 = cadence due around iter 27-29).

---

## 2026-04-20 iteration 24 — M43 Swift half: publishWithProgress stream + UI bar

**Angle:** Complete M43 end-to-end. Iter 23 shipped the Python JSONL emitter; iter 24 adds the Swift consumer + UI. Pair-paced so each iter is independently reviewable.

**Deep trace walkthrough:**
1. Read `PythonRunner.swift` as the reference pattern. `run() -> AsyncThrowingStream<ProgressEvent, Error>` with `Task.detached { await self.launch(...) }`, `proc.terminationHandler` continuation, stderr drain task that yields parsed events via `JSONLProgressParser`.
2. Replicated the pattern in `PublishService` as `publishWithProgress(modelPath:repo:isPrivate:token:)`. Key differences from PythonRunner:
   - Not an actor — PublishService is `@MainActor enum`. Stream is spawned via `Task.detached` to escape the MainActor for the subprocess wait.
   - Passes `--progress json` to subprocess (iter 23's flag). Python emits JSONL on stderr.
   - Sets `HF_HUB_TOKEN` env (iter 6 security pattern) + merges `childProcessEnvAdditions` (iter 11).
   - Scrubs `token` from stderr on failure (iter 6 layer-2).
3. Read `PublishToHuggingFaceSheet.swift`. Pre-iter-24 `runPublish()` was: set isPublishing=true → await one-shot `publish(...)` → set publishResult. Replaced with: set isPublishing=true + reset progress state → `for try await event in PublishService.publishWithProgress(...) { apply(event:) }` → reconstruct PublishResult at stream end.
4. **State design:**
   - `progressPhase: String` — current 3-phase name (scan/upload/finalize).
   - `progressBytes: (done: Int64, total: Int64)?` — optional because the first tick sets totals; before that the UI shows a spinner.
   - `progressLabel: String` — per-file filename for context.
   - `progressLog: [String]` — breadcrumb trail for diagnostics.
5. **UI design:** New Section "Uploading" visible while `isPublishing`. `ProgressView(value:, total:)` with a caption showing "X.XX / Y.YY GB (N%)". Current filename underneath in secondary color, line-limited + truncated middle. Empty-state fallback: indeterminate spinner during Phase 1 (scan) before any tick fires.
6. **Contract-pinning tests:**
   - `test_publishWithProgress_rejects_empty_token`: empty token must throw `missingToken` on the FIRST stream iteration — NO subprocess spawn (defensive early-exit preserves iter-6 security posture).
   - `test_publishWithProgress_is_async_stream`: type-level pin. If the return type ever changes from `AsyncThrowingStream<ProgressEvent, Error>`, this test fails to compile, forcing intentional migration.

**M43 end-to-end acceptance:**
- Python side (iter 23): emits 3 phase events + info + ≥1 tick per file + final 100% tick.
- Swift side (iter 24): decodes same JSONL, updates @State, drives a real progress bar.
- UI shows: scan phase spinner → "2.13 / 187.42 GB (1%)" progressing up to 100% → finalize → published confirmation with repo + commit URLs.

**Items touched:**
- M43 [x] — Swift half shipped. End-to-end complete.

**Commit:** (this iteration)

**Verification:** 111 Swift tests (was 109, +2). jang-tools 245 + ralph_runner 68 unchanged. **Cat A fully drained.**

**Closed-status tally:** 38 (prior) + M43-Swift (already counted in iter 23) = still 38 closed / 74 total = 51%. The M43 counter didn't move (it's one item across both iters) but Cat A is now ZERO remaining.

**Meta:** Two-iter decomposition worked well. Iter 23 pinned the protocol (Python JSONL with tests ensuring Swift-consumable schema). Iter 24 built the consumer against the pinned protocol. Each iter independently reviewable and reversible; total scope larger than any single iter would accept.

**Next iteration should pick:** Now that Cat A is drained, good candidates:
- M48 (default repoName missing org prefix, small polish — would complement iter-24's publish UX work)
- M87 (Mistral 4 live validation — needs real convert; could be deferred to Ralph harness)
- Cat D third pass due around iter 27-29 per the 5-7 iter cadence
- Unexplored audit categories: Cat F (spawned from iter 21 memory drift — might catch more drift on `project_bfloat16_fix.md`, `project_mistral4_architecture.md` M87, `project_cascade2.md`, etc.)
- M77-adjacent: are there other places in the Swift wizard that treat chat_template.jinja as the only file form?

---

## 2026-04-20 iteration 25 — M48 defaultHFOrg Settings + sheet prefix

**Angle:** Tiny UX polish that would have been caught by live use but is invisible on the test harness. M48 was spawned iter 7 alongside M46's validation — validation catches the bad default but the bad default is SHOWN on every sheet open. Complements iter-24's publish UX work by eliminating the "validation error on first click" friction.

**Deep trace walkthrough:**
1. Pre-iter-25 PublishToHuggingFaceSheet.init:
   ```swift
   self._repoName = State(initialValue: defaultRepoName.isEmpty
                          ? modelPath.lastPathComponent
                          : defaultRepoName)
   ```
   `modelPath.lastPathComponent` = `"MyModel-JANG_4K"`. M46 (iter 7) validation: "Repo id must be in the format 'org/model-name' (one forward slash)". Every first-click of the sheet lands the user on a validation error.
2. **Design trade-offs considered:**
   - Option A: store the user's HF org in AppSettings. User types it once in Settings; sheet prefixes automatically.
   - Option B: infer org from the HF token via `/api/whoami-v2` call. More automatic but adds a network dependency + latency + auth failure mode.
   - Option C: strip the validation entirely. Nope — M46 was added precisely because invalid repos burn 30+ seconds on HfHubHTTPError.
   - **Chose A.** Simplest, zero network cost, user fully in control. Multi-org users can leave the field empty and type each time.
3. **@State init issue caught during implementation:** Can't read `@Environment(AppSettings.self)` in the init — environment is only resolved in `body`. Solutions: pass settings through init (tight coupling from call site) OR defer the prefix to first-render via `.task`. Picked `.task` + an `orgPrefixApplied` guard so re-mounts don't double-prefix or stomp user edits.
4. **Defensive `applyOrgPrefixIfNeeded()`:** no-ops in three cases — empty org, repo already contains `/`, repo differs from the basename default (user started typing). Last guard is the critical one: SwiftUI task could fire after a user has already edited the field; we must NOT stomp.
5. **Snapshot migration:** `Snapshot.defaultHFOrg` has a Swift Codable default (`= ""`) so any pre-iter-25 UserDefaults snapshot decodes cleanly with an empty string. Regression test `test_pre_iter25_snapshot_defaults_hf_org_to_empty` pins this invariant.
6. **SettingsWindow UI:** new "Publishing" section on the General tab (between Behavior and Reset). Caption: "Pre-fills the Publish sheet's repo field as {org}/{model-name}. Leave empty if you publish to multiple orgs." — guides the user through the tradeoff so it's not a surprise when they swap orgs.

**Items touched:**
- M48 [x] — default HF org persisted, UI wired, sheet prefix applies defensively.

**Tests (4 new):** default is empty, roundtrip across process restart, reset clears it, pre-iter-25 snapshot (missing the field entirely) decodes with a defaulted empty value.

**Commit:** (this iteration)

**Verification:** 115 Swift tests (was 111, +4). jang-tools 245 + ralph_runner 68 unchanged.

**Closed-status tally:** 38 (prior) + M48 = 39 closed / 74 total = 53% closure rate.

**Adjacent issue spotted + DEFERRED:**
- **M88 (new):** the sheet's init ALSO reads token from `ProcessInfo.environment` — but `.task`'s lifecycle differs from init's. Token init happens BEFORE settings are available; org prefix happens AFTER. If a future field needs BOTH at construction time, this pattern breaks. Worth a refactor pass to unify lifecycle but not urgent.

**Next iteration should pick:** Cat D third pass (due iter 27-29 per cadence; can come early). Alternatively M87 Mistral 4 live validation (needs a real convert workflow). Alternatively do M88 adjacent — unify the sheet's init-vs-task field-lifecycle issue. Or investigate whether VerifyStep's adoption action row (iter 14's entry point to publish sheet) passes the right modelBasename through — a subtle off-by-one could mean the prefix runs against the wrong string.

---

## 2026-04-20 iteration 26 — M89 Cat D third pass: EOS coverage for Qwen3-VL

**Angle:** Third Cat D cross-ref pass, brought forward from the scheduled iter-27-29 window because iter-21 and iter-22 both found real coverage gaps in memory↔code alignment. Pattern: each pass finds one bug worth ~1 iter of cleanup. Worth maintaining the cadence.

**Deep trace walkthrough (M89):**
1. `feedback_chat_template_rules.md` (31 days old) documents the Qwen3.5 eos fix: source ships 248044 (<|endoftext|>), must be 248046 (<|im_end|>) or model loops infinitely during inference.
2. Cross-ref `convert.py`: `_eos_fixes` dict covers exactly two keys: `qwen3_5` + `qwen3_5_moe`. Works for dense + MoE.
3. Check `recommend.py` for other Qwen3-family model_types that might share the same tokenizer ID space. Found: `qwen3_vl` (image VL) is a recognised family. Plus the newer Qwen3.6 arch which is registered under `qwen3_5_moe` (confirmed from `project_qwen36.md` and the recommend.py family-map note "Qwen 3.6 (256 experts + gated_deltanet)" — shares the model_type with qwen3_5_moe, so NO gap there).
4. **Gap CONFIRMED:** if a user converts a Qwen3.5-VL source that ships the wrong 248044 eos, the fix doesn't apply. No user has reported this (maybe no Qwen3-VL sources in that tier have landed with the bug yet, or HF has since published corrected checkpoints). But the memory rule clearly extends to the whole family, and the fix gate (`if tc.get("eos_token_id") in eos_fix_map`) means adding coverage is always safe: models with correct eos see no change, models with wrong eos get corrected.
5. **Implementation decisions:**
   - Extract `_eos_fixes` from convert_model's locals to module-level `EOS_FIXES`. Testable independently without instantiating the full convert pipeline.
   - Added `qwen3_vl`, `qwen3_moe_vl`, `qwen3_5_vl` to the coverage map. All share the Qwen2Tokenizer family.
   - Kept the `in eos_fix_map` gate at the use site so extension carries zero regression risk.
6. **Test design trade-off:** could test end-to-end via convert_model on a synthetic tokenizer_config.json. Instead, tested the data structure alone because:
   - convert_model needs a real safetensors shard + weight quantization setup — heavyweight for a test.
   - The bug was in the DATA (missing map entries), not the LOGIC (which was already correct). Testing the data directly is more precise.
   - Pinned shape/coverage/negative/idempotency/numeric invariants in 6 cases.
7. **Negative test is critical:** `test_no_coverage_for_unrelated_families` asserts that llama/mistral/gemma/phi/qwen2/minimax/deepseek/nemotron are NOT in the map. Otherwise a future "helpful" PR broadening coverage to "everything" would mis-correct token IDs on those architectures (248046 isn't `<|im_end|>` for non-Qwen families).

**Items touched:**
- M89 [x] — Qwen3-VL variants now covered by the EOS fix; 6 regression tests pin shape + coverage + negative + idempotency + numeric.

**Commit:** (this iteration)

**Verification:** 251 jang-tools (was 245, +6). ralph_runner 68 + Swift 115 unchanged.

**Closed-status tally:** 39 (prior) + M89 = 40 closed / 75 total = 53% closure rate.

**Meta on Cat D cadence observed:**
- iter 5 (Cat D pass 1): M35 Osaurus remap drift
- iter 21 (Cat D pass 2): M83 MLP asymmetry threshold drift
- iter 22 (Cat D pass 2 continued): M86 Mistral 4 architecture spot-check (6/7 clean, M87 flagged)
- iter 26 (Cat D pass 3): M89 EOS coverage gap
→ **Three of four Cat D passes found a real coverage / drift bug.** The single clean pass (iter 22 Mistral 4 architecture) was the one where iter 9's loader.py audit had already verified the fixes. Suggests Cat D passes are MOST productive on memory files that haven't been cross-ref'd recently — focus rotation on oldest-unchecked memories for best yield.

**Next iteration should pick:** M88 (sheet init-vs-task field-lifecycle unification — iter-25 spawn, small cleanup). Or continue Cat D with another memory file (iter-26 pattern suggests high yield — candidates: `project_qwen36.md`, `project_cascade2.md`, `project_minimax_m27.md`, `feedback_model_checklist.md`, `feedback_readme_standards.md`). Or M87 live Mistral 4 validation (needs real convert — possible iter-27 work if a Mistral 4 source is at hand).

---

## 2026-04-20 iteration 27 — Cat D fourth pass: M90 reasoning/thinking YAML tags

**Angle:** Continue Cat D cadence since iter 26 showed the yield stays high on memory files that haven't been cross-ref'd recently. Scanned three candidates before picking M90.

**Deep trace walkthrough:**
1. `project_qwen36.md` (3 days old) — three flagged P1/P5/P10 items. **P1 already fixed** (GATED_DELTANET_CONFIGS uses `linear_attn.*` names now, per the explicit code comment at `architectures.py:217`). **P10 already fixed** (in_proj_b/a min_bits=4, preferred=8). **P5 unfixed but unfixable from our side** (mlx-lm internals) — logged as M92 for a "detect + warn" follow-up rather than a direct fix. → No concrete drift here; memory is current.
2. `project_cascade2.md` (29 days old) — 128 experts < 256 threshold so post-M83 asymmetry floor correctly bypasses. hidden=2688 < 4096 so bfloat16 guard correctly bypasses. nemotron_h loader handling confirmed. → No drift.
3. `feedback_readme_standards.md` (30 days old) — 12 rules for HF uploads. Most are about human-curated eval content. But **rule 10 is automatable**: "YAML frontmatter must include `reasoning` AND `thinking` tags for all Qwen3.5 models". Cross-referenced against `detect_capabilities` + `model-card.md.jinja`.
4. **BUG M90 CONFIRMED.** `detect_capabilities` ORed `reasoning_parser` and `enable_thinking` into ONE `has_reasoning` boolean. The template emitted a single `- reasoning` YAML tag. The separate `- thinking` tag memory rule 10 requires was NEVER emitted, even on Qwen3.5 models with explicit `enable_thinking: true`. Every Qwen3.5 HF upload silently violated the rule.
5. **Semantic argument for two flags:**
   - `has_reasoning` = "this model can produce reasoning output at all" (broader capability)
   - `has_thinking` = "the `enable_thinking` runtime toggle is available" (specific user-facing knob)
   - Users filtering HF models by `thinking` tag want the latter specifically.
6. **Fix pattern:**
   - Added `has_thinking: bool(cfg.get("enable_thinking"))` as a NEW separate field. Preserved `has_reasoning` semantics so existing callers (Python snippet generator, etc.) keep working.
   - Template gained a `{% if has_thinking %}- thinking{% endif %}` block alongside the existing reasoning block.
7. **Positive + negative test pair critical:**
   - Positive: Qwen3.5 + `enable_thinking: true` emits BOTH tags.
   - Negative: plain model without `enable_thinking` gets NEITHER the thinking tag nor a stray one from the OR-fallback path. Prevents a future reversion that ORs again.

**Additional drift noted but not fixed (logged as M91):**
Memory `feedback_readme_standards.md` specifies 9+ other HARD upload requirements (per-subject MMLU, JANG-vs-MLX comparison, speed comparison, size comparison, Korean section, per-subject comparison table, no TBD, no duplicate tables, no wrong profile data). These are UNAUTOMATABLE without running MMLU evals, but the current template produces a skeleton that would silently fail the memory's rules if published directly. Solution is UX: either explicit TODO placeholders or a GenerateModelCardSheet warning banner. Scoped out of this iter; M91 captures the follow-up.

**Items touched:**
- M90 [x] — `thinking` YAML tag now emitted on Qwen3.5 models per rule 10; regression pinned.

**Commit:** (this iteration)

**Verification:** 255 jang-tools (was 251, +4). ralph_runner 68 + Swift 115 unchanged.

**Closed-status tally:** 40 (prior) + M90 = 41 closed / 77 total = 53% closure rate.

**Cat D yield now 4 of 5 passes found bugs (iter 5, 21, 22, 26, 27 — 4 real bugs + 1 clean).** Cadence is paying off. Good candidates for next-pass: `feedback_model_checklist.md`, `project_minimax_m27.md`, `reference_architecture_details.md`.

**Next iteration should pick:** M91 (model-card skeleton vs rule-10 mismatch — UX gap, small scope). Or M88 (sheet init-vs-task unification from iter 25). Or M87 live Mistral 4 validation. Or another Cat D pass targeting `feedback_model_checklist.md` (untouched, would close the Cat D coverage loop).

---

## 2026-04-20 iteration 28 — M91 skeleton-card warning (visibility UX)

**Angle:** Close M91 — the "template would silently violate memory rules if published" gap surfaced in iter 27. Scoped to VISIBILITY rather than automation because the underlying rules (per-subject MMLU, JANG-vs-MLX tables, Korean section) require live evals that the build-time template can't produce.

**Deep trace walkthrough:**
1. Re-read `feedback_readme_standards.md` (30 days old). 12 rules total:
   - Rules 1-6: require EVAL DATA (MMLU scores, comparisons, speed, size). Can't automate without running evals.
   - Rules 7-9 (no duplicates, no wrong profile data, no empty Korean tables): negative checks that would need a curator anyway.
   - Rule 10 (YAML `reasoning`+`thinking` tags): automated in iter 27.
   - Rule 11 (MiniMax text-only): should be automatable via `has_vl` check — spawn follow-up.
   - Rule 12 (review the FULL README before upload): process/UX rule.
2. **Scope decision: visibility, not automation.** The memory is about what Eric REVIEWS before a manual HF publish. JANG Studio generates a skeleton for the user to extend. The bug is that the gap between skeleton and upload-ready is INVISIBLE.
3. **Two-layer visibility fix:**
   - Swift UI: a new `skeletonWarning` section at the top of `GenerateModelCardSheet` — renders AS SOON AS the card loads, before the Save/Copy buttons. Orange-tinted with the explicit phrase "Skeleton only — not upload-ready" plus a caption listing the missing sections. Impossible to miss.
   - CLI stderr: `jang-tools modelcard` now prints a NOTE to stderr after generation. Ralph harness tails stderr so the note appears in logs; humans running CLI manually see it.
4. **Critical design: stderr not stdout.** The CLI's `--json` mode's stdout must stay parseable — a warning on stdout would break every downstream consumer (Swift, Ralph, future CI integrations). Explicit test `test_cli_emits_skeleton_warning_to_stderr` pins this invariant.
5. **Gotcha caught during test writing:** pytest's `tmp_path` fixture creates per-test directories like `pytest-of-eric/pytest-638/test_cli_emits_skeleton_warnin0/dense`. The path gets embedded in the Python snippet's `load_jang_model("...")` string. A naive substring check `"skeleton" in stdout` false-positives on the embedded path. Switched assertion to the distinctive warning phrase `"generated card is a skeleton"` which is unique to the stderr message.

**Items touched:**
- M91 [x] — gap made visible at both Swift UI + CLI layers. Tests pin stderr-only, stdout-parseable, memory-ref in message.

**Commit:** (this iteration)

**Verification:** 256 jang-tools (was 255, +1). ralph_runner 68 + Swift 115 unchanged. Swift `skeletonWarning` section is pure markup — no test-level Swift coverage added since the behavior is "always shows when card loads" which is trivially verified by reading the code.

**Closed-status tally:** 41 (prior) + M91 = 42 closed / 77 total = 55% closure rate.

**Spawn candidates identified during scan (not fixed):**
- **M93**: `feedback_readme_standards.md` rule 11 says "MiniMax is text-only — never include VLM code". Template has an `is_vl` check so MiniMax (detected non-VL) would get the text path correctly. But the rule suggests manual curators still sometimes mix VLM code into MiniMax cards — might be worth adding a POST-generation check that if `model_type == "minimax_m2"` and the card contains VLM-specific markers (like `mlx_vlm`, `Image.open`), flag it.
- **M94**: the CLI stderr warning is shown EVERY time `jang-tools modelcard` runs — even when called from Swift where the UI banner already covers it. Could add a `--quiet-note` flag for the Swift caller so we don't double-surface. Low priority since Swift doesn't surface stderr to the user except on failure.

**Next iteration should pick:** M88 (sheet init-vs-task field-lifecycle, iter-25 spawn). Or continue Cat D with `feedback_model_checklist.md` (untouched). Or M87 live Mistral 4 validation. Or M93 (MiniMax-vs-VLM check — small + closes rule 11). Or the next natural user-flow audit — VerifyStep's adoption action row wiring (flagged at end of iter 25).

---

## 2026-04-20 iteration 29 — M93 MiniMax text-only enforcement (rule 11)

**Angle:** Close M93 — the last automatable rule from `feedback_readme_standards.md`. Pairs cleanly with iter-27 (M90 reasoning/thinking tag) and iter-28 (M91 skeleton warning) as a three-iter sweep of that memory file's template-enforceable surface.

**Deep trace walkthrough:**
1. Rule 11: "MiniMax is text-only — never include VLM code". Traced `detect_capabilities` in examples.py: `is_vl` = `(model_dir / "preprocessor_config.json").exists()`. Pure file-existence check, no model_type gating.
2. **Failure mode:** if a MiniMax output dir contains `preprocessor_config.json` (copy residue from a prior VL convert in the same workspace, user manually dragging files, broken converter from an earlier run), `is_vl` flips to True. Python snippet template then takes the `{% if is_vl or is_video_vl %}` branch, emitting `from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model` + `from mlx_vlm import generate` + `from PIL import Image`. At runtime this breaks because MiniMax has no vision tower — user's adopter sees cryptic import errors on a published card.
3. **Fix approach (conservative):**
   - New `_TEXT_ONLY_MODEL_TYPES = frozenset({"minimax_m2", "minimax", "minimax_m2_5"})` listing every alias MiniMax is registered under (cross-ref with `capabilities.py:37-39`).
   - `is_vl` / `is_video_vl` detection now requires BOTH the file AND model_type NOT in the text-only set. Conservative: MiniMax always text-only; genuine VL models (qwen2_vl, qwen3_vl) unaffected.
4. **Alias coverage note:** `capabilities.py` maps `minimax_m2 / minimax_m2_5 / minimax` all to the same arch. Missing any alias = missing enforcement for that spelling. Test pins all three.
5. **Negative test is critical:** `test_genuine_vl_model_still_vl` asserts that qwen2_vl with preprocessor IS still detected as VL. Prevents a future overzealous broadening of `_TEXT_ONLY_MODEL_TYPES` from breaking genuine VL detection.

**Items touched:**
- M93 [x] — MiniMax now forced text-only regardless of stray preprocessor files. Rule 11 enforced.

**Commit:** (this iteration)

**Verification:** 260 jang-tools (was 256, +4). ralph_runner 68 + Swift 115 unchanged.

**Closed-status tally:** 42 (prior) + M93 = 43 closed / 77 total = 56% closure rate.

**Three-iter sweep of `feedback_readme_standards.md` complete:**
- iter 27: M90 — `thinking` YAML tag (rule 10)
- iter 28: M91 — skeleton warning at Swift UI + CLI (rules 1-6, 9, 12)
- iter 29: M93 — MiniMax text-only enforcement (rule 11)

Rules 7-8 (no duplicate tables, no wrong profile data) still aren't automated — they're curator-audit rules that require reading the specific card text. Could potentially add a lint pass but deferring as M95.

**Next iteration should pick:** M88 (sheet init-vs-task field-lifecycle, iter-25 spawn — small cleanup). Or M87 live Mistral 4 validation (needs real convert). Or continue Cat D with `feedback_model_checklist.md` (untouched). Or the VerifyStep adoption action row wiring audit (flagged iter 25). Or broader UX exploration: what happens if user cancels the publish stream mid-upload (iter-23/24 flow) — does the half-uploaded repo get cleaned up, or is it stranded with partial files?

---

## 2026-04-20 iteration 30 — M96 publish cancel propagation (Cat C concern)

**Angle:** Picked the "publish cancel" question that iter 29 surfaced. Iter 23-24 shipped streaming publish + UI progress bar but never verified cancellation propagates from UI → stream → subprocess. Classic layered system where each layer assumes the next one handles cleanup. Let's trace.

**Deep trace walkthrough:**
1. UI layer: `PublishToHuggingFaceSheet.runPublish()` does `for try await event in PublishService.publishWithProgress(...) { apply(event:) }`. The Task wrapping this runs via `Task { await runPublish() }` — DETACHED, no cancellation path to it.
2. **First bug found:** no Cancel button during `isPublishing`. Even if cancellation worked, user couldn't trigger it. They'd have to dismiss the sheet, which also doesn't cancel the detached Task.
3. Service layer: `publishWithProgress` returns `AsyncThrowingStream` from `AsyncThrowingStream { continuation in Task.detached { await _streamPublish(...) } }`. **No `continuation.onTermination` handler.**
4. **Second bug found:** when the consuming Task cancels (or the stream is abandoned), the continuation terminates but NOTHING tells the subprocess. `_streamPublish` is still awaiting `proc.terminationHandler` inside `withCheckedContinuation`. Subprocess runs to natural completion.
5. **Confirmed failure mode end-to-end:** user closes Publish sheet mid-upload → SwiftUI dismisses the view → Task detached from view keeps running → Python subprocess keeps uploading to HF → half-published repo is stranded with partial files on HuggingFace → no user feedback → user has no idea the upload is continuing.
6. **Fix architecture:**
   - **Need a way** for a Sendable closure (onTermination) to SIGTERM a Process reference that's created on a DIFFERENT task. Process doesn't conform to Sendable; a global / static reference would race.
   - **Solution:** `ProcessHandle` class — locked wrapper around `process: Process?` + `wasCancelled: Bool`. The @unchecked Sendable annotation is safe because all accesses go through the lock.
   - onTermination stores `handle.cancel()` before the work Task spawns. Work Task calls `handle.set(process: proc)` after run() lands.
   - **Race case:** what if cancel() fires BEFORE set()? Happened during testing. Fixed: set() checks `_wasCancelled` and terminates immediately if true. Test `test_processHandle_cancel_before_set_is_safe` pins this with a live /bin/sleep 1000 subprocess.
7. **Exit semantics:** on cancel, subprocess exits non-zero (SIGTERM termination code 15). Without special handling that would look like a cliError in the stream's `continuation.finish(throwing:)` branch. Fix: check `handle.wasCancelled` before throwing — clean finish instead.
8. **UI wiring:**
   - `@State publishTask: Task<Void, Never>?` stores the consuming Task reference.
   - `@State wasCancelled: Bool` tracks user intent separately from stream termination.
   - "Cancel upload" button with `.keyboardShortcut(.cancelAction)` (Escape key) visible only during `isPublishing`. Tapping it sets wasCancelled + calls publishTask?.cancel().
   - `runPublish` distinguishes three branches after the stream: success, user-cancel (informative message about partial HF files), error.
   - `CancellationError` swallowed silently in the catch — surfaced via wasCancelled branch instead. Prevents "Task was cancelled" from showing as a generic red error.
9. **Not fixed — flagged M97:** local subprocess is terminated, but files already uploaded to HF stay there. Cleanup would require calling `huggingface_hub.delete_folder` on cancel with a confirmation dialog. Non-trivial because a long user-cancel window could leave HF in an intermediate state that's hard to clean up atomically. Lower priority than the local-subprocess fix which stops the bleeding.

**Items touched:**
- M96 [x] — end-to-end cancel propagation: UI button → consuming Task cancel → stream onTermination → ProcessHandle.cancel → SIGTERM + SIGKILL escalation.

**Commit:** (this iteration)

**Verification:** 119 Swift tests (was 115, +4). jang-tools 260 + ralph_runner 68 unchanged.

**Closed-status tally:** 43 (prior) + M96 = 44 closed / 78 total = 56% closure rate.

**Meta-lesson on "end-to-end cancellation":**
This bug was latent since iter 23-24. No test caught it because testing subprocess cancellation requires a real child process and timing coordination. The 4 new ProcessHandle tests use a real `/bin/sleep 1000` subprocess and check actual isRunning / termination status — slower (seconds not ms) but catches race conditions that mock-based tests wouldn't. Worth the test cost for any cancel-propagation path. **Pattern to apply:** when adding a feature that spans Task / stream / subprocess boundaries, ALWAYS add one integration test with a real subprocess. Mock tests verify the contract; subprocess tests verify the plumbing.

**Next iteration should pick:** M88 (sheet init-vs-task lifecycle, iter 25 spawn — small). Or M97 (partial HF repo cleanup — bigger, follows naturally from M96). Or M87 (Mistral 4 live RoPE validation). Or continue Cat D with `feedback_model_checklist.md`. Or audit the convert cancel path (PythonRunner) with the same end-to-end rigor applied here — iter 3 established the pattern but hasn't been stress-tested for Task-cancel-to-subprocess scenarios like M96 was for publish.

---

## 2026-04-20 iteration 31 — M98 PythonRunner cancel propagation (applying iter-30 meta-lesson)

**Angle:** Iter 30's meta-lesson: "features spanning Task/stream/subprocess boundaries need integration tests with real subprocesses". Applied to PythonRunner which iter 3 last audited but never stress-tested for Task-level cancel.

**Deep trace walkthrough:**
1. Reviewed `PythonRunner.run()`: returns `AsyncThrowingStream { continuation in Task.detached { launch } }`. **No `continuation.onTermination` handler** — same pattern that iter 30 fixed for PublishService.
2. Predicted bug: consumer cancel → stream throws CancellationError → subprocess orphaned. Wrote a test FIRST that spawns a real `/bin/bash while true; do date +%s%N > tickFile; sleep 0.2; done` subprocess, cancels the consuming Task, then checks if tickFile mtime stops advancing.
3. **Ran test pre-fix:** confirmed failure at line 104 "tick-file mtime advanced after cancel — subprocess is still running". Bug confirmed.
4. **Added fix** — onTermination in run()'s build closure calling cancel(). Ran test: NEW failure mode at line 86 "subprocess should have written a tick by now". Subprocess never spawned OR was killed before first tick.
5. **Debugged with stderr trace:** `FileHandle.standardError.write("[PR-DEBUG] onTermination fired: \(reason)")`. Output: `onTermination fired: cancelled` — fired IMMEDIATELY on stream creation under XCTest's Task isolation. Reason=cancelled means the continuation's stream was deemed cancelled. Not a bug in the fix per se — a Swift 6 isolation quirk where registering onTermination in the stream's build closure triggers a spurious termination event when consumer Task's context is XCTest-task-ish.
6. **Fix refined:** moved `continuation.onTermination = ...` INSIDE `launch()` after `try proc.run()`. Now fires only post-spawn, avoiding the false-positive cancel. Test passed.
7. **Test tuning:** initial 500ms wait for first tick was too tight under parallel test contention (spawn latency spikes past 1s on loaded M1). Switched to polling loop up to 3s.
8. **Companion test removed:** wrote `test_streamAbandon_terminatesSubprocess` too, but discovered that stream-abandon without any iterator is a DIFFERENT class of resource leak — onTermination never fires because the producer closure's `continuation` reference keeps the continuation alive indefinitely. Removed the test; logged as M99 (low urgency — all production call sites iterate the stream).
9. **Full suite re-run:** 120 tests pass (was 119). `test_consumerTaskCancel_terminatesSubprocess` reliably passes in both isolation and full-suite runs after the tuning.

**Items touched:**
- M98 [x] — Task-cancel → subprocess SIGTERM now propagates end-to-end in PythonRunner.

**Commit:** (this iteration)

**Verification:** 120 Swift tests (was 119, +1). jang-tools 260 + ralph_runner 68 unchanged.

**Closed-status tally:** 44 (prior) + M98 = 45 closed / 79 total = 57% closure rate.

**Meta-lesson reinforced:**
- **Write the failing test FIRST** when the suspected bug is cross-layer. Pre-fix test run at `line 104` assertion proved the bug existed. Without that, the "fix" might have been placebo.
- **Debug-via-stderr traces** are fast for narrowing which LAYER of the fix is miscompiling — `[PR-DEBUG] onTermination fired: cancelled` pinpointed the Swift-isolation quirk in 1 run that could have taken many iterations of code-reading.
- **Integration tests with real subprocesses** are slow (5+ seconds each) but catch cross-layer race conditions that mocks cannot. Iter 30 ProcessHandle + iter 31 PythonRunner = two confirmed-via-real-subprocess bugs that would have been invisible with pure unit tests.

**Next iteration should pick:** The iter-30/31 pair established a clear pattern for cross-layer cancel auditing. Next subprocess-holding surface to audit with the same rigor: `InferenceRunner.generate()` (iter 3 M19 fixed explicit cancel but never stress-tested consumer-Task / stream-abandon paths). Alternatively M88 (sheet init-vs-task lifecycle) or M87 (Mistral 4 live validation) or continue Cat D with `feedback_model_checklist.md`.

---

## 2026-04-20 iteration 32 — M100 InferenceRunner consumer-Task cancel (completes cross-layer cancel sweep)

**Angle:** Third iteration applying iter-30's meta-lesson. Iter 30 found M96 in PublishService. Iter 31 found M98 in PythonRunner (same class). This iter checks InferenceRunner — the last subprocess-holding surface that iter 3 established a cancel pattern for but never stress-tested end-to-end.

**Deep trace walkthrough:**
1. Read `InferenceRunner.generate()`. Unlike PythonRunner/PublishService (both streams), this is a single `async throws -> InferenceResult`. So the cancellation API shape is different.
2. `generate()` awaits `withCheckedContinuation { proc.terminationHandler = ... }`. **CheckedContinuation does not participate in Task cancellation** — the await blocks until the subprocess naturally exits OR `runner.cancel()` is called explicitly. Consumer-Task cancel has no propagation path.
3. Predicted failure mode: user's Swift code does `Task { try await runner.generate(...) }`. The Task is cancelled mid-run (view dismount, SwiftUI task lifecycle, user abort). The subprocess continues running — model load continues, inference continues, result is computed but nobody consumes it. GPU pinned, memory held, no UI feedback.
4. On a 70 GB JANG model with ~30s load time, this creates a real UX pain: user cancels a prompt, GPU stays busy for tens of seconds, user thinks the app is frozen.
5. **Write failing test first** — replicating iter-31 pattern. Added `executableOverride: URL? = nil` init parameter (matching PythonRunner's testability pattern). Test spawned a tick-writing bash script.
6. **Test HUNG indefinitely** at `try? await consumerTask.value` after cancel — because generate() was stuck in the continuation wait forever. Had to `pkill -f xcodebuild` to recover. This is the strongest possible proof of the bug: not a failed assertion, but a complete deadlock at the cancel boundary.
7. **Fix:** wrap the `withCheckedContinuation` inside `withTaskCancellationHandler { ... } onCancel: { Task.detached { await self.cancel() } }`. Task cancel now triggers SIGTERM + SIGKILL escalation via the existing actor cancel() method. The onCancel closure is `@Sendable` and nonisolated; hops onto the actor via Task.detached.
8. **Test refactor to avoid harness-timeout on regression:** removed `try? await consumerTask.value` from the test. If the fix ever regresses, we'd hang for the test harness's 10-minute timeout instead of getting an informative assertion failure. Now we sleep 5s past SIGTERM+SIGKILL window and verify via tick-file mtime non-advance. Same diagnostic power, no hang risk.
9. **Regression pin for iter 3's M19:** added `test_explicit_cancel_still_works_via_actor_method` — explicit `await runner.cancel()` must continue to work after the Task-cancel addition. Both paths must terminate the subprocess; neither must regress the other.
10. **Verification under concurrent test load:** ran full Swift suite (122 tests). InferenceRunner tests pass reliably alongside PythonRunnerTests (which also do real-subprocess spawn). No flakes observed.

**Items touched:**
- M100 [x] — InferenceRunner consumer-Task cancel now propagates to subprocess. Cross-layer cancel sweep complete across all three subprocess-holding surfaces (publish / PythonRunner / InferenceRunner).

**Commit:** (this iteration)

**Verification:** 122 Swift tests (was 120, +2 — consumer-cancel + explicit-cancel regression pin). jang-tools 260 + ralph_runner 68 unchanged.

**Closed-status tally:** 45 (prior) + M100 = 46 closed / 79 total = 58% closure rate.

**Cross-layer cancel sweep summary (iters 30-32):**
| Iter | Item | Surface | Root pattern | Fix |
|------|------|---------|--------------|-----|
| 30 | M96 | PublishService stream | no onTermination | ProcessHandle + onTermination |
| 31 | M98 | PythonRunner stream | no onTermination | onTermination inside launch() |
| 32 | M100 | InferenceRunner single-async | CheckedContinuation doesn't honor cancel | withTaskCancellationHandler wrap |

Three different Swift async primitives (AsyncThrowingStream from actor, AsyncThrowingStream from enum, single async throws from actor) — three different fix shapes. All share the same root: **subprocess-holding async APIs must explicitly handle Task cancellation**. Swift's structured concurrency doesn't propagate automatically through `await withCheckedContinuation`.

**Pattern established for future audits:** any `await withCheckedContinuation` used to bridge a callback-based C / ObjC / subprocess API to Swift async MUST be wrapped in `withTaskCancellationHandler` when the callback can take arbitrarily long. This is a reviewable code-smell worth grep-audit-ing the codebase for.

**Next iteration should pick:** Audit the above pattern across the rest of the Swift codebase. `grep -n "withCheckedContinuation" JANGStudio/` and stress-test each call site. Alternatively M87 (Mistral 4 live validation), M88 (publish sheet init-vs-task), M97 (partial HF cleanup), or Cat D with `feedback_model_checklist.md` (untouched).
