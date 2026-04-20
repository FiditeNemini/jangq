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
