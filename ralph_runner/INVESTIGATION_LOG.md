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
