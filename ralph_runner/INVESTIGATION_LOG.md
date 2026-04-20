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
