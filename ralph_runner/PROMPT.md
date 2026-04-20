# Ralph Loop — JANG Studio deep-trace audit

You are auditing JANG Studio for production readiness by **deeply tracing real user journeys, real data flows, and real edge cases** — not just checking items off a list. Each iteration you pick ONE investigation angle, trace it root-to-leaf across Swift + Python + bundled runtime, discover new concerns, and either fix them or add them to the checklist.

The loop runs forever. You emit completion only when you've traced every dimension below with real evidence AND cannot invent a new question worth investigating. That bar is very high.

## What each iteration does

1. **Read the checklist state:** `cat /Users/eric/jang/ralph_runner/AUDIT_CHECKLIST.md`
2. **Pick ONE investigation angle** using the selection policy below. Announce it in a short (≤ 40 word) opening statement.
3. **Trace it deeply** — read real files, follow the call chain, run real commands, observe actual outputs. Cross-check against memory files and documented rules.
4. **Discover new concerns** — as you trace, NEW questions will surface. Append them to the checklist as `[ ]` items in the appropriate section (or create a new section). Every iteration must net-add at least 1 new question, not just close existing ones.
5. **Fix real drift** — if you find broken behavior, fix it (commit), add SHA to the closed item.
6. **Close what you verified** — mark items `[x]` only with concrete evidence (file paths + line numbers + command outputs, not "I read the code and it looks right").
7. **Commit** with a message describing the investigation + items touched.
8. **End turn.**

## Investigation angle selection policy

Rotate through these categories. Never pick the same category two iterations in a row.

### A. User journey trace

Pick a persona + a concrete task, mentally walk through the app step by step, and at each UI interaction ask: **what if...?**

Example personas:
- **Beginner** — never heard of JANG, has a HF model folder they downloaded, wants to convert it. What if the folder is wrong? What if they pick an output dir that's gitignored? What if they click wrong buttons?
- **Power user** — has 3 models to convert, wants to do them in parallel, knows the profiles. What settings do they tweak? What failure modes matter most?
- **Adopter** — wants to ship a Swift app that loads JANG models. Copies the snippet from the Usage Examples sheet. What breaks?
- **Researcher** — converts same model at 5 different profiles, compares quality. What telemetry do they need? What's missing from the UI?

For each step of the journey, ask:
- What happens if the input is empty? malformed? enormous?
- What happens if the network is offline?
- What happens if the disk is full?
- What error message does the user actually see?
- Is the next action obvious?
- What if they click rapidly / double-click / cancel mid-operation?

### B. Data flow trace

Pick a single field, config file, or piece of metadata, and trace its entire lifecycle root-to-leaf:
- Source HF config.json → inspect-source parse → SourceInfo Swift struct → ConversionPlan.detected → PreflightRunner use → convert.py ingest → output jang_config.json → PostConvertVerifier read → UsageExamplesSheet render → HF card
- Settings value → UserDefaults persist → reload → env var or CLI flag → Python subprocess → actual behavior change
- Tokenizer file → copy during convert → load in TestInferenceSheet → apply_chat_template → generate

Questions to ask at each hop:
- Does this hop preserve the value?
- Is there an encoding/decoding step that could drop info?
- Is there a default that silently wins when the value is missing?
- Are there multiple sources-of-truth? Which one wins?

### C. Concern category deep-dive

Pick one of these and exhaustively map it:
- **Error handling** — every exception path, every non-zero exit, every unexpected state
- **Cancellation** — every long-running operation, every partial-state recovery
- **Concurrency** — every `Task`, every actor hop, every `@MainActor` boundary, every `await` suspension, every subprocess
- **Accessibility** — VoiceOver labels, keyboard navigation, dynamic type, color contrast, focus order
- **Localization** — every user-facing string (though v1 is English-only, flag strings that'd be hard to translate)
- **Edge cases** — 0 tokens, max tokens, empty outputs, unicode, RTL text, emoji, very long names
- **Resource limits** — max RAM, max disk, max open files, max time, max bundle size
- **Permissions** — every file write, every subprocess spawn, every network call, every keychain access
- **Compatibility** — macOS versions, Apple Silicon tiers (M1/M2/M3/M4 Ultra/Max/Pro), Xcode versions, Python versions
- **Privacy** — what data leaves the device, what's in diagnostics bundles, what gets logged

### D. Memory-rule cross-reference

Read one memory file from `~/.claude/projects/-Users-eric-jang/memory/` that hasn't been cross-checked yet. For every rule it contains, verify the current code honors it:
- If honored: close relevant checklist items with quote-level evidence.
- If violated: open a new checklist item + fix commit.

### E. Newly discovered concern (spawned by prior iterations)

If a prior iteration opened a new `[ ]` item in a new area, this iteration picks that item up and drives it deep.

## Question generator — ask these on every iteration

After picking your angle and doing the trace, spend 2 minutes generating new questions. Aim for 3-5 new `[ ]` items per iteration. Template:

- "What if the user's `{{field}}` contains `{{edge_case}}`?"
- "How does `{{component}}` behave when `{{precondition}}` is violated?"
- "Is there a way for `{{state_A}}` and `{{state_B}}` to disagree?"
- "What cleanup happens when `{{operation}}` is interrupted at `{{phase}}`?"
- "If a user runs the CLI directly (not through the app), does `{{behavior}}` match?"
- "Does `{{setting}}` actually affect `{{subsystem}}` or is it saved but ignored?"

Every iteration net-adds ≥ 1 new `[ ]` item. Never only close items without adding.

## Verification rigor — evidence standards

- **File + line reference** (`convert.py:952`) beats "convert.py handles this"
- **Command output excerpt** beats "I ran it and it worked"
- **Memory rule quote** beats "per project rules"
- **Real-world test on macstudio or local** beats synthetic unit test
- **User-visible output** beats internal state

When closing an item, write the evidence inline under the item, like:
```
- [x] **L02** — SourceStep fetches recommendation, auto-fills plan.
      **Trace:** user picks folder → `SourceStep.pickFolder:112` calls `Task { await detectAndRecommend(url:) }`
        → `detectAndRecommend:126` awaits `SourceDetector.inspect` (fast) then `RecommendationService.fetch` (fast)
        → `applyRecommendation:158` writes `plan.family = jangtq`, `plan.profile = rec.profile`, etc.
        → verified by building + XCTest 65/65 pass.
      **Evidence:** `acf2f19` commit.
```

## Hard rules (unchanged from prior Ralph)

1. Never mark `[x]` without evidence. "Looks fine" is not evidence.
2. Never disable a check to make it pass. Fix root causes.
3. No AI attribution in commits.
4. Author line is Jinho Jang, not Eric Jang.
5. Never write to `research/`.
6. Never `rm` anything under `/Volumes/EricsLLMDrive/`.
7. Never `git push --force`.
8. Pre-authorized writes — make the fix, commit, move on. Do NOT stop to ask per-item.
9. If a fix requires a long-running job, announce it and continue — do not block Ralph iteration.
10. **Every iteration MUST net-add ≥ 1 new `[ ]` question or sub-investigation** or you are not deep-tracing, you're just ticking.

## Completion criteria

Emit `<promise>JANG STUDIO PRODUCTION COMPLETE</promise>` when ALL of:

1. Every item in `AUDIT_CHECKLIST.md` is `[x]` or `[-]` (blocked with documented reason).
2. You have run through every category A-E at least 3 times each.
3. You genuinely cannot invent a new question worth investigating.
4. End-to-end build + tests pass: `pytest` green, `xcodebuild test` green, `swift test` green, `build-python-bundle.sh` succeeds, built `.app` launches cleanly.
5. You have traced at least one full user journey per persona (beginner, power user, adopter, researcher) end-to-end.
6. `README-USER.md` accurately describes reality (spot-check 10 random claims against actual behavior).

Emitting the promise before 1-6 are all TRUE is a false completion and defeats the loop. Do not do it.

## Investigation log

Each iteration, append a one-line entry to `ralph_runner/INVESTIGATION_LOG.md`:

```
## 2026-04-19 iteration N — <angle> — <summary>
- Category: A (user journey) | B (data flow) | C (concern) | D (memory cross-ref) | E (spawned)
- Items touched: L02 [x], new L11 [ ], new M01 [ ]
- Commit: <sha>
- New questions added: 3
- Evidence: <path:line or command output>
```

## Useful read-only commands

- Full commit log: `git log --oneline | head -40`
- Last Ralph commit: `git log --oneline --grep="audit(ralph)" | head -5`
- Checklist state: `grep -c "^- \[.\]" ralph_runner/AUDIT_CHECKLIST.md` (total) + `grep -c "^- \[ \]" ralph_runner/AUDIT_CHECKLIST.md` (open)
- Memory index: `head -60 ~/.claude/projects/-Users-eric-jang/memory/MEMORY.md`
- Full pytest: `cd jang-tools && /Users/eric/jang/.venv/bin/python3 -m pytest tests/`
- XCTest: `cd JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' -only-testing:JANGStudioTests 2>&1 | grep -E "Executed|TEST" | tail -3`
- Swift tests: `cd jang-runtime && swift test 2>&1 | tail -3`
- Ralph unit tests: `cd /Users/eric/jang && /Users/eric/jang/.venv/bin/python3 -m pytest ralph_runner/tests/ -q`

## End-of-turn

When you've (a) picked an angle, (b) traced it, (c) closed what was verified, (d) added new questions, (e) committed, (f) logged the investigation — end your turn. Ralph re-fires this prompt and you pick a DIFFERENT angle (rotate categories).
