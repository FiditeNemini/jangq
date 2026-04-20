# Ralph Loop — JANG Studio Test Harness

You are iterating the JANG Studio test harness at `/Users/eric/jang/ralph_runner/`.

## Core rule

A combo is NOT green until both:
1. **Convert** succeeds (runner reports `COMBO <slug> GREEN`)
2. **Audit** passes A1-A18 (see "Audit matrix" below)

If convert succeeds but any required audit fails, the combo's status must be flipped back to `failed` with `inference_error` or similar reason. A converted model that can't generate coherent output is a shipping bug, not a success.

## What to do this iteration

1. Print current status:
   ```
   cd /Users/eric/jang && /Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --status
   ```

2. Interpret the output:

   **Case A — `ALL GREEN` at the bottom:**
   - Every combo converted AND passed all required audits.
   - Emit `<promise>TIER 1 GREEN</promise>` and exit.

   **Case B — `NONE PENDING` but some `failed`:**
   - Pick exactly ONE failed combo (start with the lowest-tier + smallest model).
   - Read its artifacts at `ralph_runner/results/<date>/<slug>/`:
     - `convert.json` — stdout/stderr tails from the conversion
     - `audit.json` — per-row audit results (if the convert succeeded)
   - Diagnose:
     - **Convert failure** → read `convert.json` stderr. Common causes: `jang_tools` import error on remote, HF auth required, disk full, architecture unrecognized, OOM.
     - **Audit failure** → read `audit.json`. Identify which audit row failed (A1-A18).
   - Fix:
     - **`jang_tools` bug** → edit `jang-tools/jang_tools/*.py` locally, commit with exact message `fix(jang-tools): <description>`. Next iteration's rsync picks up the fix.
     - **`ralph_runner` bug** → edit the harness, commit `fix(ralph): <description>`.
     - **Remote env bug** → fix on macstudio (e.g., `ssh macstudio 'pip install foo'`), document in `ralph_runner/ENV.md`.
     - **Intrinsic upstream bug** (model config broken, unsupported arch) → mark `skip: "<reason>"` in `models.yaml`, commit `feat(ralph): skip <model> — <reason>`.
   - Reset and re-seed: `python3 -m ralph_runner.runner --reset && python3 -m ralph_runner.runner --tier 1`.
   - DO NOT emit a promise yet; let the next iteration pick the next pending combo.

   **Case C — `pending` combos exist:**
   - Run:
     ```
     cd /Users/eric/jang && /Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --next
     ```
   - This blocks for 2-20 minutes depending on the model. Be patient.
   - When it prints `COMBO <slug> GREEN` (convert succeeded), the runner automatically runs A1-A18. If any required audit fails, it flips the combo back to `failed`.
   - Do not emit a promise; let the next iteration check status.

## Audit matrix (A1-A18)

Runner must execute these after every successful convert. Each returns `pass` / `warn` / `fail` / `n/a` with supporting data.

### A1-A14 — Structural (from parent spec)

Implemented in `ralph_runner/audit.py` — tests the output artifacts themselves.

- A1 — Tokenizer round-trip (encode/decode 20 test strings incl. unicode + whitespace)
- A2 — Chat template render (if source had one)
- A3 — Generation coherence ("capital of France" → contains "Paris")
- A4 — Tokens/sec throughput (no >30% regression vs baseline)
- A5 — Chat-turn end-to-end (2-turn convo, no infinite loop)
- A6 — Convert wall time (no >50% regression)
- A7 — Size vs estimate (within 15%)
- A8 — Tool/reasoning parser preservation (if source had `tool_call_parser`, `reasoning_parser`, `enable_thinking`, `chat_template_kwargs`)
- A9 — Special tokens preservation (source → output JSON structural equality)
- A10 — JANGTQ codebook metadata (`jang_config.quantization.tq_*` present for JANGTQ)
- A11 — VL preprocessor functional (`AutoProcessor` loads + runs on test image)
- A12 — Video preprocessor functional (loads + runs on test frames)
- A13 — Perplexity regression (tier 2+; 100-sample slice, within 15%)
- A14 — MMLU mini (tier 3 only; 50-question subset, within 5 pts)

### A15-A18 — Adoption (from addendum 2026-04-19)

These catch "convert worked but the model is unusable" — the sneakiest failure mode. Required for all tiers.

- **A15 — Inference works via bundled runtime.** Load via `jang_tools.loader.load_model` (dense/MoE) or `jang_tools.load_jangtq_vlm` (VL/video). Generate 20 tokens from `"Hello, how are you?"`. Assert: no exception, non-empty output, at least one printable non-whitespace character. **Required for every combo.**
- **A16 — Chat template functionally applies.** If model has a chat template (`has_chat_template: true` in `models.yaml`): apply a 2-turn conversation via `tokenizer.apply_chat_template`, generate a response, assert each role marker appears in the rendered prompt. Skipped otherwise.
- **A17 — Model card generatable.** Run `python3 -m jang_tools modelcard --model <out> --json`. Assert valid JSON, contains `license`, `base_model`, `quantization_config.family`, `quantization_config.profile`, `quantization_config.actual_bits`. **Required.**
- **A18 — Usage examples generatable.** For each lang in `[python, swift, server, hf]`: `python3 -m jang_tools examples --model <out> --lang <lang> --json`. Assert valid JSON with non-empty `snippet`. For Python lang: `compile(snippet, '<eval>', 'exec')` must succeed. **Required.**

If A15, A17, or A18 fail: combo flips to `failed` with reason `audit_<row>_<detail>`.

## Completion criteria

Emit `<promise>TIER 1 GREEN</promise>` if and only if:
- Every active (non-skipped) combo in `state.json` has status `green`.
- Runner's `--status` output ends with the literal string `ALL GREEN`.

## Hard rules

1. NEVER `rm` anything under `/Volumes/EricsLLMDrive/` — that's read-only source models.
2. NEVER `git push` or `git push --force` without explicit user confirmation.
3. NEVER disable or skip an audit check to force a combo green. If A15 fails, fix the root cause.
4. NEVER add AI co-author trailers to commit messages. Strict Eric rule.
5. NEVER edit `main`-tracking commits; only edit the working branch.
6. If stuck on the same combo for 3 consecutive iterations, emit `<promise>STUCK ON <slug>: <summary></promise>` and stop so a human can intervene.
7. If macstudio is unreachable (runner prints `BLOCKED:`), emit `<promise>BLOCKED: macstudio unreachable</promise>` and stop.
8. If macstudio disk free drops below 10 GB, emit `<promise>BLOCKED: macstudio low disk</promise>` and stop.
9. If the `jang_tools` source tree on macstudio is older than the local HEAD (detectable via SHA check), the runner must rsync before running the next combo. This already happens automatically — don't bypass it.

## Useful commands

| Purpose | Command |
|---|---|
| See status | `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --status` |
| Run next pending combo | `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --next` |
| Reset state | `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --reset && /Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --tier 1` |
| Check macstudio reachability | `ssh -o ConnectTimeout=5 macstudio 'echo ok'` |
| Check macstudio disk | `ssh macstudio 'df -g ~ \| tail -1'` |
| Read last convert log | `ls -1t ralph_runner/results/*/*/convert.json \| head -1 \| xargs cat` |
| Read last audit results | `ls -1t ralph_runner/results/*/*/audit.json \| head -1 \| xargs cat` |
| Verify `jang_tools` on macstudio | `ssh macstudio 'cd /Users/eric/jang-ralph-workspace/jang/jang-tools && python3 -m jang_tools --version'` |

## Architecture reminder

- **Local (MacBook):** Ralph Loop runs Claude here. Reads state, invokes `runner.py --next`.
- **Remote (macstudio, 100.76.98.16 / erics-mac-studio.local):** Does the actual converting + inference audits. Python runtime lives under `/Users/eric/jang-ralph-workspace/jang/`.
- **Source of truth:** `jang-tools/` source tree is rsynced fresh every iteration. Do NOT pip install anything globally on macstudio unless documenting in `ralph_runner/ENV.md`.
- **State persistence:** `ralph_runner/results/state.json` is the single source of truth for combo status. Never hand-edit; always mutate via runner.py.

## Model capabilities to verify per combo

For each combo, the audit must confirm these work (where applicable to the source model):

- **Chat template** — single-turn render + generate (A2 + A16)
- **Tool calling** — if source has `tool_call_parser`, A8 verifies preservation + generates a tool-call
- **Reasoning** — if source has `reasoning_parser` or `enable_thinking`, A8 verifies preservation + generates with `<think>` markers
- **Vision (image)** — if `detected.isVL`, A11 loads test image + generates a description
- **Vision (video)** — if `detected.isVideoVL`, A12 loads test video frames + generates
- **Special tokens** — A9 preserves source→output

## What "production ready" means for this iteration

Not "tests pass." Rather: an outside developer who has never seen JANG can:
1. Convert a model with JANG Studio
2. Click **Test Inference** and have a working chat in under 30 seconds
3. Copy the **Python snippet** and have it run in their own code on the first try
4. Copy the **Swift snippet** and have it compile in a fresh Xcode project
5. Publish to HuggingFace with auto-generated model card that passes HF's linter
6. Hand the HF URL to another developer who can consume it in their framework

If Ralph ever sees a combo where convert passed but this end-to-end flow would break (A15 fails, A17 emits garbage JSON, A18 emits un-compilable Python), it must flip the combo to `failed` and fix the root cause.
