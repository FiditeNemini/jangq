# Ralph Loop — JANG Studio Test Harness

You are iterating the JANG Studio test harness at `/Users/eric/jang/ralph_runner/`.

## What to do this iteration

1. Run this command to see the status:
   ```bash
   cd /Users/eric/jang && /Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --status
   ```

2. If status shows `ALL GREEN` at the bottom:
   - Emit `<promise>TIER 1 GREEN</promise>` and exit.

3. If status shows `NONE PENDING` but there are `failed` combos:
   - Pick ONE failed combo.
   - Read its logs at `ralph_runner/results/<date>/<slug>/convert.json`.
   - Diagnose the failure:
     - If it's a bug in `jang-tools` → fix it, commit, rsync will pick up on next iteration, then `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --reset` and re-activate the tier.
     - If it's a test-harness bug (in `ralph_runner/runner.py` or `remote.py`) → fix, commit, reset state.
     - If it's an environment bug (macstudio disk full, HF auth needed) → fix the environment, reset state.
     - If it's intrinsic to the model (e.g., upstream config is malformed) → mark `skip: "<reason>"` in `models.yaml`, reset state.
   - After fixing, trigger another iteration.

4. If status has `pending` combos:
   - Run: `cd /Users/eric/jang && /Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --next`
   - Wait for it to complete (this can take 2-20 minutes depending on the model).
   - The command prints `COMBO <slug> GREEN` or `COMBO <slug> FAILED: ...` when done.
   - Don't emit any promise yet; let the next iteration check status.

## Completion criteria

Emit `<promise>TIER 1 GREEN</promise>` when and ONLY when status shows `ALL GREEN` at the bottom of `--status` output.

## Hard rules

- NEVER `rm` anything under `/Volumes/EricsLLMDrive/` — that's read-only source models.
- NEVER `git push` or `git force-push` without explicit user confirmation.
- NEVER disable or skip audit checks to make something "pass" — if a check fails, fix the root cause.
- If you get stuck for 3 iterations on the same combo, emit `<promise>STUCK ON <slug>: <summary></promise>` and stop.
- If macstudio becomes unreachable (`BLOCKED:` output), emit `<promise>BLOCKED: macstudio unreachable</promise>` and stop.
- NEVER add AI co-author trailers to commit messages (strict Eric rule).

## Useful commands

- See status: `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --status`
- Run next combo: `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --next`
- Reset state: `/Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --reset && /Users/eric/jang/.venv/bin/python3 -m ralph_runner.runner --tier 1`
- Check macstudio disk: `ssh macstudio df -g ~ | tail -1`
- See recent convert log: `ls -1t ralph_runner/results/*/*/convert.json | head -1 | xargs cat`
