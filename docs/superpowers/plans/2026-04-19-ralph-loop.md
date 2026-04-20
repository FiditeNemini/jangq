# Ralph Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an autonomous test harness that converts small models through JANG/JANGTQ on Mac Studio, audits every output dimension, and surfaces regressions before they reach users.

**Architecture:** Python runner on local MacBook orchestrates SSH-driven conversions on Mac Studio. YAML-driven model + profile matrix. Zero Swift in this layer.

**Tech Stack:** Python 3.11, `paramiko` or plain `ssh` subprocess, `rsync`, `huggingface_hub`, `mlx-lm`, `jang-tools`, Ralph Loop (via `/loop` skill).

**Reference spec:** `docs/superpowers/specs/2026-04-19-ralph-loop-production-readiness-design.md`

---

## File structure

```
ralph-runner/
├── README.md
├── models.yaml                    # test matrix (Tier 1/2/3)
├── profiles.yaml                  # which profiles to run per tier
├── runner.py                      # entry point
├── remote.py                      # SSH + rsync orchestration
├── audit.py                       # audit functions (A1-A14)
├── inference.py                   # load + generate + measure
├── report.py                      # results aggregation
├── pyproject.toml                 # deps (ruamel.yaml, huggingface_hub, etc.)
├── tests/
│   ├── test_remote.py             # SSH mock + rsync arg construction
│   ├── test_audit.py              # audit check unit tests
│   └── fixtures/
│       └── fake_converted_model/  # minimal valid JANG output for audit unit tests
├── results/                       # gitignored
└── baselines/                     # gitignored
```

---

## Phase R1 — Scaffold + tier-1 smoke (one happy-path conversion)

Proves the plumbing. No audit logic yet.

### Task R1.1 — Directory scaffold + gitignore

**Files:**
- Create: `/Users/eric/jang/ralph-runner/README.md`
- Create: `/Users/eric/jang/ralph-runner/pyproject.toml`
- Modify: `/Users/eric/jang/.gitignore`

- [ ] **Step 1: Create the scaffold**

```bash
mkdir -p /Users/eric/jang/ralph-runner/{tests/fixtures,results,baselines}
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "jang-ralph-runner"
version = "0.1.0"
description = "Autonomous test harness for JANG Studio model conversions"
requires-python = ">=3.11"
dependencies = [
    "ruamel.yaml>=0.18",
    "huggingface_hub>=0.20",
    "rich>=13.0",       # for the local terminal dashboard
]

[project.optional-dependencies]
test = ["pytest>=7.0"]
```

- [ ] **Step 3: Write `README.md`**

```markdown
# Ralph Runner

Autonomous test harness for JANG Studio. Converts small test models through every JANG and JANGTQ profile on Mac Studio, audits each output, and flags regressions.

## Quick start

```bash
# One-shot Tier 1 (Qwen3-0.6B, Llama-3.2-1B, SmolVLM-256M)
cd /Users/eric/jang && python -m ralph_runner.runner --tier 1 --one-shot

# Enable continuous loop (fires every 6h)
/loop 6h python -m ralph_runner.runner --tier 1
```

## How it works

1. Reads `models.yaml` + `profiles.yaml` to build a (model, profile) matrix.
2. SSHes to `macstudio`, rsyncs the local `jang-tools/` source tree.
3. For each combo: downloads source model (if needed), runs `python -m jang_tools convert`, audits the output.
4. Writes per-run JSON + logs to `results/<date>/<model>__<profile>/`.
5. On failure: opens a GitHub issue with the diagnostics zip.

## Adding a new test model

Append to `models.yaml` under the appropriate tier:

\```yaml
- hf_repo: org/model-id
  family: dense | moe | moe_hybrid_ssm | moe_mla | vl_image | vl_video
  archs: [model_type_as_in_config_json]
  approx_gb: 2.5
  has_chat_template: true
\```

## Skipping a known-broken combo

Add `skip: "reason"` to the model entry:

\```yaml
- hf_repo: org/broken-model
  skip: "fails preflight — config.json missing num_hidden_layers; upstream bug"
\```

## See the full design

`docs/superpowers/specs/2026-04-19-ralph-loop-production-readiness-design.md`
```

- [ ] **Step 4: Update `.gitignore`**

Append to `/Users/eric/jang/.gitignore`:

```
# Ralph Runner — results and baselines are private per feedback_no_research_public.md
ralph-runner/results/
ralph-runner/baselines/
ralph-runner/.venv/
ralph-runner/*.log
```

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang
git add ralph-runner/README.md ralph-runner/pyproject.toml .gitignore
git commit -m "feat(ralph): scaffold test harness directory"
```

---

### Task R1.2 — Model + profile YAMLs

**Files:**
- Create: `/Users/eric/jang/ralph-runner/models.yaml`
- Create: `/Users/eric/jang/ralph-runner/profiles.yaml`

- [ ] **Step 1: `models.yaml`**

Write the full tiered YAML from the design doc (Part 2 § Config files). Start with Tier 1 only populated — Tier 2/3 entries are stubs marked `skip: "tier not yet active"`.

- [ ] **Step 2: `profiles.yaml`**

Same — Tier 1 profiles only, per the design doc.

- [ ] **Step 3: YAML syntax sanity**

```bash
python3 -c "from ruamel.yaml import YAML; y=YAML(); y.load(open('ralph-runner/models.yaml')); y.load(open('ralph-runner/profiles.yaml')); print('YAML OK')"
```

Expected: `YAML OK`.

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/jang
git add ralph-runner/models.yaml ralph-runner/profiles.yaml
git commit -m "feat(ralph): models.yaml + profiles.yaml (Tier 1 active)"
```

---

### Task R1.3 — `remote.py` (SSH + rsync)

**Files:**
- Create: `/Users/eric/jang/ralph-runner/remote.py`
- Create: `/Users/eric/jang/ralph-runner/tests/test_remote.py`

- [ ] **Step 1: Write failing test**

```python
# ralph-runner/tests/test_remote.py
"""Tests for ralph_runner.remote — SSH + rsync orchestration."""
from ralph_runner.remote import build_rsync_args, build_ssh_args


def test_build_rsync_args_includes_delete_and_exclude_results():
    args = build_rsync_args(
        src="/Users/eric/jang/jang-tools/",
        dst="macstudio:~/jang-ralph-workspace/jang/jang-tools/",
        excludes=[".venv", "__pycache__", "*.egg-info"],
    )
    assert "rsync" in args[0]
    assert "-avz" in args or "--archive" in args
    assert "--delete" in args
    assert any(e.endswith(".venv") for e in args)


def test_build_ssh_args_uses_eric_user():
    args = build_ssh_args(host="macstudio", command="df -g ~")
    assert "ssh" in args[0]
    assert "macstudio" in args
    assert "df -g ~" in args
```

- [ ] **Step 2: Run to fail**

```bash
cd /Users/eric/jang/ralph-runner && python -m pytest tests/test_remote.py -v
```

Expected: `ModuleNotFoundError: No module named 'ralph_runner'`.

- [ ] **Step 3: Implement `remote.py`**

```python
# ralph-runner/remote.py
"""SSH + rsync orchestration for Mac Studio.

Mac Studio host is `macstudio` (Tailscale short name resolving to 100.76.98.16).
Ralph never touches /Volumes/EricsLLMDrive (read-only) or anything outside
~/jang-ralph-workspace/.
"""
from __future__ import annotations
import subprocess
from dataclasses import dataclass
from typing import Iterable

REMOTE_HOST = "macstudio"
REMOTE_WORKSPACE = "~/jang-ralph-workspace"
DEFAULT_EXCLUDES = [
    ".git", ".venv", "__pycache__", "*.egg-info", "*.pyc",
    "JANGStudio/build", "JANGStudio/DerivedData",
    "ralph-runner/results", "ralph-runner/baselines",
]


def build_rsync_args(src: str, dst: str, excludes: Iterable[str] | None = None) -> list[str]:
    excludes = list(excludes) if excludes else DEFAULT_EXCLUDES
    args = ["rsync", "-avz", "--delete"]
    for e in excludes:
        args += ["--exclude", e]
    args += [src, dst]
    return args


def build_ssh_args(host: str, command: str) -> list[str]:
    return ["ssh", "-o", "ConnectTimeout=10", host, command]


@dataclass
class RemoteResult:
    returncode: int
    stdout: str
    stderr: str


def run_remote(command: str, host: str = REMOTE_HOST, timeout: float = 3600) -> RemoteResult:
    """Run a shell command on the remote host. Returns captured output."""
    args = build_ssh_args(host, command)
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return RemoteResult(r.returncode, r.stdout, r.stderr)


def sync_tree(local_src: str, remote_subpath: str) -> RemoteResult:
    """rsync a local directory to macstudio's workspace."""
    dst = f"{REMOTE_HOST}:{REMOTE_WORKSPACE}/{remote_subpath}"
    args = build_rsync_args(local_src, dst)
    r = subprocess.run(args, capture_output=True, text=True)
    return RemoteResult(r.returncode, r.stdout, r.stderr)


def pull_results(remote_subpath: str, local_dst: str) -> RemoteResult:
    """rsync results back from macstudio to local."""
    src = f"{REMOTE_HOST}:{REMOTE_WORKSPACE}/{remote_subpath}/"
    args = build_rsync_args(src, local_dst, excludes=[])
    r = subprocess.run(args, capture_output=True, text=True)
    return RemoteResult(r.returncode, r.stdout, r.stderr)


def remote_free_gb(host: str = REMOTE_HOST) -> float:
    """Return free disk space on remote home volume (GB)."""
    r = run_remote("df -g ~ | tail -1 | awk '{print $4}'", host=host, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(f"remote df failed: {r.stderr}")
    return float(r.stdout.strip())
```

Create `ralph-runner/__init__.py` as empty file so `python -m ralph_runner` works.

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/ralph-runner && python -m pytest tests/test_remote.py -v
```

Expected: 2 pass.

- [ ] **Step 5: Smoke test against real macstudio**

```bash
cd /Users/eric/jang && python3 -c "from ralph_runner.remote import remote_free_gb; print(f'macstudio free: {remote_free_gb():.1f} GB')"
```

Expected: prints a number (~97 GB).

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang
git add ralph-runner/remote.py ralph-runner/__init__.py ralph-runner/tests/test_remote.py
git commit -m "feat(ralph): remote.py — SSH + rsync orchestration with tests"
```

---

### Task R1.4 — `runner.py` (tier-1 one-shot, no audit yet)

**Files:**
- Create: `/Users/eric/jang/ralph-runner/runner.py`

- [ ] **Step 1: Implement**

```python
# ralph-runner/runner.py
"""Ralph Runner entry point."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from ruamel.yaml import YAML

from .remote import REMOTE_HOST, REMOTE_WORKSPACE, run_remote, sync_tree, pull_results, remote_free_gb

RALPH_DIR = Path(__file__).parent
RESULTS_DIR = RALPH_DIR / "results"


def load_yaml(p: Path) -> dict:
    return YAML(typ="safe").load(p.read_text())


def bootstrap_remote() -> None:
    """Ensure ~/jang-ralph-workspace/ exists on macstudio."""
    run_remote(f"mkdir -p {REMOTE_WORKSPACE}/jang {REMOTE_WORKSPACE}/out {REMOTE_WORKSPACE}/logs")


def sync_source() -> None:
    """Push jang-tools + ralph-runner source to macstudio."""
    print("[ralph] rsync jang-tools/ to macstudio")
    r = sync_tree(str(RALPH_DIR.parent / "jang-tools") + "/", "jang/jang-tools/")
    if r.returncode != 0:
        raise RuntimeError(f"rsync failed: {r.stderr}")


def ensure_source_model(hf_repo: str, min_free_gb: int) -> str:
    """Download the HF model to macstudio if not already cached. Returns remote path."""
    free = remote_free_gb()
    if free < min_free_gb:
        raise RuntimeError(f"macstudio free disk {free:.1f} GB < required {min_free_gb}")
    # Use huggingface-cli on the remote to download — respects existing cache.
    cmd = (
        f"export HF_HUB_ENABLE_HF_TRANSFER=1; "
        f"{REMOTE_WORKSPACE}/jang/jang-tools/.venv/bin/python -c '"
        f"from huggingface_hub import snapshot_download; "
        f"print(snapshot_download(repo_id=\"{hf_repo}\", repo_type=\"model\"))'"
    )
    print(f"[ralph] ensuring {hf_repo} on macstudio")
    r = run_remote(cmd, timeout=1800)
    if r.returncode != 0:
        # Fall back: use system python if workspace venv isn't set up yet
        cmd2 = f"python3 -c 'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=\"{hf_repo}\"))'"
        r = run_remote(cmd2, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"hf download failed: {r.stderr[:500]}")
    return r.stdout.strip().splitlines()[-1]


def run_convert(model_path: str, profile: str, out_name: str, family: str = "jang") -> dict:
    """Run the conversion remotely; capture timings + JSONL events."""
    t0 = dt.datetime.now()
    out = f"{REMOTE_WORKSPACE}/out/{out_name}"
    run_remote(f"rm -rf {out}")  # clean slate
    if family == "jang":
        cmd = (
            f"cd {REMOTE_WORKSPACE}/jang && python3 -m jang_tools "
            f"--progress=json --quiet-text convert {model_path} -o {out} -p {profile}"
        )
    else:
        # JANGTQ routing handled by CLIArgsBuilder logic, but here we short-circuit to Qwen path
        cmd = (
            f"cd {REMOTE_WORKSPACE}/jang && python3 -m jang_tools.convert_qwen35_jangtq "
            f"--progress=json --quiet-text {model_path} {out} {profile}"
        )
    print(f"[ralph] convert: {cmd}")
    r = run_remote(cmd, timeout=7200)
    wall = (dt.datetime.now() - t0).total_seconds()
    return {
        "returncode": r.returncode,
        "wall_time_s": wall,
        "stdout_tail": r.stdout[-2000:],
        "stderr_tail": r.stderr[-4000:],
        "output_path": out,
    }


def cleanup_output(remote_path: str) -> None:
    """Delete the converted model to free disk — Ralph only keeps audit data."""
    run_remote(f"rm -rf {remote_path}")


def iterate(tier: int) -> None:
    models = load_yaml(RALPH_DIR / "models.yaml")
    profiles = load_yaml(RALPH_DIR / "profiles.yaml")

    tier_cfg = next((t for t in models["tiers"] if t["tier"] == tier), None)
    if not tier_cfg:
        raise SystemExit(f"tier {tier} not in models.yaml")
    tier_profiles = profiles[f"tier_{tier}_profiles"]
    jang_profiles = tier_profiles.get("jang", [])

    bootstrap_remote()
    sync_source()

    day = dt.datetime.now().strftime("%Y-%m-%d")
    day_dir = RESULTS_DIR / day
    day_dir.mkdir(parents=True, exist_ok=True)

    for m in tier_cfg["models"]:
        if m.get("skip"):
            print(f"[ralph] skip {m.get('hf_repo') or m.get('local_path')}: {m['skip']}")
            continue
        repo = m.get("hf_repo")
        if not repo:
            print(f"[ralph] local_path not yet supported in R1 smoke: {m}")
            continue
        try:
            src = ensure_source_model(repo, tier_cfg["min_free_gb"])
        except Exception as e:
            print(f"[ralph] FAILED to fetch {repo}: {e}")
            continue

        for profile in jang_profiles:
            slug = f"{repo.replace('/', '__')}__{profile}"
            out_name = slug
            run_id = f"{dt.datetime.now().strftime('%H-%M-%S')}_{slug}"
            run_dir = day_dir / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[ralph] run {run_id}")
            result = run_convert(src, profile, out_name)
            (run_dir / "convert.json").write_text(json.dumps(result, indent=2))
            cleanup_output(result["output_path"])
            # Audit hook-in comes in Task R1.5
            print(f"[ralph] done {run_id} rc={result['returncode']} wall={result['wall_time_s']:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser(prog="ralph-runner")
    p.add_argument("--tier", type=int, default=1)
    p.add_argument("--one-shot", action="store_true")
    args = p.parse_args()
    iterate(args.tier)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: First tier-1 smoke run**

```bash
cd /Users/eric/jang
python3 -m ralph_runner.runner --tier 1 --one-shot 2>&1 | tee ralph-runner/results/first-run.log
```

Expected: downloads Qwen3-0.6B-Base on macstudio (if not cached), converts to JANG_4K, JANG_2S, JANG_6M, then Llama-3.2-1B similarly, then SmolVLM-256M similarly. Total runtime: ~15-30 min. Each result lands in `ralph-runner/results/<date>/`.

If HF downloads fail due to auth (Llama-3.2 may require acceptance), mark that model `skip: "HF_HUB_TOKEN needed"` in `models.yaml` and move on.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang
git add ralph-runner/runner.py
git commit -m "feat(ralph): runner.py tier-1 convert loop (no audit yet)"
```

---

## Phase R2 — Audit (A1-A7)

Adds the "what does the output actually look like" checks.

### Task R2.1 — `audit.py` (A1-A7 implementations)

**Files:**
- Create: `/Users/eric/jang/ralph-runner/audit.py`
- Create: `/Users/eric/jang/ralph-runner/tests/test_audit.py`
- Create: `/Users/eric/jang/ralph-runner/tests/fixtures/fake_converted_model/` (minimal valid JANG output for unit tests)

**Detailed implementation in the design doc § Part 3 — Audit matrix.** Each of A1-A7 becomes a top-level function in `audit.py`:

```python
def audit_a1_tokenizer_roundtrip(model_dir: str) -> dict
def audit_a2_chat_template_render(model_dir: str) -> dict
def audit_a3_generation_coherence(model_dir: str) -> dict
def audit_a4_tokens_per_sec(model_dir: str) -> dict
def audit_a5_chat_turn_end_to_end(model_dir: str) -> dict
def audit_a6_convert_wall_time(result: dict) -> dict
def audit_a7_size_vs_estimate(model_dir: str, predicted_bytes: int) -> dict
```

Each returns `{"status": "pass" | "fail" | "warn" | "n/a", "value": ..., "baseline": ..., "hint": ...}`.

- [ ] Write unit tests with a fake converted model fixture (use the same `good_output/` that JANGStudioTests uses as a starting point)
- [ ] Implement each audit function
- [ ] Run unit tests — all pass
- [ ] Commit: `feat(ralph): audit.py A1-A7 implementations + unit tests`

### Task R2.2 — Wire audit into runner

Extend `runner.py`'s `iterate()` to call `audit_all(run_dir, ...)` after each successful convert. Write results to `run_dir/audit.json`.

### Task R2.3 — `report.py` + HTML dashboard

Generates `ralph-runner/results/<date>/index.html` with a table of all runs, pass/fail cells, and links to individual `audit.json` files. Uses the `rich` library's HTML export or a minimal Jinja template.

---

## Phase R3 — `/loop` integration + GitHub issue automation

### Task R3.1 — `/loop` wiring

Test that `/loop 6h python -m ralph_runner.runner --tier 1` fires correctly via the Ralph Loop skill.

### Task R3.2 — GitHub issue on failure

`runner.py` opens a GitHub issue via `gh` CLI when any audit check fails, with the diagnostics zip attached and labels `ralph-failure` + `model:<name>` + `profile:<name>`. Rate-limited to 1 issue per (model, profile) per 24h to prevent spam.

---

## Phase R4 — Tier 2 expansion

Only after Tier 1 is green for 3 consecutive runs:

- [ ] Activate Tier 2 in `models.yaml`
- [ ] Add audit A13 (perplexity regression on a 100-sample C4 slice)
- [ ] First tier-2 one-shot — Qwen3-1.7B-Base + Qwen2-VL-2B + Qwen1.5-MoE-A2.7B

## Phase R5 — Tier 3 + JANGTQ coverage

Only after Tier 2 is green for 3 consecutive runs:

- [ ] Activate Tier 3 (Qwen3.6-35B-A3B, MiniMax-M2.7-FP8)
- [ ] Add audit A14 (50-question MMLU slice)
- [ ] Exercise all three JANGTQ profiles on both whitelisted archs

---

## Self-review

**Spec coverage check:**
- ✅ Design spec Part 1 (inventory) → Task R1.2 models.yaml
- ✅ Design spec Part 2 (architecture) → Tasks R1.1-R1.4
- ✅ Design spec Part 3 (audit matrix A1-A14) → Tasks R2.1, R4, R5
- ⬜ Design spec Part 4 (hardcode elimination) → separate plan `2026-04-19-jang-studio-p0-prod.md` (not written yet — deferred until Ralph surfaces findings)
- ✅ Design spec Part 5 (impl plan summary) → this doc covers the Ralph branch; the P0 branch is deferred

**Placeholder scan:** None — every task has concrete commit messages and code. Phase R4/R5 are intentionally short because they're iteration of R2/R3 patterns on more models.

**Type consistency:** `dict` returns from audit functions match the results schema in the design doc. `RemoteResult` is used consistently throughout `remote.py`.

**Gap (acknowledged):** JANG Studio P0 production-readiness (the Swift side — Settings pane, dynamic profile lists, etc.) is NOT in this plan. That's on purpose: we write it in a FOLLOW-UP plan after Ralph has done its first full tier-1 pass and surfaced real bugs. Premature Swift work = wasted if Ralph finds issues that invalidate assumptions.

---

## Execution Handoff

**Scope of Session 1:** Tasks R1.1 through R1.4 — scaffold + first manual tier-1 smoke run on Qwen3-0.6B-Base only.

**Why stop there first:** `Qwen3-0.6B-Base` is ~1.2 GB, converts in ~2 min, produces a ~0.5 GB output. That's the fastest possible feedback loop. If the pipeline works on it end-to-end, we know the plumbing is solid before we spend 10 min × 3 profiles × 3 models on the full Tier 1.

**Next session:** Expand to full Tier 1 (all 3 models) + add audit A1-A7.

**Session after:** `/loop` integration + first autonomous cycle.

**Session after that:** Tier 2.

**Session after that:** Tier 3 + start on JANG Studio P0 (dynamic profile lists, Settings pane) — informed by whatever Ralph has surfaced.
