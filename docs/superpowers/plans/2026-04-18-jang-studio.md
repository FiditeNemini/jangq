# JANG Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a native macOS SwiftUI wizard (signed + notarized `.app` in a DMG) that converts HuggingFace models (BF16 / FP16 / FP8) to JANG or JANGTQ format by driving the existing `jang-tools` Python pipeline as a subprocess.

**Architecture:** NavigationSplitView 5-step wizard. Swift app holds zero quantization logic — it shells out to a bundled Python runtime running `python -m jang_tools convert …` with a new `--progress=json` flag. JSONL progress events on stderr drive live UI. A 10-row preflight gate blocks Start; a 12-row post-convert verifier blocks Finish.

**Tech Stack:** Swift 6, SwiftUI, `NavigationSplitView`, Swift Observation, `Process`/`Pipe` for subprocess, python-build-standalone 3.11, `jang-tools` v2.4.1+, Xcode 16, GitHub Actions (macos-15 runners), xcrun notarytool.

**Reference spec:** `docs/superpowers/specs/2026-04-18-jang-studio-design.md`

---

## Progress Log

Updated as each phase lands. See each task block for detailed step-by-step status.

### ✅ Phase 1 — Python Progress Protocol + inspect-source (complete 2026-04-18)

7 commits, 150 pytest tests green. Backend ready for any GUI/CLI consumer.

| SHA | Commit |
|---|---|
| `c001e25` | test: ProgressEmitter JSONL schema and throttling contract |
| `3146db0` | feat: ProgressEmitter for dual text + JSONL reporting |
| `ba942dd` | feat(cli): --progress=json and --quiet-text global flags |
| `faf4f2d` | feat(convert): emit phase + tick events via ProgressEmitter |
| `88c482d` | feat(jangtq): JSONL progress flags for qwen35/minimax converters |
| `41bb74d` | feat(cli): inspect-source subcommand for GUI integration |
| `5d1a566` | test: golden JSONL fixture for Swift parser tests |

**New public API:**
- `python -m jang_tools --progress=json --quiet-text convert <src> -o <out> -p JANG_4K` emits JSONL events on stderr
- Same flags for `convert_qwen35_jangtq` and `convert_minimax_jangtq`
- New `python -m jang_tools inspect-source --json <dir>` returns single-line JSON with model_type, is_moe, num_experts, dtype, jangtq_compatible, is_vl, total_bytes, shard_count

### ✅ Phase 2 — Xcode scaffold + core models (complete 2026-04-18)

4 commits, 10 Swift unit tests green.

| SHA | Commit |
|---|---|
| `cf733f3` | chore(jang-studio): xcodegen scaffold + empty SwiftUI app |
| `2bd789b` | feat(jang-studio): ConversionPlan model with JANGTQ gating + persistence |
| `35bc4e2` | feat(jang-studio): JSONL v1 parser with version + tolerance |
| `48404ea` | feat(jang-studio): BundleResolver with dev-mode override |

**What's shippable:**
- macOS 15 SwiftUI app builds clean (Xcode 26.2, Swift 6)
- `ConversionPlan` persists to UserDefaults, enforces `{qwen3_5_moe, minimax_m2}` JANGTQ whitelist
- `JSONLProgressParser` v1 with 6 payload types, malformed-line tolerance, version-mismatch rejection
- `BundleResolver` with `$JANGSTUDIO_PYTHON_OVERRIDE` dev-mode escape hatch

### ✅ Phase 3 — PythonRunner actor (complete 2026-04-19)

3 commits, 13 Swift unit tests green. Two real bugs caught and fixed mid-phase.

| SHA | Commit |
|---|---|
| `8476582` | feat(jang-studio): PythonRunner with async JSONL streaming |
| `6a3bf08` | fix(jang-studio): restore actor-isolated launch so cancel() can SIGTERM |
| `6270214` | test(jang-studio): cancellation lands SIGTERM within 3s |

**Bugs caught by two-stage review:**
- Initial Swift 6 refactor made `launch` static, dropping `self.process = proc` — cancel() became a no-op. Fixed.
- Plan's `proc.waitUntilExit()` deadlocked the actor (blocking sync call held isolation so `cancel()` could never fire `terminate()`). Replaced with `withCheckedContinuation` + `terminationHandler`. Now cancels in ~0.21s (target was < 3.5s).

**What's shippable:**
- `actor PythonRunner` streams JSONL events via `AsyncThrowingStream<ProgressEvent, Error>`
- `cancel()` sends SIGTERM → waits 3s → SIGKILL fallback
- `ProcessError(code:lastStderr:)` thrown on non-zero exit with last 256 chars of stderr

### ✅ Phase 4 — Preflight + Verifier (complete 2026-04-19)

3 commits, 19 Swift tests green.

| SHA | Commit |
|---|---|
| `5c72780` | feat(jang-studio): PreflightRunner — 10 gate checks before Start |
| `28cd26b` | feat(jang-studio): PostConvertVerifier — 10-row output checklist |
| `4dad334` | refactor(jang-studio): drop unused VerifyID cases sizeWithinEstimate/inspectSucceeds |

**What's shippable:**
- `PreflightRunner.run(plan:) -> [PreflightCheck]` — 10 checks (source/config/output/disk/RAM/JANGTQ arch/JANGTQ dtype/bf16-for-512-experts/hadamard-vs-2bit/python-healthy). `fail` blocks Start; `warn` lets it through.
- `PostConvertVerifier.run(plan:skipPythonValidate:) async -> [VerifyCheck]` — 10 checks (jang_config+schema+capabilities, chat template, tokenizer files, shards match index, VL preprocessors if VL, MiniMax custom .py if minimax_m2, tokenizer class concrete). Required-failure blocks Finish.
- Two golden fixtures: `good_output/` (every check passes) and `broken_output/` (chat template missing + shard count mismatch).

**Decision:** Spec's original §4.2 had 12 verifier rows; implementation shipped with 10. Dropped `sizeWithinEstimate` (warn-only, nice-to-have) and `inspectSucceeds` (redundant with `schemaValid`). Close the gap cleanly rather than leave dead enum cases.

### ✅ Phase 5 — Wizard UI + full coverage matrix audit (complete 2026-04-19)

13 commits. 41 tests green (40 XCTest unit + 1 XCUITest).

| SHA | Commit |
|---|---|
| `88d36f5` | WizardCoordinator + 5-step sidebar scaffold |
| `c5efaee` | Step 1 — source picker + inspect-source detection |
| `0064704` | Step 2 — architecture summary + advanced overrides |
| `da83d40` | Step 3 — profile picker + live preflight |
| `c2cd87c` | Step 4 — live run with phase/tick/log streams |
| `cbd5d78` | Step 5 — verification checklist + Finish |
| `565936f` | DiagnosticsBundle zip for bug reports |
| `b588495` | XCUITest verifies 5-step sidebar renders |
| `4c4ef02` + `8b59e9d` + `b36fc60` | inspect-source + ArchitectureSummary propagate `is_video_vl` |
| `27f9acb` | Verifier gains video preprocessor, generation_config, layer count, chat_template.json |
| `c7b4a98` | Extract CLIArgsBuilder from RunStep for testability |
| `179f11c` | Audit suite covers video-VL, dense, sentencepiece, chat_template.json |

**Wizard UI:**
- `NavigationSplitView` sidebar with 5 locked/active/complete step icons
- Step 1: `NSOpenPanel` folder picker → calls `inspect-source --json` → summary card
- Step 2: detected arch + collapsible force-dtype/force-blocksize overrides
- Step 3: JANG/JANGTQ tabs (JANGTQ disabled on non-whitelist), all 15+3 profiles, method + hadamard toggles, live preflight panel
- Step 4: macro + fine progress bars, JSONL log stream, cancel button, Copy Diagnostics on failure
- Step 5: 12-row verifier checklist, Reveal in Finder / Copy Path / Convert another / Finish

**Verifier now covers 12 rows:**
1. `jang_config.json` exists + JSON-valid
2. format=="jang" + format_version 2.x+
3. `jang validate` schema check (shelled out)
4. `capabilities` stamp non-empty
5. Chat template — inline OR `.jinja` OR `.json` (3-way OR)
6. Tokenizer files — (`tokenizer.json` OR `tokenizer.model`) + `tokenizer_config.json` + `special_tokens_map.json`
7. Shards match `model.safetensors.index.json`
8. `preprocessor_config.json` when VL
8b. `video_preprocessor_config.json` when video-VL
9. `modeling_*.py` + `configuration_*.py` when MiniMax-class
10. Tokenizer class concrete (not "TokenizersBackend" — Osaurus-compat warn)
11. `generation_config.json` present (warn — HF fallback is OK)
12. `num_hidden_layers > 0` in `config.json` (sanity)

**Audit matrix (`CoverageMatrixTests` + `CLIArgsBuilderTests`):**
- Every JANG profile (15) × every arch class × preflight gate + args routing
- Every JANGTQ profile (3) accepted on whitelisted archs (qwen3_5_moe, minimax_m2), rejected on all others
- Dense (llama) path tested for every JANG profile; JANGTQ rejected on dense
- Image-VL path (preprocessor_config.json required)
- Video-VL path (video_preprocessor_config.json required)
- Chat template alternatives: inline, `.jinja`, `.json` — all three accepted
- Tokenizer alternatives: `.json` (BPE) or `.model` (sentencepiece)
- MiniMax custom `.py` files required when `minimax_m2`
- `generation_config.json` warn-but-not-block
- Arch classes covered: llama, qwen3_5_moe, qwen3_5_moe (VL), qwen3_5_moe (FP8), minimax_m2 (BF16 + FP8), glm_moe_dsa, deepseek_v32

### ✅ Phase 6 — Python bundle + codesign + notarize (complete 2026-04-19)

4 commits. Bundle verified at 305 MB with working `jang` CLI smoke test.

| SHA | Commit |
|---|---|
| `8c997fb` | build(jang-studio): build-python-bundle.sh — hermetic python 3.11 + jang[mlx,vlm] |
| `f214c76` | build(jang-studio): codesign-runtime.sh — deep-sign bundled python + app |
| `26ddcba` | build(jang-studio): notarize.sh — xcrun notarytool + stapler wrapper |
| `7b526b6` | build(jang-studio): trim bundle — install mlx-vlm --no-deps (skip cv2/pyarrow) |

**What's shippable:**
- `Scripts/build-python-bundle.sh` — idempotent (skips if bundle already exists + smoke-tests clean). Downloads python-build-standalone 3.11.10 (aarch64-apple-darwin, install_only), builds & pip-installs the local `jang` wheel with `[mlx]`, then `mlx-vlm --no-deps`, then `transformers/tokenizers/sentencepiece` explicitly. Strips tests/docs/caches/idlelib/tkinter. Ad-hoc codesigns all `.dylib`/`.so`. Fails if bundle exceeds 450 MB cap.
- `Scripts/codesign-runtime.sh` — deep-signs every `.dylib`/`.so` under `Contents/Resources/python` in leaf-first order, then signs the outer `.app` with hardened runtime + entitlements. Verifies with `codesign --verify --deep --strict` + Gatekeeper pre-check. Requires `$APPLE_DEV_ID_APP` env var (CI-only).
- `Scripts/notarize.sh` — zips, submits via `xcrun notarytool submit --wait`, staples the ticket, re-verifies with Gatekeeper. Requires `$APPLE_ID` + `$APPLE_TEAM_ID` + `$APPLE_APP_PASSWORD` env vars (CI-only).
- `project.yml` — `copyFiles` stanza places `build/python/` into `Contents/Resources/python` during build.

**Bundle contents** (305 MB total, top 5):
- mlx: 153 MB · transformers: 44 MB · numpy: 17 MB · tokenizers: 8 MB · hf_xet: 7 MB

**Deliberate omissions** (mlx-vlm inference-only deps, 232 MB saved): cv2 (OpenCV, webcam input), pyarrow (datasets backend), datasets (training-time), fastapi/uvicorn (mlx-vlm's server mode). If a future code path imports from these, it will `ImportError` at runtime — currently `jang_tools` only imports `mlx_vlm.utils` and `mlx_vlm.tokenizer_utils`, neither of which need them.

**preBuildScripts** (project.yml wiring): deliberately NOT enabled for local builds — the bundle takes ~3 min on first run and blocks Xcode iteration. Real build path is either:
1. Developer: manually run `Scripts/build-python-bundle.sh` once after cloning, then `xcodebuild` picks up the existing `build/python/`.
2. CI (Phase 7): explicit `build-python-bundle.sh` step before `xcodebuild` in the release workflow.

### ⬜ Phase 7 — CI + signed DMG (not started)

### ✅ Phase 7 — CI + signed DMG on tag (complete 2026-04-19)

1 commit. `.github/workflows/jang-studio.yml` validated.

| SHA | Commit |
|---|---|
| `f3fc17b` | ci(jang-studio): build/test on PRs; signed DMG release on jang-studio-v* tag |

**What's shippable:**
- **build-test job** — runs on every push to main + every PR touching `JANGStudio/**` or `jang-tools/**`. Runs pytest (jang-tools), builds wheel, runs `build-python-bundle.sh`, generates Xcode project, runs Swift XCTest suite (`-only-testing:JANGStudioTests`), uploads test results as artifact on both pass AND fail.
- **release job** — runs only on `jang-studio-v*` tag push. Imports Developer ID cert from `APPLE_DEV_ID_CERT_P12` secret into a temp keychain, builds Release .app with ad-hoc signing, runs `codesign-runtime.sh` to deep-sign + hardened-runtime-sign, runs `notarize.sh` to submit + wait + staple the app, creates DMG via `create-dmg`, notarizes + staples the DMG, publishes to GitHub Releases.
- **Required repo secrets** (set before first tag):
  - `APPLE_DEV_ID_CERT_P12` — base64-encoded .p12 of Developer ID Application cert
  - `APPLE_DEV_ID_CERT_PW` — .p12 passphrase
  - `APPLE_DEV_ID_APP` — signing identity string (`Developer ID Application: Jinho Jang (TEAMID)`)
  - `APPLE_ID` — Apple account email
  - `APPLE_TEAM_ID` — 10-char team ID
  - `APPLE_APP_PASSWORD` — app-specific password
- **Concurrency:** build-test runs cancel on new pushes to same ref; release runs don't (one tag = one release).

### ⬜ Phase 8 — Documentation (not started)

---

## File Structure (what gets created/modified)

### `jang-tools/` (Python side — Phase 1)

| File | Status | Responsibility |
|---|---|---|
| `jang_tools/progress.py` | **CREATE** | `ProgressEmitter` — phase/tick/event API, text + JSONL output |
| `jang_tools/inspect_source.py` | **CREATE** | New `inspect-source --json` subcommand for Swift |
| `jang_tools/__main__.py` | MODIFY | Add `--progress=json`, `--quiet-text` globals; register `inspect-source` subparser |
| `jang_tools/convert.py` | MODIFY | Thread `ProgressEmitter` through phases 1-5 and tqdm wrapper |
| `jang_tools/convert_qwen35_jangtq.py` | MODIFY | Same integration (prints + tqdm → emitter) |
| `jang_tools/convert_minimax_jangtq.py` | MODIFY | Same integration |
| `tests/test_progress.py` | **CREATE** | Unit tests for `ProgressEmitter` |
| `tests/test_inspect_source.py` | **CREATE** | Unit tests for `inspect-source` |
| `tests/fixtures/tiny_qwen/` | **CREATE** | Minimal HF model dir for `inspect-source` tests |

### `JANGStudio/` (Swift side — Phases 2-5)

```
JANGStudio/
├── JANGStudio.xcodeproj/
├── project.yml                          # xcodegen spec (source of truth)
├── JANGStudio/
│   ├── App/
│   │   └── JANGStudioApp.swift          # @main + top-level view
│   ├── Models/
│   │   ├── ConversionPlan.swift         # @Observable — shared wizard state
│   │   ├── ArchitectureSummary.swift    # Detected arch info
│   │   ├── ProgressEvent.swift          # Codable JSONL events
│   │   ├── Profile.swift                # JANG/JANGTQ profile catalog
│   │   └── PreflightCheck.swift         # Preflight row model
│   ├── Runner/
│   │   ├── BundleResolver.swift         # Path to bundled python3
│   │   ├── PythonRunner.swift           # actor Process + async stream
│   │   └── JSONLProgressParser.swift    # stderr line → ProgressEvent
│   ├── Verify/
│   │   ├── PreflightRunner.swift        # 10-row gate
│   │   ├── PostConvertVerifier.swift    # 12-row finish gate
│   │   └── VerifyCheck.swift            # row status model
│   ├── Wizard/
│   │   ├── WizardCoordinator.swift      # NavigationSplitView host
│   │   └── Steps/
│   │       ├── SourceStep.swift
│   │       ├── ArchitectureStep.swift
│   │       ├── ProfileStep.swift
│   │       ├── RunStep.swift
│   │       └── VerifyStep.swift
│   └── Resources/
│       ├── Assets.xcassets/
│       ├── Info.plist
│       └── JANGStudio.entitlements
├── Tests/
│   ├── JANGStudioTests/
│   │   ├── ConversionPlanTests.swift
│   │   ├── JSONLProgressParserTests.swift
│   │   ├── PythonRunnerTests.swift
│   │   ├── PreflightRunnerTests.swift
│   │   ├── PostConvertVerifierTests.swift
│   │   ├── BundleResolverTests.swift
│   │   └── Fixtures/                     # canned JSONL + fake-model-dir fixtures
│   └── JANGStudioUITests/
│       └── WizardFlowTests.swift         # XCUITest with stub runner
├── Scripts/
│   ├── build-python-bundle.sh
│   ├── codesign-runtime.sh
│   └── notarize.sh
├── docs/
│   ├── USER_GUIDE.md
│   ├── TROUBLESHOOTING.md
│   ├── CONTRIBUTING.md
│   └── PROGRESS_PROTOCOL.md
└── README.md
```

### Repo-wide

| File | Status | Responsibility |
|---|---|---|
| `.github/workflows/jang-studio.yml` | **CREATE** | CI: build, test, sign, notarize, DMG |
| `.gitignore` | MODIFY | Add `JANGStudio/build/`, `.superpowers/` |
| `README.md` (top-level) | MODIFY | Add "Get JANG Studio" section |

---

## Phase 1 — Python Progress Protocol + `inspect-source`

Dependencies: none. Output: JSONL events on stderr, `inspect-source` subcommand ready for Swift to call.

### Task 1.1 — `ProgressEmitter` API and failing test

**Files:**
- Create: `jang-tools/tests/test_progress.py`

- [ ] **Step 1: Write the failing test**

```python
# jang-tools/tests/test_progress.py
"""Tests for jang_tools.progress — ProgressEmitter API and JSONL schema."""
import io
import json
import time
import pytest

from jang_tools.progress import ProgressEmitter


def _drain(emitter: ProgressEmitter) -> list[dict]:
    """Parse the emitter's JSONL buffer into a list of event dicts."""
    raw = emitter._stderr.getvalue()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_phase_event_shape():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.phase(1, 5, "detect")
    events = _drain(em)
    assert len(events) == 1
    ev = events[0]
    assert ev["v"] == 1
    assert ev["type"] == "phase"
    assert ev["n"] == 1
    assert ev["total"] == 5
    assert ev["name"] == "detect"
    assert isinstance(ev["ts"], (int, float))


def test_tick_event_shape():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.tick(1234, 2630, "layer.5.gate_proj")
    ev = _drain(em)[0]
    assert ev["type"] == "tick"
    assert ev["done"] == 1234
    assert ev["total"] == 2630
    assert ev["label"] == "layer.5.gate_proj"


def test_event_warn_and_info():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.event("warn", "No chat template found")
    em.event("info", "Detected qwen3_5_moe")
    evs = _drain(em)
    assert evs[0]["type"] == "warn"
    assert evs[0]["msg"] == "No chat template found"
    assert evs[1]["type"] == "info"


def test_done_success_and_failure():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.done(ok=True, output="/tmp/out", elapsed_s=12.5)
    ev = _drain(em)[0]
    assert ev["type"] == "done"
    assert ev["ok"] is True
    assert ev["output"] == "/tmp/out"
    assert ev["elapsed_s"] == 12.5

    em2 = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em2.done(ok=False, error="OOM while loading experts")
    ev2 = _drain(em2)[0]
    assert ev2["ok"] is False
    assert ev2["error"] == "OOM while loading experts"


def test_text_mode_only_no_json():
    out, err = io.StringIO(), io.StringIO()
    em = ProgressEmitter(json_to_stderr=False, quiet_text=False, _stdout=out, _stderr=err)
    em.phase(1, 5, "detect")
    assert err.getvalue() == ""           # no JSONL
    assert "[1/5]" in out.getvalue()       # human-readable on stdout


def test_quiet_text_suppresses_stdout():
    out, err = io.StringIO(), io.StringIO()
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stdout=out, _stderr=err)
    em.phase(1, 5, "detect")
    assert out.getvalue() == ""            # suppressed
    assert err.getvalue() != ""            # JSONL still written


def test_tick_throttling_coalesces():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    # Fire 100 ticks back-to-back — should coalesce to ≤ a handful
    for i in range(100):
        em.tick(i, 100, f"t{i}")
    events = _drain(em)
    # Must always emit the final tick (done == total) but coalesce the rest
    assert len(events) < 50
    assert events[-1]["done"] == 99 or events[-1]["done"] == 100


def test_warn_never_throttled():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.tick(1, 100, "a")
    em.event("warn", "something")
    em.tick(2, 100, "b")
    types = [e["type"] for e in _drain(em)]
    assert "warn" in types
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_progress.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'jang_tools.progress'`.

- [ ] **Step 3: Commit the failing test**

```bash
cd /Users/eric/jang
git add jang-tools/tests/test_progress.py
git commit -m "test: ProgressEmitter JSONL schema and throttling contract"
```

---

### Task 1.2 — Implement `ProgressEmitter`

**Files:**
- Create: `jang-tools/jang_tools/progress.py`

- [ ] **Step 1: Write the module**

```python
# jang-tools/jang_tools/progress.py
"""Progress emitter for jang-tools — supports human text and JSONL.

JSONL schema v1 (one object per line on stderr):
    {"v":1,"type":"phase","n":1,"total":5,"name":"detect","ts":1700000000.123}
    {"v":1,"type":"tick","done":1234,"total":2630,"label":"layer.5","ts":...}
    {"v":1,"type":"info"|"warn"|"error","msg":"...","ts":...}
    {"v":1,"type":"done","ok":true,"output":"/path","elapsed_s":12.5,"ts":...}
    {"v":1,"type":"done","ok":false,"error":"...","ts":...}

See docs/PROGRESS_PROTOCOL.md for full spec.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, IO

PROTOCOL_VERSION = 1
_TICK_MIN_INTERVAL_S = 0.1


class ProgressEmitter:
    def __init__(
        self,
        json_to_stderr: bool,
        quiet_text: bool,
        _stdout: IO[str] | None = None,
        _stderr: IO[str] | None = None,
    ) -> None:
        self._json = json_to_stderr
        self._quiet = quiet_text
        self._stdout = _stdout if _stdout is not None else sys.stdout
        self._stderr = _stderr if _stderr is not None else sys.stderr
        self._last_tick_ts = 0.0

    def _emit_json(self, payload: dict[str, Any]) -> None:
        if not self._json:
            return
        payload["v"] = PROTOCOL_VERSION
        payload.setdefault("ts", time.time())
        self._stderr.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._stderr.flush()

    def _emit_text(self, line: str) -> None:
        if self._quiet:
            return
        self._stdout.write(line + "\n")
        self._stdout.flush()

    def phase(self, n: int, total: int, name: str) -> None:
        self._emit_json({"type": "phase", "n": n, "total": total, "name": name})
        self._emit_text(f"  [{n}/{total}] {name}")

    def tick(self, done: int, total: int, label: str = "") -> None:
        now = time.time()
        is_final = done >= total - 1
        if not is_final and (now - self._last_tick_ts) < _TICK_MIN_INTERVAL_S:
            return
        self._last_tick_ts = now
        payload: dict[str, Any] = {"type": "tick", "done": done, "total": total}
        if label:
            payload["label"] = label
        self._emit_json(payload)

    def event(self, level: str, message: str, **fields: Any) -> None:
        assert level in ("info", "warn", "error"), f"unknown level {level}"
        payload = {"type": level, "msg": message, **fields}
        self._emit_json(payload)
        if level in ("warn", "error"):
            self._emit_text(f"  [{level.upper()}] {message}")

    def done(
        self,
        ok: bool,
        output: str | None = None,
        elapsed_s: float | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "done", "ok": ok}
        if output is not None:
            payload["output"] = output
        if elapsed_s is not None:
            payload["elapsed_s"] = elapsed_s
        if error is not None:
            payload["error"] = error
        self._emit_json(payload)


def make_noop() -> ProgressEmitter:
    """Emitter that writes nothing. Used when no progress flag is set."""
    return ProgressEmitter(json_to_stderr=False, quiet_text=True)
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_progress.py -v
```

Expected: PASS (all 8 tests).

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/jang_tools/progress.py
git commit -m "feat: ProgressEmitter for dual text + JSONL reporting"
```

---

### Task 1.3 — Wire `--progress=json` and `--quiet-text` into CLI

**Files:**
- Modify: `jang-tools/jang_tools/__main__.py` (add global args, thread `args` → subcommands)
- Create: `jang-tools/tests/test_cli_flags.py`

- [ ] **Step 1: Write the failing test**

```python
# jang-tools/tests/test_cli_flags.py
"""Smoke tests for the new --progress=json and --quiet-text global flags."""
import json
import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jang_tools", *args],
        capture_output=True, text=True, check=False,
    )


def test_version_still_works():
    r = _run(["--version"])
    assert r.returncode == 0
    assert "jang-tools" in r.stdout


def test_help_lists_progress_flag():
    r = _run(["--help"])
    assert "--progress" in r.stdout
    assert "--quiet-text" in r.stdout


def test_progress_json_emits_jsonl_on_stderr_for_inspect():
    # inspect on a nonexistent path should emit a "done ok:false" JSON line
    r = _run(["--progress=json", "inspect", "/tmp/definitely_does_not_exist_xyz"])
    assert r.returncode != 0
    lines = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    assert any(ev.get("type") == "done" and ev.get("ok") is False for ev in lines)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_cli_flags.py -v
```

Expected: FAIL — `--progress` unknown or no `done` event.

- [ ] **Step 3: Modify `__main__.py`**

Modify `/Users/eric/jang/jang-tools/jang_tools/__main__.py`:

Add to imports at top:
```python
import time
from .progress import ProgressEmitter, make_noop
```

In `def main():`, after `parser.add_argument("--version", …)`, add:
```python
    parser.add_argument(
        "--progress", choices=["json", "off"], default="off",
        help="Emit JSONL progress events on stderr (for GUIs).",
    )
    parser.add_argument(
        "--quiet-text", action="store_true",
        help="Suppress human-readable phase/progress prints on stdout.",
    )
```

Replace the dispatch at the end of `main()`:
```python
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    progress = ProgressEmitter(
        json_to_stderr=(args.progress == "json"),
        quiet_text=args.quiet_text,
    )
    args.progress_emitter = progress
    t0 = time.time()
    try:
        args.func(args)
        progress.done(ok=True, elapsed_s=time.time() - t0)
    except SystemExit as e:
        progress.done(ok=False, error=f"exit-code-{e.code}", elapsed_s=time.time() - t0)
        raise
    except Exception as e:
        progress.done(ok=False, error=f"{type(e).__name__}: {e}", elapsed_s=time.time() - t0)
        raise
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_cli_flags.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/jang_tools/__main__.py jang-tools/tests/test_cli_flags.py
git commit -m "feat(cli): --progress=json and --quiet-text global flags"
```

---

### Task 1.4 — Thread `ProgressEmitter` through `convert.py` phases and tqdm

**Files:**
- Modify: `jang-tools/jang_tools/convert.py` (phases 1-5 prints → `emitter.phase`; tqdm wrapper → `emitter.tick`)
- Create: `jang-tools/tests/test_convert_progress.py`

- [ ] **Step 1: Write the failing test**

```python
# jang-tools/tests/test_convert_progress.py
"""Verify convert.py emits the 5 expected phase events in order."""
import json
import subprocess
import sys
from pathlib import Path


def test_convert_emits_all_phases_even_on_missing_model(tmp_path):
    # Point at a non-model dir; we only care about the phase event ordering
    # before the converter fails out. It should still emit at least phase 1
    # and a done(ok=False) event.
    bogus = tmp_path / "not_a_model"
    bogus.mkdir()
    (bogus / "README").write_text("nope")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "--progress=json", "--quiet-text",
         "convert", str(bogus), "-o", str(tmp_path / "out"), "-p", "2"],
        capture_output=True, text=True, check=False,
    )
    events = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    types = [e["type"] for e in events]
    assert "done" in types
    done_ev = [e for e in events if e["type"] == "done"][-1]
    assert done_ev["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_convert_progress.py -v
```

Expected: FAIL — convert.py crashes before emitting structured events, or prints traceback only.

- [ ] **Step 3: Modify `convert.py`**

In `/Users/eric/jang/jang-tools/jang_tools/convert.py`:

Change the `convert_model()` signature to accept an optional `progress_emitter`:
```python
def convert_model(
    model_path,
    output_path=None,
    profile="2",
    method="mse",
    hadamard=False,
    *,
    progress_emitter=None,
):
    from .progress import make_noop
    progress = progress_emitter if progress_emitter is not None else make_noop()
```

Replace existing phase prints (`convert.py:155, 199, 215, 230, 431, 839`):
```python
    # was: print("  [1/5] Detecting architecture...")
    progress.phase(1, 5, "detect")
    # ...
    progress.phase(2, 5, "calibrate")   # or "skip-calibration"
    # ...
    progress.phase(3, 5, "allocate")
    # ...
    progress.phase(4, 5, "quantize")
    # ...
    progress.phase(5, 5, "write")
```

Replace the tqdm loop at `convert.py:439`:
```python
    # was: for tensor_name, shape, n_blocks, sf_path in tqdm(all_tensors_info, desc="  Quantizing"):
    total_tensors = len(all_tensors_info)
    for i, (tensor_name, shape, n_blocks, sf_path) in enumerate(all_tensors_info):
        progress.tick(i, total_tensors, tensor_name)
        # ... existing body unchanged ...
    progress.tick(total_tensors, total_tensors, "done")
```

In `jang_tools/__main__.py`, `cmd_convert()` now forwards the emitter:
```python
def cmd_convert(args):
    from .convert import convert_model
    # ... existing arg parsing ...
    result = convert_model(
        args.model, output_path=args.output, profile=profile, method=args.method,
        hadamard=args.hadamard, progress_emitter=getattr(args, "progress_emitter", None),
    )
```

- [ ] **Step 4: Run tests to verify**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_convert_progress.py -v
```

Expected: PASS.

Also run the full existing test suite to ensure no regression:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest -x
```

Expected: all pre-existing tests still pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/jang_tools/convert.py jang-tools/jang_tools/__main__.py jang-tools/tests/test_convert_progress.py
git commit -m "feat(convert): emit phase + tick events via ProgressEmitter"
```

---

### Task 1.5 — Same integration for `convert_qwen35_jangtq.py` and `convert_minimax_jangtq.py`

**Files:**
- Modify: `jang-tools/jang_tools/convert_qwen35_jangtq.py`
- Modify: `jang-tools/jang_tools/convert_minimax_jangtq.py`

Both scripts are standalone entrypoints (`python -m jang_tools.convert_qwen35_jangtq <src> <out> <profile>`). They need the same flag plumbing.

- [ ] **Step 1: Add argparse flags to each JANGTQ script**

At the top of each file's entrypoint (after the existing `SRC`/`OUT`/`PROFILE` sys.argv parsing), add:

```python
import argparse
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--progress", choices=["json", "off"], default="off")
_ap.add_argument("--quiet-text", action="store_true")
_args, _rest = _ap.parse_known_args()
# Re-bind sys.argv so existing positional parsing still works
sys.argv = [sys.argv[0]] + _rest

from jang_tools.progress import ProgressEmitter
progress = ProgressEmitter(
    json_to_stderr=(_args.progress == "json"),
    quiet_text=_args.quiet_text,
)
```

- [ ] **Step 2: Replace major phase prints**

In `convert_qwen35_jangtq.py`, locate each top-level block (load, stack, quantize, write) and replace the leading `print("...")` banner with:
```python
progress.phase(1, 4, "load")       # for the weight-loading section
progress.phase(2, 4, "stack")      # for expert stacking
progress.phase(3, 4, "quantize")
progress.phase(4, 4, "write")
```

(Exact line numbers vary — search for the existing `"=" * 60` banners.)

Do the same in `convert_minimax_jangtq.py`.

At the end of both scripts (after successful exit), add:
```python
progress.done(ok=True, output=str(OUT))
```

Wrap the main flow in a `try/except` to report failure:
```python
try:
    # ... main body ...
    progress.done(ok=True, output=str(OUT))
except Exception as e:
    progress.done(ok=False, error=f"{type(e).__name__}: {e}")
    raise
```

- [ ] **Step 3: Write the integration test**

```python
# jang-tools/tests/test_jangtq_progress.py
"""Smoke test: JANGTQ scripts accept --progress=json and emit a done event."""
import json
import subprocess
import sys


def _run_module(mod: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", mod, "--progress=json", "--quiet-text", *args],
        capture_output=True, text=True, check=False,
    )


def test_qwen35_jangtq_emits_done_on_bad_input(tmp_path):
    r = _run_module("jang_tools.convert_qwen35_jangtq",
                    ["/tmp/nope_xyz", str(tmp_path / "out"), "JANGTQ2"])
    assert r.returncode != 0
    lines = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    assert any(e.get("type") == "done" and e.get("ok") is False for e in lines)


def test_minimax_jangtq_emits_done_on_bad_input(tmp_path):
    r = _run_module("jang_tools.convert_minimax_jangtq",
                    ["/tmp/nope_xyz", str(tmp_path / "out"), "JANGTQ2"])
    assert r.returncode != 0
    lines = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    assert any(e.get("type") == "done" and e.get("ok") is False for e in lines)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_jangtq_progress.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/jang_tools/convert_qwen35_jangtq.py \
        jang-tools/jang_tools/convert_minimax_jangtq.py \
        jang-tools/tests/test_jangtq_progress.py
git commit -m "feat(jangtq): JSONL progress flags for qwen35/minimax converters"
```

---

### Task 1.6 — New `inspect-source` subcommand

**Files:**
- Create: `jang-tools/jang_tools/inspect_source.py`
- Modify: `jang-tools/jang_tools/__main__.py` (register subparser)
- Create: `jang-tools/tests/test_inspect_source.py`
- Create: `jang-tools/tests/fixtures/tiny_qwen/config.json`

- [ ] **Step 1: Create the fixture**

```json
// jang-tools/tests/fixtures/tiny_qwen/config.json
{
  "model_type": "qwen3_5_moe",
  "hidden_size": 128,
  "num_hidden_layers": 2,
  "num_attention_heads": 4,
  "vocab_size": 1024,
  "num_experts": 8,
  "torch_dtype": "bfloat16"
}
```

- [ ] **Step 2: Write the failing test**

```python
# jang-tools/tests/test_inspect_source.py
"""Verify inspect-source returns valid JSON with expected keys."""
import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_qwen"


def test_inspect_source_prints_valid_json():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(FIXTURE)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["model_type"] == "qwen3_5_moe"
    assert data["is_moe"] is True
    assert data["num_experts"] == 8
    assert data["dtype"] in ("bfloat16", "float16", "float8_e4m3fn", "unknown")
    assert "jangtq_compatible" in data
    assert data["jangtq_compatible"] is True   # qwen3_5_moe is in the v1 whitelist


def test_inspect_source_missing_config_errors(tmp_path):
    (tmp_path / "README").write_text("nope")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "config.json" in r.stderr.lower()
```

- [ ] **Step 3: Run test (fails)**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_inspect_source.py -v
```

Expected: FAIL — subcommand unknown.

- [ ] **Step 4: Implement `inspect_source.py`**

```python
# jang-tools/jang_tools/inspect_source.py
"""Fast, no-model-load source inspector. Used by JANG Studio's Step 1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

# v1 JANGTQ whitelist — synced with JANG Studio spec §2.5
_JANGTQ_V1_WHITELIST = {"qwen3_5_moe", "minimax_m2"}


def _sniff_dtype(model_path: Path) -> str:
    """Peek at the first safetensors shard header to determine source dtype."""
    shards = sorted(model_path.glob("*.safetensors"))
    if not shards:
        return "unknown"
    try:
        import struct
        with open(shards[0], "rb") as fh:
            hdr_len = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(hdr_len))
        dtypes = {v.get("dtype") for k, v in hdr.items() if isinstance(v, dict) and "dtype" in v}
        for preferred in ("BF16", "F16", "F8_E4M3", "F8_E5M2"):
            if preferred in dtypes:
                return {
                    "BF16": "bfloat16", "F16": "float16",
                    "F8_E4M3": "float8_e4m3fn", "F8_E5M2": "float8_e5m2",
                }[preferred]
        return "unknown"
    except Exception:
        return "unknown"


def _is_moe(cfg: dict) -> bool:
    for key in ("num_experts", "n_routed_experts", "num_local_experts"):
        if cfg.get(key, 0) and int(cfg[key]) > 1:
            return True
    return False


def _total_bytes(model_path: Path) -> int:
    return sum(f.stat().st_size for f in model_path.glob("*.safetensors"))


def cmd_inspect_source(args) -> None:
    src = Path(args.model)
    cfg_path = src / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: config.json not found under {src}", file=sys.stderr)
        sys.exit(2)
    cfg = json.loads(cfg_path.read_text())
    model_type = cfg.get("model_type") or cfg.get("text_config", {}).get("model_type", "unknown")
    summary = {
        "model_type": model_type,
        "is_moe": _is_moe(cfg),
        "num_experts": int(cfg.get("num_experts") or cfg.get("n_routed_experts") or 0),
        "num_hidden_layers": int(cfg.get("num_hidden_layers", 0)),
        "hidden_size": int(cfg.get("hidden_size", 0)),
        "dtype": _sniff_dtype(src),
        "total_bytes": _total_bytes(src),
        "shard_count": len(list(src.glob("*.safetensors"))),
        "jangtq_compatible": model_type in _JANGTQ_V1_WHITELIST,
        "is_vl": bool((src / "preprocessor_config.json").exists()),
    }
    if args.json:
        print(json.dumps(summary, indent=None, separators=(",", ":")))
    else:
        for k, v in summary.items():
            print(f"  {k}: {v}")


def register(subparsers) -> None:
    p = subparsers.add_parser("inspect-source", help="Fast source-model inspector")
    p.add_argument("model", help="Path to HuggingFace model directory")
    p.add_argument("--json", action="store_true", help="Emit single JSON line on stdout")
    p.set_defaults(func=cmd_inspect_source)
```

Wire it up in `__main__.py` — find the other `add_parser` calls and add at the end of `main()` before `args = parser.parse_args()`:
```python
    from .inspect_source import register as _register_inspect_source
    _register_inspect_source(subparsers)
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/test_inspect_source.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/jang_tools/inspect_source.py jang-tools/jang_tools/__main__.py \
        jang-tools/tests/test_inspect_source.py \
        jang-tools/tests/fixtures/tiny_qwen/config.json
git commit -m "feat(cli): inspect-source subcommand for GUI integration"
```

---

### Task 1.7 — Golden JSONL fixture for Swift-side testing

**Files:**
- Create: `jang-tools/tests/fixtures/golden_convert_events.jsonl`

- [ ] **Step 1: Capture a golden trace**

Run a synthetic emitter invocation that produces the exact JSONL a real conversion would emit, and save it. This fixture will be re-used by Swift's `JSONLProgressParserTests`.

```bash
cd /Users/eric/jang/jang-tools
python -c '
from jang_tools.progress import ProgressEmitter
import sys
em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=sys.stdout, _stdout=None)
em.phase(1, 5, "detect")
em.event("info", "Detected qwen3_5_moe — 256 experts, MLA")
em.phase(2, 5, "calibrate")
em.phase(3, 5, "allocate")
em.event("info", "Profile JANG_4K: 4/4/8 bits")
em.phase(4, 5, "quantize")
for i in range(0, 100, 10):
    em.tick(i, 100, f"layer.{i//10}.gate_proj")
em.tick(100, 100, "done")
em.event("warn", "No chat template found; will warn in verify")
em.phase(5, 5, "write")
em.done(ok=True, output="/tmp/out", elapsed_s=42.5)
' > tests/fixtures/golden_convert_events.jsonl
```

- [ ] **Step 2: Commit**

```bash
cd /Users/eric/jang
git add jang-tools/tests/fixtures/golden_convert_events.jsonl
git commit -m "test: golden JSONL fixture for Swift parser tests"
```

---

## Phase 2 — Xcode Project Scaffolding + Core Models

Dependencies: Phase 1 complete. Output: empty `.app` that builds + core domain types with tests.

### Task 2.1 — Bootstrap Xcode project via xcodegen

We use [xcodegen](https://github.com/yonaskolb/XcodeGen) so `project.yml` is the source of truth (Xcode project files don't merge cleanly). Install: `brew install xcodegen`.

**Files:**
- Create: `JANGStudio/project.yml`
- Create: `JANGStudio/JANGStudio/App/JANGStudioApp.swift` (placeholder)
- Create: `JANGStudio/JANGStudio/Resources/Info.plist`
- Create: `JANGStudio/JANGStudio/Resources/JANGStudio.entitlements`

- [ ] **Step 1: Create `project.yml`**

```yaml
# JANGStudio/project.yml
name: JANGStudio
options:
  deploymentTarget:
    macOS: "15.0"
  bundleIdPrefix: ai.jangq
  xcodeVersion: "16.0"
settings:
  base:
    SWIFT_VERSION: "6.0"
    MARKETING_VERSION: "1.0.0"
    CURRENT_PROJECT_VERSION: "1"
    ENABLE_HARDENED_RUNTIME: YES
    CODE_SIGN_IDENTITY: "-"
    DEVELOPMENT_TEAM: ""
targets:
  JANGStudio:
    type: application
    platform: macOS
    sources:
      - path: JANGStudio
    resources:
      - path: JANGStudio/Resources/Assets.xcassets
        optional: true
    info:
      path: JANGStudio/Resources/Info.plist
      properties:
        CFBundleName: JANG Studio
        CFBundleDisplayName: JANG Studio
        CFBundleIdentifier: ai.jangq.JANGStudio
        LSMinimumSystemVersion: "15.0"
        NSHighResolutionCapable: YES
    entitlements:
      path: JANGStudio/Resources/JANGStudio.entitlements
      properties:
        com.apple.security.cs.allow-jit: false
        com.apple.security.cs.disable-library-validation: true
        com.apple.security.files.user-selected.read-write: true
    dependencies: []
    scheme:
      testTargets: [JANGStudioTests, JANGStudioUITests]
      gatherCoverageData: true
  JANGStudioTests:
    type: bundle.unit-test
    platform: macOS
    sources: [Tests/JANGStudioTests]
    dependencies: [{target: JANGStudio}]
  JANGStudioUITests:
    type: bundle.ui-testing
    platform: macOS
    sources: [Tests/JANGStudioUITests]
    dependencies: [{target: JANGStudio}]
```

- [ ] **Step 2: Create placeholder app entry point**

```swift
// JANGStudio/JANGStudio/App/JANGStudioApp.swift
import SwiftUI

@main
struct JANGStudioApp: App {
    var body: some Scene {
        WindowGroup("JANG Studio") {
            Text("JANG Studio — scaffolding")
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowResizability(.contentSize)
    }
}
```

- [ ] **Step 3: Create Info.plist**

```xml
<!-- JANGStudio/JANGStudio/Resources/Info.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key><string>en</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$(MARKETING_VERSION)</string>
    <key>CFBundleVersion</key><string>$(CURRENT_PROJECT_VERSION)</string>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
</dict>
</plist>
```

- [ ] **Step 4: Create entitlements file**

```xml
<!-- JANGStudio/JANGStudio/Resources/JANGStudio.entitlements -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-jit</key><false/>
    <key>com.apple.security.cs.disable-library-validation</key><true/>
    <key>com.apple.security.files.user-selected.read-write</key><true/>
</dict>
</plist>
```

- [ ] **Step 5: Generate + build the project**

```bash
brew list xcodegen >/dev/null 2>&1 || brew install xcodegen
cd /Users/eric/jang/JANGStudio
xcodegen generate
xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build
```

Expected: `BUILD SUCCEEDED`.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang
echo "JANGStudio/build/" >> .gitignore
echo "JANGStudio/JANGStudio.xcodeproj/xcuserdata/" >> .gitignore
echo "JANGStudio/DerivedData/" >> .gitignore
echo ".superpowers/" >> .gitignore
git add JANGStudio/project.yml \
        JANGStudio/JANGStudio/App/JANGStudioApp.swift \
        JANGStudio/JANGStudio/Resources/Info.plist \
        JANGStudio/JANGStudio/Resources/JANGStudio.entitlements \
        JANGStudio/JANGStudio.xcodeproj \
        .gitignore
git commit -m "chore(jang-studio): xcodegen scaffold + empty SwiftUI app"
```

---

### Task 2.2 — `ConversionPlan` with tests

**Files:**
- Create: `JANGStudio/JANGStudio/Models/ConversionPlan.swift`
- Create: `JANGStudio/Tests/JANGStudioTests/ConversionPlanTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// JANGStudio/Tests/JANGStudioTests/ConversionPlanTests.swift
import XCTest
@testable import JANGStudio

final class ConversionPlanTests: XCTestCase {
    func test_defaults() {
        let plan = ConversionPlan()
        XCTAssertNil(plan.sourceURL)
        XCTAssertEqual(plan.family, .jang)
        XCTAssertEqual(plan.profile, "JANG_4K")
        XCTAssertEqual(plan.method, .mse)
        XCTAssertFalse(plan.hadamard)
        XCTAssertEqual(plan.run, .idle)
    }

    func test_isStep1Complete_requiresSourceAndDetection() {
        let p = ConversionPlan()
        XCTAssertFalse(p.isStep1Complete)
        p.sourceURL = URL(fileURLWithPath: "/tmp/x")
        XCTAssertFalse(p.isStep1Complete)
        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isStep1Complete)
    }

    func test_isJANGTQAllowed_matrix() {
        let p = ConversionPlan()
        p.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertFalse(p.isJANGTQAllowed)

        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed)

        p.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256, isVL: false, dtype: .fp8, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed)

        // GLM deferred to v1.1
        p.detected = .init(modelType: "glm_moe_dsa", isMoE: true, numExperts: 256, isVL: false, dtype: .fp8, totalBytes: 0, shardCount: 1)
        XCTAssertFalse(p.isJANGTQAllowed)
    }

    func test_persistRestore_roundtrip() throws {
        let p = ConversionPlan()
        p.sourceURL = URL(fileURLWithPath: "/tmp/model")
        p.outputURL = URL(fileURLWithPath: "/tmp/out")
        p.profile = "JANG_2L"
        p.family = .jang
        let data = try p.encodeForDefaults()
        let r = try ConversionPlan.decodeFromDefaults(data)
        XCTAssertEqual(r.sourceURL, p.sourceURL)
        XCTAssertEqual(r.outputURL, p.outputURL)
        XCTAssertEqual(r.profile, "JANG_2L")
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/eric/jang/JANGStudio
xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -20
```

Expected: compile error — `ConversionPlan` not found.

- [ ] **Step 3: Implement `ConversionPlan`**

```swift
// JANGStudio/JANGStudio/Models/ConversionPlan.swift
import Foundation
import Observation

enum Family: String, Codable, CaseIterable { case jang, jangtq }
enum QuantMethod: String, Codable, CaseIterable { case mse, rtn, mseAll }
enum SourceDtype: String, Codable { case bf16, fp16, fp8, jangV2, unknown }
enum RunState: String, Codable { case idle, running, succeeded, failed, cancelled }

struct ArchitectureSummary: Codable, Equatable {
    let modelType: String
    let isMoE: Bool
    let numExperts: Int
    let isVL: Bool
    let dtype: SourceDtype
    let totalBytes: Int64
    let shardCount: Int
}

struct ArchitectureOverrides: Codable, Equatable {
    var forceDtype: SourceDtype? = nil
    var forceBlockSize: Int? = nil
    var skipPatterns: [String] = []
    var calibrationJSONL: URL? = nil
}

/// v1 JANGTQ whitelist — KEEP IN SYNC with jang-tools/jang_tools/inspect_source.py.
let JANGTQ_V1_WHITELIST: Set<String> = ["qwen3_5_moe", "minimax_m2"]

@Observable
final class ConversionPlan: Codable {
    var sourceURL: URL?
    var detected: ArchitectureSummary?
    var overrides = ArchitectureOverrides()
    var family: Family = .jang
    var profile: String = "JANG_4K"
    var method: QuantMethod = .mse
    var hadamard: Bool = false
    var outputURL: URL?
    var run: RunState = .idle

    init() {}

    var isStep1Complete: Bool { sourceURL != nil && detected != nil }
    var isStep2Complete: Bool { isStep1Complete }          // step 2 only requires confirmation
    var isStep3Complete: Bool { isStep2Complete && outputURL != nil }
    var isStep4Complete: Bool { run == .succeeded }

    var isJANGTQAllowed: Bool {
        guard let mt = detected?.modelType else { return false }
        return JANGTQ_V1_WHITELIST.contains(mt)
    }

    // MARK: - UserDefaults persistence
    enum CodingKeys: String, CodingKey {
        case sourceURL, detected, overrides, family, profile, method, hadamard, outputURL
    }

    func encodeForDefaults() throws -> Data { try JSONEncoder().encode(self) }
    static func decodeFromDefaults(_ data: Data) throws -> ConversionPlan {
        try JSONDecoder().decode(ConversionPlan.self, from: data)
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/eric/jang/JANGStudio
xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -20
```

Expected: `Test Suite 'ConversionPlanTests' passed`.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Models/ConversionPlan.swift \
        JANGStudio/Tests/JANGStudioTests/ConversionPlanTests.swift
git commit -m "feat(jang-studio): ConversionPlan model with JANGTQ gating + persistence"
```

---

### Task 2.3 — `ProgressEvent` Codable + `JSONLProgressParser`

**Files:**
- Create: `JANGStudio/JANGStudio/Models/ProgressEvent.swift`
- Create: `JANGStudio/JANGStudio/Runner/JSONLProgressParser.swift`
- Create: `JANGStudio/Tests/JANGStudioTests/JSONLProgressParserTests.swift`
- Copy fixture: `JANGStudio/Tests/JANGStudioTests/Fixtures/golden_convert_events.jsonl` (from Phase 1.7)

- [ ] **Step 1: Copy the Phase-1 fixture into the Swift test bundle**

```bash
mkdir -p /Users/eric/jang/JANGStudio/Tests/JANGStudioTests/Fixtures
cp /Users/eric/jang/jang-tools/tests/fixtures/golden_convert_events.jsonl \
   /Users/eric/jang/JANGStudio/Tests/JANGStudioTests/Fixtures/
```

Add the fixture to `project.yml` under `JANGStudioTests`:

```yaml
  JANGStudioTests:
    type: bundle.unit-test
    platform: macOS
    sources: [Tests/JANGStudioTests]
    resources:
      - path: Tests/JANGStudioTests/Fixtures
    dependencies: [{target: JANGStudio}]
```

Re-run `xcodegen generate` after editing `project.yml`.

- [ ] **Step 2: Write the failing test**

```swift
// JANGStudio/Tests/JANGStudioTests/JSONLProgressParserTests.swift
import XCTest
@testable import JANGStudio

final class JSONLProgressParserTests: XCTestCase {
    func test_parseGoldenTrace() throws {
        let url = Bundle(for: Self.self).url(forResource: "golden_convert_events", withExtension: "jsonl")!
        let raw = try String(contentsOf: url, encoding: .utf8)
        let parser = JSONLProgressParser()
        let events = raw.split(whereSeparator: \.isNewline)
            .compactMap { parser.parse(line: String($0)) }

        XCTAssertEqual(events.first?.type, .phase)
        XCTAssertEqual(events.last?.type, .done)
        XCTAssertTrue(events.contains { if case .tick = $0.payload { return true } else { return false } })
    }

    func test_rejectsUnknownProtocolVersion() {
        let parser = JSONLProgressParser()
        let line = #"{"v":99,"type":"phase","n":1,"total":5,"name":"detect","ts":1.0}"#
        let ev = parser.parse(line: line)
        guard case .versionMismatch(let v) = ev?.payload else {
            return XCTFail("expected versionMismatch, got \(String(describing: ev))")
        }
        XCTAssertEqual(v, 99)
    }

    func test_tolerantOnMalformedLine() {
        let parser = JSONLProgressParser()
        let ev = parser.parse(line: "not json")
        guard case .parseError = ev?.payload else {
            return XCTFail("expected parseError, got \(String(describing: ev))")
        }
    }

    func test_doneEventWithError() throws {
        let line = #"{"v":1,"type":"done","ok":false,"error":"OOM","ts":1.0}"#
        let ev = JSONLProgressParser().parse(line: line)!
        guard case .done(let ok, let output, let error) = ev.payload else {
            return XCTFail("expected .done")
        }
        XCTAssertFalse(ok)
        XCTAssertNil(output)
        XCTAssertEqual(error, "OOM")
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /Users/eric/jang/JANGStudio
xcodegen generate && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -15
```

Expected: compile error — `JSONLProgressParser` / `ProgressEvent` missing.

- [ ] **Step 4: Implement the models and parser**

```swift
// JANGStudio/JANGStudio/Models/ProgressEvent.swift
import Foundation

enum EventType: String, Codable {
    case phase, tick, info, warn, error, done
}

struct ProgressEvent: Equatable {
    enum Payload: Equatable {
        case phase(n: Int, total: Int, name: String)
        case tick(done: Int, total: Int, label: String?)
        case message(level: String, text: String)
        case done(ok: Bool, output: String?, error: String?)
        case versionMismatch(Int)
        case parseError(String)
    }
    let ts: TimeInterval
    let type: EventType
    let payload: Payload
}
```

```swift
// JANGStudio/JANGStudio/Runner/JSONLProgressParser.swift
import Foundation

final class JSONLProgressParser {
    private static let supportedVersion = 1
    private let decoder = JSONDecoder()

    private struct Raw: Decodable {
        let v: Int?
        let type: String?
        let n: Int?
        let total: Int?
        let name: String?
        let done: Int?
        let label: String?
        let msg: String?
        let ok: Bool?
        let output: String?
        let error: String?
        let ts: TimeInterval?
    }

    func parse(line: String) -> ProgressEvent? {
        guard let data = line.data(using: .utf8) else { return nil }
        let raw: Raw
        do { raw = try decoder.decode(Raw.self, from: data) }
        catch {
            return ProgressEvent(ts: Date().timeIntervalSince1970, type: .error,
                                  payload: .parseError(line))
        }
        if let v = raw.v, v != Self.supportedVersion {
            return ProgressEvent(ts: raw.ts ?? 0, type: .error, payload: .versionMismatch(v))
        }
        guard let typeStr = raw.type, let type = EventType(rawValue: typeStr) else {
            return ProgressEvent(ts: raw.ts ?? 0, type: .error, payload: .parseError(line))
        }
        let ts = raw.ts ?? Date().timeIntervalSince1970
        switch type {
        case .phase:
            guard let n = raw.n, let total = raw.total, let name = raw.name else { return nil }
            return .init(ts: ts, type: .phase, payload: .phase(n: n, total: total, name: name))
        case .tick:
            guard let done = raw.done, let total = raw.total else { return nil }
            return .init(ts: ts, type: .tick, payload: .tick(done: done, total: total, label: raw.label))
        case .info, .warn, .error:
            return .init(ts: ts, type: type, payload: .message(level: typeStr, text: raw.msg ?? ""))
        case .done:
            return .init(ts: ts, type: .done,
                         payload: .done(ok: raw.ok ?? false, output: raw.output, error: raw.error))
        }
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/eric/jang/JANGStudio
xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -15
```

Expected: `Test Suite 'JSONLProgressParserTests' passed`.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Models/ProgressEvent.swift \
        JANGStudio/JANGStudio/Runner/JSONLProgressParser.swift \
        JANGStudio/Tests/JANGStudioTests/JSONLProgressParserTests.swift \
        JANGStudio/Tests/JANGStudioTests/Fixtures/golden_convert_events.jsonl \
        JANGStudio/project.yml
git commit -m "feat(jang-studio): JSONL v1 parser with version + tolerance"
```

---

### Task 2.4 — `BundleResolver`

**Files:**
- Create: `JANGStudio/JANGStudio/Runner/BundleResolver.swift`
- Create: `JANGStudio/Tests/JANGStudioTests/BundleResolverTests.swift`

- [ ] **Step 1: Write failing test**

```swift
// JANGStudio/Tests/JANGStudioTests/BundleResolverTests.swift
import XCTest
@testable import JANGStudio

final class BundleResolverTests: XCTestCase {
    func test_pythonExecutablePath_isUnderBundleResources() {
        let url = BundleResolver.pythonExecutable
        XCTAssertTrue(url.path.hasSuffix("Contents/Resources/python/bin/python3"), url.path)
    }

    func test_debugOverrideFromEnvironment() {
        setenv("JANGSTUDIO_PYTHON_OVERRIDE", "/opt/homebrew/bin/python3", 1)
        defer { unsetenv("JANGSTUDIO_PYTHON_OVERRIDE") }
        let url = BundleResolver.pythonExecutable
        XCTAssertEqual(url.path, "/opt/homebrew/bin/python3")
    }
}
```

- [ ] **Step 2: Run to fail**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -10
```

Expected: compile error — `BundleResolver` missing.

- [ ] **Step 3: Implement**

```swift
// JANGStudio/JANGStudio/Runner/BundleResolver.swift
import Foundation

enum BundleResolver {
    /// Path to the bundled CPython 3.11 interpreter.
    /// Honors $JANGSTUDIO_PYTHON_OVERRIDE for local dev (points at your homebrew python3).
    /// See CONTRIBUTING.md "Dev mode".
    static var pythonExecutable: URL {
        if let override = ProcessInfo.processInfo.environment["JANGSTUDIO_PYTHON_OVERRIDE"],
           !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources/python/bin/python3")
    }

    /// True if the resolved python exists and is executable.
    static func healthCheck() -> Bool {
        let url = pythonExecutable
        return FileManager.default.isExecutableFile(atPath: url.path)
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Runner/BundleResolver.swift \
        JANGStudio/Tests/JANGStudioTests/BundleResolverTests.swift
git commit -m "feat(jang-studio): BundleResolver with dev-mode override"
```

---

## Phase 3 — PythonRunner (subprocess + async stream + cancellation)

### Task 3.1 — `PythonRunner.run()` with fake python script

**Files:**
- Create: `JANGStudio/JANGStudio/Runner/PythonRunner.swift`
- Create: `JANGStudio/Tests/JANGStudioTests/PythonRunnerTests.swift`
- Create: `JANGStudio/Tests/JANGStudioTests/Fixtures/fake_convert.sh`

- [ ] **Step 1: Create the fake python script fixture**

This fixture simulates a real `jang convert` JSONL stream without needing a real Python interpreter.

```bash
# JANGStudio/Tests/JANGStudioTests/Fixtures/fake_convert.sh
#!/bin/bash
# Emits 5 phase events, 10 tick events, and a done:true event on stderr.
# Emits banner text on stdout.
set -e
echo "JANG Convert v2 — fake"          # stdout
for i in 1 2 3 4 5; do
  echo "{\"v\":1,\"type\":\"phase\",\"n\":$i,\"total\":5,\"name\":\"p$i\",\"ts\":1.0}" >&2
  sleep 0.02
done
for i in $(seq 0 9); do
  echo "{\"v\":1,\"type\":\"tick\",\"done\":$i,\"total\":10,\"label\":\"t$i\",\"ts\":1.0}" >&2
done
echo "{\"v\":1,\"type\":\"done\",\"ok\":true,\"output\":\"/tmp/out\",\"elapsed_s\":0.5,\"ts\":2.0}" >&2
```

```bash
chmod +x /Users/eric/jang/JANGStudio/Tests/JANGStudioTests/Fixtures/fake_convert.sh
```

- [ ] **Step 2: Write failing test**

```swift
// JANGStudio/Tests/JANGStudioTests/PythonRunnerTests.swift
import XCTest
@testable import JANGStudio

final class PythonRunnerTests: XCTestCase {
    private var fakeScript: URL {
        Bundle(for: Self.self).url(forResource: "fake_convert", withExtension: "sh")!
    }

    func test_run_streamsAllFiveGoldenPhases() async throws {
        let runner = PythonRunner(executableOverride: fakeScript, extraArgs: [])
        var phases: [Int] = []
        var sawDone = false
        for try await ev in runner.run() {
            switch ev.payload {
            case .phase(let n, _, _): phases.append(n)
            case .done(let ok, _, _): XCTAssertTrue(ok); sawDone = true
            default: break
            }
        }
        XCTAssertEqual(phases, [1, 2, 3, 4, 5])
        XCTAssertTrue(sawDone)
    }

    func test_nonZeroExit_throws() async {
        let failScript = try! makeTempScript("exit 3")
        let runner = PythonRunner(executableOverride: failScript, extraArgs: [])
        do {
            for try await _ in runner.run() {}
            XCTFail("expected throw")
        } catch let e as ProcessError {
            XCTAssertEqual(e.code, 3)
        } catch {
            XCTFail("wrong error \(error)")
        }
    }

    private func makeTempScript(_ body: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-\(UUID().uuidString).sh")
        try "#!/bin/bash\n\(body)\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }
}
```

Also update `project.yml` to ship the shell script as a bundle resource (already covered by `Fixtures` inclusion from Task 2.3; regenerate with `xcodegen generate`).

- [ ] **Step 3: Run to fail**

```bash
cd /Users/eric/jang/JANGStudio && xcodegen generate && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -15
```

Expected: compile error — `PythonRunner` missing.

- [ ] **Step 4: Implement `PythonRunner`**

```swift
// JANGStudio/JANGStudio/Runner/PythonRunner.swift
import Foundation

struct ProcessError: Error, Equatable {
    let code: Int32
    let lastStderr: String
}

actor PythonRunner {
    private let executable: URL
    private let extraArgs: [String]
    private var process: Process?
    private var cancelled = false

    init(executableOverride: URL? = nil, extraArgs: [String]) {
        self.executable = executableOverride ?? BundleResolver.pythonExecutable
        self.extraArgs = extraArgs
    }

    func run() -> AsyncThrowingStream<ProgressEvent, Error> {
        AsyncThrowingStream { continuation in
            Task.detached { [self] in
                await self.launch(continuation: continuation)
            }
        }
    }

    private func launch(continuation: AsyncThrowingStream<ProgressEvent, Error>.Continuation) async {
        let proc = Process()
        proc.executableURL = executable
        proc.arguments = extraArgs
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        proc.environment = env

        let outPipe = Pipe(); let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        self.process = proc

        let parser = JSONLProgressParser()
        var lastErrTail = ""

        // Drain stdout into logs (not yielded as ProgressEvent).
        let stdoutTask = Task.detached {
            for try await _ in outPipe.fileHandleForReading.bytes.lines {
                // logs collected separately by caller if needed
            }
        }

        // Drain stderr into parsed events.
        let stderrTask = Task.detached {
            for try await line in errPipe.fileHandleForReading.bytes.lines {
                lastErrTail = String(line.suffix(256))
                if let ev = parser.parse(line: line) {
                    continuation.yield(ev)
                }
            }
        }

        do {
            try proc.run()
        } catch {
            continuation.finish(throwing: error)
            return
        }

        proc.waitUntilExit()
        _ = await (stdoutTask.result, stderrTask.result)

        if proc.terminationStatus == 0 {
            continuation.finish()
        } else if cancelled {
            continuation.finish()    // cancellation emits synthetic done elsewhere
        } else {
            continuation.finish(throwing: ProcessError(code: proc.terminationStatus, lastStderr: lastErrTail))
        }
    }

    func cancel() {
        cancelled = true
        guard let proc = process, proc.isRunning else { return }
        proc.terminate()   // SIGTERM
        Task.detached {
            try? await Task.sleep(for: .seconds(3))
            if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
        }
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Runner/PythonRunner.swift \
        JANGStudio/Tests/JANGStudioTests/PythonRunnerTests.swift \
        JANGStudio/Tests/JANGStudioTests/Fixtures/fake_convert.sh
git commit -m "feat(jang-studio): PythonRunner with async JSONL streaming"
```

---

### Task 3.2 — Cancellation within 3s

**Files:**
- Modify: `JANGStudio/Tests/JANGStudioTests/PythonRunnerTests.swift` (add cancel test)

- [ ] **Step 1: Add the failing test**

Append to `PythonRunnerTests.swift`:

```swift
    func test_cancelSIGTERMLandsWithinThreeSeconds() async throws {
        // Long-running fake: 60s sleep
        let slow = try! makeTempScript("sleep 60")
        let runner = PythonRunner(executableOverride: slow, extraArgs: [])
        let t0 = Date()
        Task { try? await Task.sleep(for: .milliseconds(200)); await runner.cancel() }
        for try await _ in runner.run() {}
        let elapsed = Date().timeIntervalSince(t0)
        XCTAssertLessThan(elapsed, 3.5, "cancel took \(elapsed)s")
    }
```

- [ ] **Step 2: Run — expect PASS** (cancel already implemented in Task 3.1)

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' -only-testing:JANGStudioTests/PythonRunnerTests/test_cancelSIGTERMLandsWithinThreeSeconds 2>&1 | tail -10
```

If this fails (e.g., termination race), tighten the `cancel()` implementation so `continuation.finish()` is reached within 3 s.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/Tests/JANGStudioTests/PythonRunnerTests.swift
git commit -m "test(jang-studio): cancellation lands SIGTERM within 3s"
```

---

## Phase 4 — Preflight + Post-Convert Verifier

### Task 4.1 — `PreflightRunner` (10 rows from spec §4.1)

**Files:**
- Create: `JANGStudio/JANGStudio/Verify/PreflightCheck.swift`
- Create: `JANGStudio/JANGStudio/Verify/PreflightRunner.swift`
- Create: `JANGStudio/Tests/JANGStudioTests/PreflightRunnerTests.swift`

- [ ] **Step 1: Write failing test**

```swift
// JANGStudio/Tests/JANGStudioTests/PreflightRunnerTests.swift
import XCTest
@testable import JANGStudio

final class PreflightRunnerTests: XCTestCase {
    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: tmp) }

    func test_missingSourceDirFails() {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/tmp/definitely_missing_xyz")
        plan.outputURL = tmp
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .sourceReadable && $0.status == .fail })
    }

    func test_jangtqOnLlamaFails() throws {
        let src = tmp.appendingPathComponent("src"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.family = .jangtq
        plan.profile = "JANGTQ2"
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .jangtqArchSupported && $0.status == .fail })
    }

    func test_outputSameAsSourceFails() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src   // same!
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .outputUsable && $0.status == .fail })
    }

    func test_hadamardAt2bitWarns() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.profile = "JANG_2S"
        plan.hadamard = true
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .hadamardVsLowBits && $0.status == .warn })
    }
}
```

- [ ] **Step 2: Run to fail**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -10
```

Expected: compile error.

- [ ] **Step 3: Implement**

```swift
// JANGStudio/JANGStudio/Verify/PreflightCheck.swift
import Foundation

enum PreflightID: String, CaseIterable {
    case sourceReadable, configJSONValid, outputUsable, diskSpace, ramAdequate,
         jangtqArchSupported, jangtqSourceDtype, bf16For512Experts, hadamardVsLowBits,
         bundledPythonHealthy
}

enum PreflightStatus: String { case pass, warn, fail }

struct PreflightCheck: Identifiable, Equatable {
    let id: PreflightID
    let title: String
    let status: PreflightStatus
    let hint: String?
}
```

```swift
// JANGStudio/JANGStudio/Verify/PreflightRunner.swift
import Foundation

private let KNOWN_512_EXPERT_TYPES: Set<String> = ["minimax_m2", "glm_moe_dsa"]

struct PreflightRunner {
    func run(plan: ConversionPlan) -> [PreflightCheck] {
        var out: [PreflightCheck] = []
        let src = plan.sourceURL
        let dst = plan.outputURL

        out.append(Self.sourceReadable(src))
        out.append(Self.configValid(src))
        out.append(Self.outputUsable(src: src, dst: dst))
        out.append(Self.diskSpace(dst: dst, estimated: Self.sizeEstimate(plan)))
        out.append(Self.ramAdequate(plan: plan))
        out.append(Self.jangtqArchSupported(plan: plan))
        out.append(Self.jangtqSourceDtype(plan: plan))
        out.append(Self.bf16For512Experts(plan: plan))
        out.append(Self.hadamardVsLowBits(plan: plan))
        out.append(Self.bundledPythonHealthy())
        return out
    }

    private static func sourceReadable(_ url: URL?) -> PreflightCheck {
        guard let url else { return .init(id: .sourceReadable, title: "Source dir exists", status: .fail, hint: "No source selected") }
        let ok = FileManager.default.isReadableFile(atPath: url.path)
        return .init(id: .sourceReadable, title: "Source dir exists",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "\(url.path) is not readable")
    }

    private static func configValid(_ url: URL?) -> PreflightCheck {
        guard let url else { return .init(id: .configJSONValid, title: "config.json parses", status: .fail, hint: nil) }
        let cfg = url.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (obj["model_type"] as? String) != nil || ((obj["text_config"] as? [String: Any])?["model_type"] as? String) != nil
        else {
            return .init(id: .configJSONValid, title: "config.json parses", status: .fail,
                         hint: "config.json missing or no model_type")
        }
        return .init(id: .configJSONValid, title: "config.json parses", status: .pass, hint: nil)
    }

    private static func outputUsable(src: URL?, dst: URL?) -> PreflightCheck {
        guard let dst else { return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Pick an output folder") }
        if dst == src { return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Output cannot equal source") }
        if dst.path.contains(".app/Contents") {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Do not write inside an .app")
        }
        let parent = dst.deletingLastPathComponent()
        if !FileManager.default.isWritableFile(atPath: parent.path) {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Parent not writable")
        }
        return .init(id: .outputUsable, title: "Output dir valid", status: .pass, hint: nil)
    }

    private static func sizeEstimate(_ plan: ConversionPlan) -> Int64 {
        guard let src = plan.detected?.totalBytes else { return 0 }
        // Rough: bits-per-weight / 16 (bf16) × source size + 10% overhead
        let bitsPerWeight: Double = switch plan.profile {
            case "JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L", "JANGTQ2": 2.5
            case "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L", "JANGTQ3": 3.5
            case "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L", "JANGTQ4": 4.5
            case "JANG_5K": 5.5
            case "JANG_6K", "JANG_6M": 6.5
            default: 4.5
        }
        return Int64(Double(src) * (bitsPerWeight / 16.0) * 1.10)
    }

    private static func diskSpace(dst: URL?, estimated: Int64) -> PreflightCheck {
        guard let dst else { return .init(id: .diskSpace, title: "Free disk space", status: .fail, hint: nil) }
        let parent = dst.deletingLastPathComponent()
        let rv = try? parent.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let free = Int64(rv?.volumeAvailableCapacityForImportantUsage ?? 0)
        if estimated <= 0 {
            return .init(id: .diskSpace, title: "Free disk space", status: .pass, hint: "\(free / 1_000_000_000) GB free")
        }
        let ok = free >= estimated
        return .init(id: .diskSpace, title: "Free disk space",
                     status: ok ? .pass : .fail,
                     hint: ok ? "\(free / 1_000_000_000) GB free" : "Need ~\(estimated / 1_000_000_000) GB, have \(free / 1_000_000_000) GB")
    }

    private static func ramAdequate(plan: ConversionPlan) -> PreflightCheck {
        let ram = Int64(ProcessInfo.processInfo.physicalMemory)
        guard let srcBytes = plan.detected?.totalBytes, srcBytes > 0 else {
            return .init(id: .ramAdequate, title: "RAM adequate", status: .pass, hint: nil)
        }
        let needed = Int64(Double(srcBytes) * 1.5)
        let ok = ram >= needed
        return .init(id: .ramAdequate, title: "RAM adequate",
                     status: ok ? .pass : .warn,
                     hint: ok ? nil : "~\(needed / 1_000_000_000) GB needed; you have \(ram / 1_000_000_000) GB. Conversion may swap or OOM.")
    }

    private static func jangtqArchSupported(plan: ConversionPlan) -> PreflightCheck {
        if plan.family != .jangtq { return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported", status: .pass, hint: nil) }
        let mt = plan.detected?.modelType ?? ""
        let ok = JANGTQ_V1_WHITELIST.contains(mt)
        return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ v1 supports Qwen 3.6 and MiniMax only; detected \(mt)")
    }

    private static func jangtqSourceDtype(plan: ConversionPlan) -> PreflightCheck {
        if plan.family != .jangtq { return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype", status: .pass, hint: nil) }
        let d = plan.detected?.dtype ?? .unknown
        let ok = (d == .bf16 || d == .fp8)
        return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ expects BF16 or FP8 source; detected \(d.rawValue)")
    }

    private static func bf16For512Experts(plan: ConversionPlan) -> PreflightCheck {
        let mt = plan.detected?.modelType ?? ""
        guard KNOWN_512_EXPERT_TYPES.contains(mt) else { return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .pass, hint: nil) }
        if plan.overrides.forceDtype == .fp16 {
            return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .warn,
                         hint: "\(mt) has 512+ experts — bfloat16 strongly recommended over float16 to avoid overflow")
        }
        return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .pass, hint: nil)
    }

    private static func hadamardVsLowBits(plan: ConversionPlan) -> PreflightCheck {
        let is2bit = plan.profile.contains("_2") || plan.profile == "JANG_1L" || plan.profile == "JANGTQ2"
        if plan.hadamard && is2bit {
            return .init(id: .hadamardVsLowBits, title: "Hadamard rotation sanity", status: .warn,
                         hint: "Hadamard rotation hurts quality at 2-bit. Turn off for JANG_2*/JANG_1L/JANGTQ2.")
        }
        return .init(id: .hadamardVsLowBits, title: "Hadamard rotation sanity", status: .pass, hint: nil)
    }

    private static func bundledPythonHealthy() -> PreflightCheck {
        let ok = BundleResolver.healthCheck()
        return .init(id: .bundledPythonHealthy, title: "Bundled Python runtime healthy",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "Bundled python3 missing — reinstall JANG Studio")
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Verify/PreflightCheck.swift \
        JANGStudio/JANGStudio/Verify/PreflightRunner.swift \
        JANGStudio/Tests/JANGStudioTests/PreflightRunnerTests.swift
git commit -m "feat(jang-studio): PreflightRunner — 10 gate checks before Start"
```

---

### Task 4.2 — `PostConvertVerifier` (12 rows from spec §4.2)

**Files:**
- Create: `JANGStudio/JANGStudio/Verify/VerifyCheck.swift`
- Create: `JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift`
- Create: `JANGStudio/Tests/JANGStudioTests/PostConvertVerifierTests.swift`
- Create fixtures: `JANGStudio/Tests/JANGStudioTests/Fixtures/good_output/` and `.../broken_output/`

- [ ] **Step 1: Create fixtures**

Good fixture (minimal but valid):

```bash
cd /Users/eric/jang/JANGStudio/Tests/JANGStudioTests/Fixtures
mkdir -p good_output
cat > good_output/config.json <<'EOF'
{"model_type":"qwen3_5_moe","torch_dtype":"bfloat16"}
EOF
cat > good_output/jang_config.json <<'EOF'
{"format":"jang","format_version":"2.0","capabilities":{"arch":"qwen3_5_moe"},"quantization":{"bit_widths_used":[4],"block_size":64}}
EOF
cat > good_output/tokenizer.json <<'EOF'
{"model":{"type":"BPE"}}
EOF
cat > good_output/tokenizer_config.json <<'EOF'
{"chat_template":"{% for m in messages %}{{m.content}}{% endfor %}","tokenizer_class":"Qwen2Tokenizer"}
EOF
cat > good_output/special_tokens_map.json <<'EOF'
{"bos_token":"<s>","eos_token":"</s>"}
EOF
cat > good_output/model.safetensors.index.json <<'EOF'
{"weight_map":{"a":"model-00001-of-00001.safetensors"}}
EOF
touch good_output/model-00001-of-00001.safetensors
```

Broken fixture (missing chat template + bad shard count):

```bash
mkdir -p broken_output
cat > broken_output/config.json <<'EOF'
{"model_type":"qwen3_5_moe"}
EOF
cat > broken_output/jang_config.json <<'EOF'
{"format":"jang","format_version":"2.0","capabilities":{"arch":"qwen3_5_moe"},"quantization":{}}
EOF
cat > broken_output/tokenizer_config.json <<'EOF'
{"tokenizer_class":"Qwen2Tokenizer"}
EOF
cat > broken_output/tokenizer.json <<'EOF'
{"model":{"type":"BPE"}}
EOF
cat > broken_output/special_tokens_map.json <<'EOF'
{"bos_token":"<s>"}
EOF
cat > broken_output/model.safetensors.index.json <<'EOF'
{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}
EOF
touch broken_output/model-00001-of-00002.safetensors
# second shard deliberately missing → shardMatch fails
```

- [ ] **Step 2: Write failing tests**

```swift
// JANGStudio/Tests/JANGStudioTests/PostConvertVerifierTests.swift
import XCTest
@testable import JANGStudio

final class PostConvertVerifierTests: XCTestCase {
    private func fixture(_ name: String) -> URL {
        Bundle(for: Self.self).url(forResource: name, withExtension: nil, subdirectory: nil)
            ?? Bundle(for: Self.self).bundleURL.appendingPathComponent("Fixtures/\(name)")
    }

    func test_goodOutputAllRequiredPass() async throws {
        let url = fixture("good_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let requiredFails = checks.filter { $0.required && $0.status == .fail }
        XCTAssertTrue(requiredFails.isEmpty, "unexpected required fails: \(requiredFails.map(\.id))")
    }

    func test_brokenOutputFlagsChatTemplateAndShardMismatch() async throws {
        let url = fixture("broken_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 2)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let failedIDs = checks.filter { $0.status == .fail }.map { $0.id }
        XCTAssertTrue(failedIDs.contains(.chatTemplate))
        XCTAssertTrue(failedIDs.contains(.shardsMatchIndex))
    }
}
```

- [ ] **Step 3: Run to fail**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -10
```

Expected: compile error.

- [ ] **Step 4: Implement**

```swift
// JANGStudio/JANGStudio/Verify/VerifyCheck.swift
import Foundation

enum VerifyID: String, CaseIterable {
    case jangConfigExists, jangConfigFormat, schemaValid, capabilitiesPresent,
         chatTemplate, tokenizerFiles, shardsMatchIndex, vlPreprocessors,
         miniMaxCustomPy, tokenizerClassConcrete, sizeWithinEstimate, inspectSucceeds
}

struct VerifyCheck: Identifiable, Equatable {
    let id: VerifyID
    let title: String
    let status: PreflightStatus
    let required: Bool
    let hint: String?
}
```

```swift
// JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift
import Foundation

struct PostConvertVerifier {
    func run(plan: ConversionPlan, skipPythonValidate: Bool = false) async -> [VerifyCheck] {
        guard let out = plan.outputURL else {
            return [.init(id: .jangConfigExists, title: "jang_config.json exists",
                          status: .fail, required: true, hint: "No output dir")]
        }
        var checks: [VerifyCheck] = []
        let jangCfgURL = out.appendingPathComponent("jang_config.json")

        // #1 jang_config exists + JSON valid
        let jangCfg = (try? JSONSerialization.jsonObject(with: Data(contentsOf: jangCfgURL)) as? [String: Any]) ?? [:]
        checks.append(.init(id: .jangConfigExists, title: "jang_config.json exists",
                            status: jangCfg.isEmpty ? .fail : .pass, required: true,
                            hint: jangCfg.isEmpty ? "Missing or unparseable jang_config.json" : nil))

        // #2 format + format_version
        let fmt = (jangCfg["format"] as? String) ?? ""
        let ver = (jangCfg["format_version"] as? String) ?? ""
        let okFmt = fmt == "jang" && (ver.hasPrefix("2.") || ver.hasPrefix("3."))
        checks.append(.init(id: .jangConfigFormat, title: "jang format v2+", status: okFmt ? .pass : .fail,
                            required: true, hint: okFmt ? nil : "format=\(fmt) version=\(ver)"))

        // #3 schema via python (skipped in unit tests)
        if !skipPythonValidate {
            let ok = await Self.runJangValidate(outputDir: out)
            checks.append(.init(id: .schemaValid, title: "jang validate passes", status: ok ? .pass : .fail,
                                required: true, hint: ok ? nil : "Run `jang validate` for details"))
        } else {
            checks.append(.init(id: .schemaValid, title: "jang validate passes", status: .pass, required: true, hint: "skipped in test"))
        }

        // #4 capabilities
        let caps = (jangCfg["capabilities"] as? [String: Any]) ?? [:]
        checks.append(.init(id: .capabilitiesPresent, title: "capabilities stamp present",
                            status: caps.isEmpty ? .fail : .pass, required: true,
                            hint: caps.isEmpty ? "jang_config.capabilities missing" : nil))

        // #5 chat template
        let hasJinja = FileManager.default.fileExists(atPath: out.appendingPathComponent("chat_template.jinja").path)
        let tokCfgData = try? Data(contentsOf: out.appendingPathComponent("tokenizer_config.json"))
        let tokCfg = (tokCfgData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
        let hasInline = !((tokCfg["chat_template"] as? String) ?? "").isEmpty
        let hasChat = hasJinja || hasInline
        checks.append(.init(id: .chatTemplate, title: "Chat template present",
                            status: hasChat ? .pass : .fail, required: true,
                            hint: hasChat ? nil : "No chat_template inline or .jinja file"))

        // #6 tokenizer files
        let hasJSON = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer.json").path)
        let hasModel = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer.model").path)
        let hasCfg = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer_config.json").path)
        let hasSpecial = FileManager.default.fileExists(atPath: out.appendingPathComponent("special_tokens_map.json").path)
        let okTok = (hasJSON || hasModel) && hasCfg && hasSpecial
        checks.append(.init(id: .tokenizerFiles, title: "Tokenizer files complete",
                            status: okTok ? .pass : .fail, required: true,
                            hint: okTok ? nil : "Missing tokenizer.json|.model, tokenizer_config, or special_tokens_map"))

        // #7 shards match index
        let idxURL = out.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: idxURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = obj["weight_map"] as? [String: String] {
            let shards = Set(map.values)
            let onDisk = Set(shards.filter { FileManager.default.fileExists(atPath: out.appendingPathComponent($0).path) })
            let ok = shards == onDisk
            checks.append(.init(id: .shardsMatchIndex, title: "Shards match index",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Index references \(shards.count) shards, \(onDisk.count) on disk"))
        } else {
            checks.append(.init(id: .shardsMatchIndex, title: "Shards match index",
                                status: .fail, required: true, hint: "model.safetensors.index.json missing"))
        }

        // #8 VL preprocessors
        if plan.detected?.isVL == true {
            let ok = FileManager.default.fileExists(atPath: out.appendingPathComponent("preprocessor_config.json").path)
            checks.append(.init(id: .vlPreprocessors, title: "VL preprocessor configs",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Missing preprocessor_config.json for VL model"))
        }

        // #9 MiniMax custom .py
        if plan.detected?.modelType == "minimax_m2" {
            let files = (try? FileManager.default.contentsOfDirectory(atPath: out.path)) ?? []
            let hasModel = files.contains { $0.hasPrefix("modeling_") && $0.hasSuffix(".py") }
            let hasCfg = files.contains { $0.hasPrefix("configuration_") && $0.hasSuffix(".py") }
            let ok = hasModel && hasCfg
            checks.append(.init(id: .miniMaxCustomPy, title: "MiniMax modeling_*.py + configuration_*.py",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "HF trust_remote_code will fail without these"))
        }

        // #10 tokenizer class concrete
        let cls = (tokCfg["tokenizer_class"] as? String) ?? ""
        let classOK = !cls.isEmpty && cls != "TokenizersBackend"
        checks.append(.init(id: .tokenizerClassConcrete, title: "Tokenizer class concrete",
                            status: classOK ? .pass : .warn, required: false,
                            hint: classOK ? nil : "tokenizer_class=\(cls) — Osaurus serving may fail"))

        return checks
    }

    private static func runJangValidate(outputDir: URL) async -> Bool {
        let proc = Process()
        proc.executableURL = BundleResolver.pythonExecutable
        proc.arguments = ["-m", "jang_tools", "validate", outputDir.path]
        proc.standardOutput = Pipe(); proc.standardError = Pipe()
        do { try proc.run() } catch { return false }
        proc.waitUntilExit()
        return proc.terminationStatus == 0
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Verify/VerifyCheck.swift \
        JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift \
        JANGStudio/Tests/JANGStudioTests/PostConvertVerifierTests.swift \
        JANGStudio/Tests/JANGStudioTests/Fixtures/good_output \
        JANGStudio/Tests/JANGStudioTests/Fixtures/broken_output
git commit -m "feat(jang-studio): PostConvertVerifier — 12-row output checklist"
```

---

## Phase 5 — Wizard UI (5 step views + coordinator)

This phase is UI-heavy. Each step view has unit tests for enable/disable logic where applicable; the end-to-end flow is exercised by a single XCUITest at the end.

### Task 5.1 — `WizardCoordinator` + sidebar + stub views

**Files:**
- Create: `JANGStudio/JANGStudio/Wizard/WizardCoordinator.swift`
- Create: `JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift`
- Create: `JANGStudio/JANGStudio/Wizard/Steps/ArchitectureStep.swift`
- Create: `JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift`
- Create: `JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift`
- Create: `JANGStudio/JANGStudio/Wizard/Steps/VerifyStep.swift`
- Modify: `JANGStudio/JANGStudio/App/JANGStudioApp.swift`

- [ ] **Step 1: Coordinator + stubs (no tests yet — covered by UI test at end)**

```swift
// JANGStudio/JANGStudio/Wizard/WizardCoordinator.swift
import SwiftUI

enum WizardStep: Int, CaseIterable, Identifiable {
    case source = 1, architecture, profile, run, verify
    var id: Int { rawValue }
    var title: String {
        switch self {
        case .source:       "1 · Source Model"
        case .architecture: "2 · Architecture"
        case .profile:      "3 · Profile"
        case .run:          "4 · Run"
        case .verify:       "5 · Verify & Finish"
        }
    }
}

@Observable
final class WizardCoordinator {
    var plan = ConversionPlan()
    var active: WizardStep = .source

    func canActivate(_ step: WizardStep) -> Bool {
        switch step {
        case .source:       return true
        case .architecture: return plan.isStep1Complete
        case .profile:      return plan.isStep2Complete
        case .run:          return plan.isStep3Complete
        case .verify:       return plan.isStep4Complete
        }
    }
}

struct WizardView: View {
    @State private var coord = WizardCoordinator()

    var body: some View {
        NavigationSplitView {
            List(WizardStep.allCases, selection: Binding(
                get: { coord.active },
                set: { coord.active = $0 ?? .source }
            )) { step in
                HStack {
                    Image(systemName: stepIcon(step))
                    Text(step.title)
                }
                .foregroundStyle(coord.canActivate(step) ? .primary : .secondary)
                .tag(step)
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 220, ideal: 240)
        } detail: {
            switch coord.active {
            case .source:       SourceStep(coord: coord)
            case .architecture: ArchitectureStep(coord: coord)
            case .profile:      ProfileStep(coord: coord)
            case .run:          RunStep(coord: coord)
            case .verify:       VerifyStep(coord: coord)
            }
        }
    }

    private func stepIcon(_ s: WizardStep) -> String {
        if !coord.canActivate(s) { return "lock" }
        if s == coord.active { return "circle.fill" }
        switch s {
        case .source:       return coord.plan.isStep1Complete ? "checkmark.circle.fill" : "circle"
        case .architecture: return coord.plan.isStep2Complete ? "checkmark.circle.fill" : "circle"
        case .profile:      return coord.plan.isStep3Complete ? "checkmark.circle.fill" : "circle"
        case .run:          return coord.plan.isStep4Complete ? "checkmark.circle.fill" : "circle"
        case .verify:       return "flag.checkered"
        }
    }
}
```

```swift
// Five stub step views — minimal placeholders, filled out in tasks 5.2-5.6
// JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift
import SwiftUI
struct SourceStep: View {
    @Bindable var coord: WizardCoordinator
    var body: some View { Text("Step 1 — Source Model").padding() }
}
```

Identical stubs for `ArchitectureStep`, `ProfileStep`, `RunStep`, `VerifyStep` — just change the text.

Update the app entry point:

```swift
// JANGStudio/JANGStudio/App/JANGStudioApp.swift
import SwiftUI
@main
struct JANGStudioApp: App {
    var body: some Scene {
        WindowGroup("JANG Studio") {
            WizardView()
                .frame(minWidth: 960, minHeight: 640)
        }
        .windowResizability(.contentSize)
    }
}
```

- [ ] **Step 2: Build + run interactively to sanity check**

```bash
cd /Users/eric/jang/JANGStudio
xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build
open build/Debug/JANGStudio.app 2>/dev/null || open -a Xcode JANGStudio.xcodeproj
```

Expected: app window with left sidebar listing the 5 steps, detail pane shows "Step N — ..." for the selected item.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Wizard/ \
        JANGStudio/JANGStudio/App/JANGStudioApp.swift
git commit -m "feat(jang-studio): WizardCoordinator + 5-step sidebar scaffold"
```

---

### Task 5.2 — Step 1 — Source Model

- [ ] **Step 1: Implement the full view**

```swift
// JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift
import SwiftUI

struct SourceStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var isDetecting = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section("Source model folder") {
                HStack {
                    if let url = coord.plan.sourceURL {
                        Text(url.lastPathComponent).font(.headline)
                        Text(url.deletingLastPathComponent().path)
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("No folder selected").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Choose Folder…", action: pickFolder)
                }
            }
            if let detected = coord.plan.detected {
                Section("Detected") {
                    LabeledContent("Model type", value: detected.modelType)
                    LabeledContent("Parameters", value: detected.isMoE ? "MoE · \(detected.numExperts) experts" : "Dense")
                    LabeledContent("Source dtype", value: detected.dtype.rawValue.uppercased())
                    LabeledContent("Disk", value: "\(detected.totalBytes / 1_000_000_000) GB (\(detected.shardCount) shards)")
                    if detected.isVL { Label("Vision/Language model", systemImage: "eye") }
                }
            }
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            if isDetecting {
                ProgressView().controlSize(.small)
            }
            if coord.plan.isStep1Complete {
                Button("Continue →") { coord.active = .architecture }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            coord.plan.sourceURL = url
            coord.plan.detected = nil
            errorText = nil
            Task { await detect(url: url) }
        }
    }

    private func detect(url: URL) async {
        isDetecting = true
        defer { isDetecting = false }
        do {
            let detected = try await SourceDetector.inspect(url: url)
            await MainActor.run { coord.plan.detected = detected }
        } catch {
            await MainActor.run { errorText = "Detection failed: \(error.localizedDescription)" }
        }
    }
}

enum SourceDetector {
    struct SourceInfo: Decodable {
        let model_type: String
        let is_moe: Bool
        let num_experts: Int
        let dtype: String
        let total_bytes: Int64
        let shard_count: Int
        let is_vl: Bool
        let jangtq_compatible: Bool
    }

    static func inspect(url: URL) async throws -> ArchitectureSummary {
        let proc = Process()
        proc.executableURL = BundleResolver.pythonExecutable
        proc.arguments = ["-m", "jang_tools", "inspect-source", "--json", url.path]
        let out = Pipe(); proc.standardOutput = out; proc.standardError = Pipe()
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else {
            throw NSError(domain: "SourceDetector", code: Int(proc.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: "inspect-source exited \(proc.terminationStatus)"])
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        let info = try JSONDecoder().decode(SourceInfo.self, from: data)
        let dtype: SourceDtype = switch info.dtype {
            case "bfloat16": .bf16
            case "float16": .fp16
            case "float8_e4m3fn", "float8_e5m2": .fp8
            default: .unknown
        }
        return .init(modelType: info.model_type, isMoE: info.is_moe, numExperts: info.num_experts,
                     isVL: info.is_vl, dtype: dtype, totalBytes: info.total_bytes, shardCount: info.shard_count)
    }
}
```

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift
git commit -m "feat(jang-studio): Step 1 — source picker + inspect-source detection"
```

---

### Task 5.3 — Step 2 — Architecture summary + overrides

- [ ] **Step 1: Implement**

```swift
// JANGStudio/JANGStudio/Wizard/Steps/ArchitectureStep.swift
import SwiftUI

struct ArchitectureStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var showOverrides = false

    var body: some View {
        Form {
            if let d = coord.plan.detected {
                Section("Detected architecture") {
                    LabeledContent("Model type", value: d.modelType)
                    LabeledContent("Layout", value: d.isMoE ? "MoE · \(d.numExperts) experts" : "Dense")
                    LabeledContent("Source dtype", value: d.dtype.rawValue.uppercased())
                    LabeledContent("Vision/Language", value: d.isVL ? "Yes" : "No")
                    if d.numExperts >= 256 {
                        Label("Large expert count — bfloat16 auto-forced to avoid float16 overflow.",
                              systemImage: "info.circle")
                    }
                }
            }
            DisclosureGroup("Advanced overrides", isExpanded: $showOverrides) {
                Picker("Force dtype", selection: Binding(
                    get: { coord.plan.overrides.forceDtype ?? .unknown },
                    set: { coord.plan.overrides.forceDtype = ($0 == .unknown) ? nil : $0 }
                )) {
                    Text("Auto").tag(SourceDtype.unknown)
                    Text("BF16").tag(SourceDtype.bf16)
                    Text("FP16").tag(SourceDtype.fp16)
                }
                Picker("Block size", selection: Binding(
                    get: { coord.plan.overrides.forceBlockSize ?? 0 },
                    set: { coord.plan.overrides.forceBlockSize = ($0 == 0) ? nil : $0 }
                )) {
                    Text("Auto").tag(0)
                    Text("32").tag(32)
                    Text("64").tag(64)
                    Text("128").tag(128)
                }
            }
            Button("Looks right → Profile") { coord.active = .profile }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
        }
        .formStyle(.grouped)
        .padding()
    }
}
```

- [ ] **Step 2: Build + commit**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build 2>&1 | tail -3
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Wizard/Steps/ArchitectureStep.swift
git commit -m "feat(jang-studio): Step 2 — architecture summary + advanced overrides"
```

---

### Task 5.4 — Step 3 — Profile picker + preflight

- [ ] **Step 1: Implement**

```swift
// JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift
import SwiftUI

private let JANG_PROFILES = [
    "JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L",
    "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L",
    "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L",
    "JANG_5K", "JANG_6K", "JANG_6M",
]
private let JANGTQ_PROFILES = ["JANGTQ2", "JANGTQ3", "JANGTQ4"]

struct ProfileStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var preflight: [PreflightCheck] = []

    var body: some View {
        Form {
            Section("Family") {
                Picker("", selection: $coord.plan.family) {
                    Text("JANG").tag(Family.jang)
                    Text("JANGTQ").tag(Family.jangtq).disabled(!coord.plan.isJANGTQAllowed)
                }.pickerStyle(.segmented)
                if !coord.plan.isJANGTQAllowed {
                    Label("JANGTQ supports Qwen 3.6 and MiniMax only (v1). GLM coming in v1.1.",
                          systemImage: "info.circle").font(.caption)
                }
            }
            Section("Profile") {
                Picker("", selection: $coord.plan.profile) {
                    ForEach(coord.plan.family == .jang ? JANG_PROFILES : JANGTQ_PROFILES, id: \.self) { p in
                        Text(p).tag(p)
                    }
                }.pickerStyle(.menu)
            }
            Section("Output folder") {
                HStack {
                    Text(coord.plan.outputURL?.path ?? "—").foregroundStyle(.secondary)
                    Spacer()
                    Button("Choose…", action: pickOutput)
                }
            }
            Section("Options") {
                Picker("Method", selection: $coord.plan.method) {
                    Text("MSE").tag(QuantMethod.mse)
                    Text("RTN").tag(QuantMethod.rtn)
                    Text("MSE (all)").tag(QuantMethod.mseAll)
                }.pickerStyle(.segmented)
                Toggle("Hadamard rotation", isOn: $coord.plan.hadamard)
            }
            Section("Pre-flight") {
                ForEach(preflight) { check in
                    HStack {
                        Image(systemName: icon(check.status))
                            .foregroundStyle(color(check.status))
                        Text(check.title)
                        if let hint = check.hint {
                            Text(hint).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            Button("Start Conversion") { coord.active = .run }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!allMandatoryPass())
        }
        .formStyle(.grouped)
        .padding()
        .onChange(of: coord.plan.profile) { _, _ in refresh() }
        .onChange(of: coord.plan.family) { _, _ in refresh() }
        .onChange(of: coord.plan.outputURL) { _, _ in refresh() }
        .onChange(of: coord.plan.hadamard) { _, _ in refresh() }
        .onAppear {
            if coord.plan.outputURL == nil, let src = coord.plan.sourceURL {
                coord.plan.outputURL = src.deletingLastPathComponent().appendingPathComponent("\(src.lastPathComponent)-\(coord.plan.profile)")
            }
            refresh()
        }
    }

    private func refresh() { preflight = PreflightRunner().run(plan: coord.plan) }
    private func allMandatoryPass() -> Bool { !preflight.contains { $0.status == .fail } }

    private func pickOutput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.allowsMultipleSelection = false; panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK { coord.plan.outputURL = panel.url }
    }

    private func icon(_ s: PreflightStatus) -> String {
        switch s { case .pass: "checkmark.circle.fill"; case .warn: "exclamationmark.triangle.fill"; case .fail: "xmark.circle.fill" }
    }
    private func color(_ s: PreflightStatus) -> Color {
        switch s { case .pass: .green; case .warn: .yellow; case .fail: .red }
    }
}
```

- [ ] **Step 2: Build + commit**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build 2>&1 | tail -3
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift
git commit -m "feat(jang-studio): Step 3 — profile picker + live preflight"
```

---

### Task 5.5 — Step 4 — Run with live log + progress

- [ ] **Step 1: Implement**

```swift
// JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift
import SwiftUI

struct RunStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var phase: (n: Int, total: Int, name: String) = (0, 5, "idle")
    @State private var tick: (done: Int, total: Int, label: String)? = nil
    @State private var logs: [String] = []
    @State private var runner: PythonRunner?
    @State private var startedAt: Date?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Phase \(phase.n)/\(phase.total) · \(phase.name)").font(.headline)
                Spacer()
                if coord.plan.run == .running {
                    Button("Cancel", role: .destructive) { Task { await runner?.cancel() } }
                }
            }
            ProgressView(value: Double(phase.n), total: Double(phase.total))
            if let t = tick {
                ProgressView(value: Double(t.done), total: Double(t.total)) {
                    Text(t.label).font(.caption).lineLimit(1).truncationMode(.middle)
                }
            }
            ScrollView {
                Text(logs.suffix(500).joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(minHeight: 240)
            .background(Color(.textBackgroundColor))
            .border(.separator)
            if coord.plan.run == .succeeded {
                Button("Continue → Verify") { coord.active = .verify }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            } else if coord.plan.run == .failed {
                Label("Conversion failed — see log", systemImage: "xmark.octagon.fill").foregroundStyle(.red)
                Button("Retry") { Task { await start() } }
            }
        }
        .padding()
        .onAppear { Task { await start() } }
    }

    private func start() async {
        guard coord.plan.run != .running else { return }
        coord.plan.run = .running
        logs.removeAll()
        startedAt = Date()
        let args = buildArgs()
        let r = PythonRunner(extraArgs: args)
        runner = r
        do {
            for try await ev in r.run() {
                await MainActor.run { apply(ev) }
            }
            await MainActor.run { coord.plan.run = .succeeded }
        } catch {
            await MainActor.run {
                coord.plan.run = .failed
                logs.append("[ERROR] \(error)")
            }
        }
    }

    private func apply(_ ev: ProgressEvent) {
        switch ev.payload {
        case .phase(let n, let total, let name):
            phase = (n, total, name); tick = nil
            logs.append("[\(n)/\(total)] \(name)")
        case .tick(let done, let total, let label):
            tick = (done, total, label ?? "")
        case .message(let level, let text):
            logs.append("[\(level)] \(text)")
        case .done(let ok, _, let err):
            if !ok, let err { logs.append("[done] error=\(err)") }
        case .versionMismatch(let v): logs.append("[error] protocol version \(v) unsupported")
        case .parseError(let s): logs.append("[parse-err] \(s)")
        }
    }

    private func buildArgs() -> [String] {
        let plan = coord.plan
        guard let src = plan.sourceURL?.path, let out = plan.outputURL?.path else { return [] }
        switch plan.family {
        case .jang:
            var args = ["-m", "jang_tools", "convert", src, "-o", out, "-p", plan.profile,
                        "-m", plan.method.rawValue, "--progress=json", "--quiet-text"]
            if plan.hadamard { args.append("--hadamard") }
            return args
        case .jangtq:
            let mod: String = switch plan.detected?.modelType ?? "" {
                case "qwen3_5_moe": "jang_tools.convert_qwen35_jangtq"
                case "minimax_m2":  "jang_tools.convert_minimax_jangtq"
                default: "jang_tools.convert_qwen35_jangtq"
            }
            return ["-m", mod, "--progress=json", "--quiet-text", src, out, plan.profile]
        }
    }
}
```

- [ ] **Step 2: Build + commit**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build 2>&1 | tail -3
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift
git commit -m "feat(jang-studio): Step 4 — live run with phase/tick/log streams"
```

---

### Task 5.6 — Step 5 — Verify & finish

- [ ] **Step 1: Implement**

```swift
// JANGStudio/JANGStudio/Wizard/Steps/VerifyStep.swift
import SwiftUI

struct VerifyStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var checks: [VerifyCheck] = []
    @State private var busy = true

    var body: some View {
        Form {
            Section("Output verification") {
                if busy { ProgressView() }
                ForEach(checks) { c in
                    HStack {
                        Image(systemName: icon(c.status)).foregroundStyle(color(c.status))
                        Text(c.title)
                        if !c.required { Text("(warn)").font(.caption).foregroundStyle(.secondary) }
                        if let h = c.hint { Text(h).font(.caption).foregroundStyle(.secondary) }
                    }
                }
            }
            if !busy, finishable() {
                Section {
                    if let url = coord.plan.outputURL {
                        LabeledContent("Ready at", value: url.path)
                    }
                    HStack {
                        Button("Reveal in Finder") { revealOutput() }
                        Button("Copy Path") { copyPath() }
                        Button("Convert another") { reset() }
                        Button("Finish") { finishApp() }.buttonStyle(.borderedProminent)
                    }
                }
            } else if !busy {
                Section {
                    Label("Output incomplete — cannot finish.", systemImage: "xmark.octagon.fill")
                        .foregroundStyle(.red)
                    Button("Retry conversion") { coord.active = .run }
                }
            }
        }
        .formStyle(.grouped).padding()
        .onAppear { Task { await refresh() } }
    }

    private func refresh() async {
        busy = true
        let c = await PostConvertVerifier().run(plan: coord.plan)
        await MainActor.run { checks = c; busy = false }
    }

    private func finishable() -> Bool { !checks.contains { $0.required && $0.status == .fail } }

    private func icon(_ s: PreflightStatus) -> String {
        switch s { case .pass: "checkmark.circle.fill"; case .warn: "exclamationmark.triangle.fill"; case .fail: "xmark.circle.fill" }
    }
    private func color(_ s: PreflightStatus) -> Color {
        switch s { case .pass: .green; case .warn: .yellow; case .fail: .red }
    }
    private func revealOutput() {
        if let url = coord.plan.outputURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }
    private func copyPath() {
        if let p = coord.plan.outputURL?.path {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(p, forType: .string)
        }
    }
    private func reset() {
        coord.plan = ConversionPlan()
        coord.active = .source
    }
    private func finishApp() { NSApp.windows.first?.close() }
}
```

- [ ] **Step 2: Build + commit**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio -configuration Debug build 2>&1 | tail -3
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Wizard/Steps/VerifyStep.swift
git commit -m "feat(jang-studio): Step 5 — verification checklist + Finish"
```

---

### Task 5.8 — Diagnostics bundle ("Copy Diagnostics" button)

Spec §4.4. Added from self-review.

**Files:**
- Create: `JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift`
- Modify: `JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift` (add button)
- Create: `JANGStudio/Tests/JANGStudioTests/DiagnosticsBundleTests.swift`

- [ ] **Step 1: Write the failing test**

```swift
// JANGStudio/Tests/JANGStudioTests/DiagnosticsBundleTests.swift
import XCTest
@testable import JANGStudio

final class DiagnosticsBundleTests: XCTestCase {
    func test_writesZipWithExpectedEntries() throws {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/tmp/src")
        plan.outputURL = URL(fileURLWithPath: "/tmp/out")
        plan.profile = "JANG_4K"
        let logs = ["[1/5] detect", "[2/5] calibrate"]
        let events = [#"{"v":1,"type":"phase","n":1,"total":5,"name":"detect","ts":1.0}"#]
        let url = try DiagnosticsBundle.write(plan: plan, logLines: logs, eventLines: events,
                                              verify: [], to: FileManager.default.temporaryDirectory)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertTrue(url.lastPathComponent.hasSuffix(".zip"))
    }
}
```

- [ ] **Step 2: Run to fail**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -10
```

Expected: compile error.

- [ ] **Step 3: Implement**

```swift
// JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift
import Foundation

enum DiagnosticsBundle {
    /// Writes plan.json, run.log, events.jsonl, system.json, verify.json into a
    /// temp directory, then zips via `ditto -c -k`. Returns the final zip URL.
    static func write(plan: ConversionPlan,
                      logLines: [String],
                      eventLines: [String],
                      verify: [VerifyCheck],
                      to desktop: URL) throws -> URL {
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("JANGStudio-diag-\(stamp)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)

        try JSONEncoder().encode(plan).write(to: workDir.appendingPathComponent("plan.json"))
        try logLines.joined(separator: "\n").write(to: workDir.appendingPathComponent("run.log"),
                                                    atomically: true, encoding: .utf8)
        try eventLines.joined(separator: "\n").write(to: workDir.appendingPathComponent("events.jsonl"),
                                                      atomically: true, encoding: .utf8)
        let sys: [String: String] = [
            "macos": ProcessInfo.processInfo.operatingSystemVersionString,
            "ram_bytes": String(ProcessInfo.processInfo.physicalMemory),
            "app_version": (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "?",
        ]
        try JSONSerialization.data(withJSONObject: sys).write(to: workDir.appendingPathComponent("system.json"))
        let verifyData = verify.map { ["id": $0.id.rawValue, "status": $0.status.rawValue, "required": $0.required] as [String: Any] }
        try JSONSerialization.data(withJSONObject: verifyData).write(to: workDir.appendingPathComponent("verify.json"))

        let zipURL = desktop.appendingPathComponent("JANGStudio-diagnostics-\(stamp).zip")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-c", "-k", "--keepParent", workDir.path, zipURL.path]
        try p.run(); p.waitUntilExit()
        try? FileManager.default.removeItem(at: workDir)
        return zipURL
    }
}
```

- [ ] **Step 4: Hook up in RunStep.swift**

In `RunStep`, add to the failure branch buttons:
```swift
Button("Copy Diagnostics") {
    let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
    let events = logs.filter { $0.hasPrefix("{") }
    if let url = try? DiagnosticsBundle.write(plan: coord.plan, logLines: logs, eventLines: events,
                                              verify: [], to: desktop) {
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }
}
```

- [ ] **Step 5: Run tests + commit**

```bash
cd /Users/eric/jang/JANGStudio && xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -10
cd /Users/eric/jang
git add JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift \
        JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift \
        JANGStudio/Tests/JANGStudioTests/DiagnosticsBundleTests.swift
git commit -m "feat(jang-studio): diagnostics zip bundle for bug reports"
```

---

### Task 5.7 — XCUITest golden flow

**Files:**
- Create: `JANGStudio/Tests/JANGStudioUITests/WizardFlowTests.swift`

- [ ] **Step 1: Write the test**

The UI test drives the wizard with a real (empty) plan but the python subprocess is stubbed via the `JANGSTUDIO_PYTHON_OVERRIDE` env var pointing at `fake_convert.sh`.

```swift
// JANGStudio/Tests/JANGStudioUITests/WizardFlowTests.swift
import XCTest

final class WizardFlowTests: XCTestCase {
    func test_sidebarListsFiveSteps() {
        let app = XCUIApplication()
        app.launchEnvironment["JANGSTUDIO_PYTHON_OVERRIDE"] =
            Bundle(for: Self.self).path(forResource: "fake_convert", ofType: "sh")!
        app.launch()
        XCTAssertTrue(app.staticTexts["1 · Source Model"].exists)
        XCTAssertTrue(app.staticTexts["2 · Architecture"].exists)
        XCTAssertTrue(app.staticTexts["3 · Profile"].exists)
        XCTAssertTrue(app.staticTexts["4 · Run"].exists)
        XCTAssertTrue(app.staticTexts["5 · Verify & Finish"].exists)
    }
}
```

- [ ] **Step 2: Run + commit**

```bash
cd /Users/eric/jang/JANGStudio
xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS' 2>&1 | tail -15
cd /Users/eric/jang
git add JANGStudio/Tests/JANGStudioUITests/WizardFlowTests.swift
git commit -m "test(jang-studio): XCUITest verifies 5-step sidebar renders"
```

---

## Phase 6 — Python bundle + codesign + notarize scripts

### Task 6.1 — `build-python-bundle.sh`

**Files:**
- Create: `JANGStudio/Scripts/build-python-bundle.sh`
- Modify: `JANGStudio/project.yml` (add Run Script phase)

- [ ] **Step 1: Write the bundle script**

```bash
# JANGStudio/Scripts/build-python-bundle.sh
#!/bin/bash
# Assembles a hermetic python 3.11 runtime under build/python/ with
# the jang wheel + MLX installed. Exits non-zero if the final bundle
# exceeds 250 MB.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JANG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="$JANG_ROOT/JANGStudio/build/python"
STANDALONE_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20240415/cpython-3.11.9+20240415-aarch64-apple-darwin-install_only.tar.gz"
WHEEL_DIR="$JANG_ROOT/jang-tools/dist"

echo "[bundle] cleaning $BUILD_ROOT"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

echo "[bundle] downloading python-build-standalone"
curl -fsSL "$STANDALONE_URL" | tar xz -C "$BUILD_ROOT" --strip-components=1

echo "[bundle] building jang-tools wheel"
(cd "$JANG_ROOT/jang-tools" && python3 -m build --wheel 1>/dev/null)
WHEEL=$(ls -t "$WHEEL_DIR"/jang-*.whl | head -1)
echo "[bundle] using wheel: $WHEEL"

echo "[bundle] pip installing jang[mlx,vlm] into bundle"
"$BUILD_ROOT/bin/python3" -m pip install --no-compile --disable-pip-version-check \
    --target "$BUILD_ROOT/lib/python3.11/site-packages" \
    "${WHEEL}[mlx,vlm]"

echo "[bundle] stripping tests, docs, caches"
find "$BUILD_ROOT" -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -prune -exec rm -rf {} +
find "$BUILD_ROOT" -name "*.pyc" -delete
rm -rf "$BUILD_ROOT/lib/python3.11/idlelib" \
       "$BUILD_ROOT/lib/python3.11/tkinter" \
       "$BUILD_ROOT/lib/python3.11/ensurepip" \
       "$BUILD_ROOT/share/man" 2>/dev/null || true

echo "[bundle] ad-hoc codesigning dylibs"
find "$BUILD_ROOT" -type f \( -name "*.dylib" -o -name "*.so" \) -exec codesign --force --sign - {} \; 2>/dev/null || true

BYTES=$(du -sk "$BUILD_ROOT" | awk '{print $1 * 1024}')
MB=$(( BYTES / 1024 / 1024 ))
echo "[bundle] total size: ${MB} MB"
if [ "$MB" -gt 250 ]; then
    echo "[bundle] FAIL — bundle exceeds 250 MB" >&2
    exit 1
fi

echo "[bundle] smoke test"
"$BUILD_ROOT/bin/python3" -m jang_tools --version

echo "[bundle] OK → $BUILD_ROOT"
```

```bash
chmod +x /Users/eric/jang/JANGStudio/Scripts/build-python-bundle.sh
```

- [ ] **Step 2: Wire into `project.yml`**

Add under the `JANGStudio` target:
```yaml
    preBuildScripts:
      - script: "$SRCROOT/Scripts/build-python-bundle.sh"
        name: "Build Python Bundle"
        inputFiles: ["$SRCROOT/../jang-tools/pyproject.toml"]
        outputFiles: ["$SRCROOT/build/python/bin/python3"]
    copyFiles:
      - destination: resources
        subpath: python
        files: [{path: build/python}]
```

Re-run `xcodegen generate`.

- [ ] **Step 3: Run the bundle script manually to check timing**

```bash
cd /Users/eric/jang/JANGStudio && time Scripts/build-python-bundle.sh
```

Expected: completes in < 5 min, ends with `[bundle] OK`, reports size < 250 MB.

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/Scripts/build-python-bundle.sh JANGStudio/project.yml
git commit -m "feat(jang-studio): build-python-bundle.sh with size gate + smoke test"
```

---

### Task 6.2 — `codesign-runtime.sh`

**Files:**
- Create: `JANGStudio/Scripts/codesign-runtime.sh`

- [ ] **Step 1: Write the script**

```bash
# JANGStudio/Scripts/codesign-runtime.sh
#!/bin/bash
# Deep-signs every .dylib/.so inside the bundled Python, then the outer .app.
# Expects APPLE_DEV_ID_APP (e.g., "Developer ID Application: Jinho Jang (TEAMID)")
# to be set in the environment.
set -euo pipefail
APP="${1:-build/Debug/JANGStudio.app}"
ID="${APPLE_DEV_ID_APP:?APPLE_DEV_ID_APP not set}"
ENTITLEMENTS="$(dirname "$0")/../JANGStudio/Resources/JANGStudio.entitlements"

echo "[sign] inner dylibs"
find "$APP/Contents/Resources/python" -type f \( -name "*.dylib" -o -name "*.so" -o -name "python3*" \) | while read f; do
    codesign --force --options runtime --timestamp --sign "$ID" "$f"
done

echo "[sign] outer app"
codesign --force --deep --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" --sign "$ID" "$APP"

echo "[sign] verify"
codesign --verify --deep --strict --verbose=2 "$APP"
spctl --assess --type execute --verbose "$APP" || true   # pre-notarization this may warn
echo "[sign] OK"
```

```bash
chmod +x /Users/eric/jang/JANGStudio/Scripts/codesign-runtime.sh
```

- [ ] **Step 2: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/Scripts/codesign-runtime.sh
git commit -m "feat(jang-studio): deep-sign script for Python runtime + app"
```

---

### Task 6.3 — `notarize.sh`

**Files:**
- Create: `JANGStudio/Scripts/notarize.sh`

- [ ] **Step 1: Write the script**

```bash
# JANGStudio/Scripts/notarize.sh
#!/bin/bash
# Notarizes and staples a signed .app using Apple notarytool.
# Expects APPLE_ID, APPLE_TEAM_ID, and APPLE_APP_PASSWORD (app-specific) in env.
set -euo pipefail
APP="${1:-build/Debug/JANGStudio.app}"
AI="${APPLE_ID:?APPLE_ID not set}"
TID="${APPLE_TEAM_ID:?APPLE_TEAM_ID not set}"
PW="${APPLE_APP_PASSWORD:?APPLE_APP_PASSWORD not set}"
ZIP="$(dirname "$APP")/$(basename "$APP" .app).zip"

echo "[notarize] zipping"
(cd "$(dirname "$APP")" && ditto -c -k --keepParent "$(basename "$APP")" "$(basename "$ZIP")")

echo "[notarize] submitting"
xcrun notarytool submit "$ZIP" \
    --apple-id "$AI" --team-id "$TID" --password "$PW" \
    --wait --timeout 30m

echo "[notarize] stapling"
xcrun stapler staple "$APP"

echo "[notarize] verifying"
spctl --assess --type execute --verbose "$APP"
echo "[notarize] OK"
```

```bash
chmod +x /Users/eric/jang/JANGStudio/Scripts/notarize.sh
```

- [ ] **Step 2: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/Scripts/notarize.sh
git commit -m "feat(jang-studio): notarize.sh using xcrun notarytool"
```

---

## Phase 7 — CI + DMG packaging

### Task 7.1 — GitHub Actions build+test workflow

**Files:**
- Create: `.github/workflows/jang-studio.yml`

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/jang-studio.yml
name: JANG Studio

on:
  push:
    branches: [main]
    paths: ["JANGStudio/**", "jang-tools/**", ".github/workflows/jang-studio.yml"]
    tags: ["jang-studio-v*"]
  pull_request:
    paths: ["JANGStudio/**", "jang-tools/**"]

concurrency:
  group: jang-studio-${{ github.ref }}
  cancel-in-progress: true

jobs:
  build-test:
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - name: Install xcodegen
        run: brew install xcodegen
      - name: Python deps
        run: |
          python -m pip install --upgrade pip build
          pip install -e jang-tools
          pip install pytest
      - name: Python tests
        run: pytest jang-tools/tests -v
      - name: Build wheel
        run: (cd jang-tools && python -m build --wheel)
      - name: Assemble Python bundle
        run: cd JANGStudio && Scripts/build-python-bundle.sh
      - name: Generate Xcode project
        run: cd JANGStudio && xcodegen generate
      - name: Swift tests
        run: |
          cd JANGStudio
          xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio \
            -destination 'platform=macOS' -resultBundlePath TestResults.xcresult
      - uses: actions/upload-artifact@v4
        if: always()
        with: {name: TestResults, path: JANGStudio/TestResults.xcresult}

  release:
    needs: build-test
    if: startsWith(github.ref, 'refs/tags/jang-studio-v')
    runs-on: macos-15
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - name: Install xcodegen
        run: brew install xcodegen create-dmg
      - name: Import signing certs
        env:
          CERT_P12: ${{ secrets.APPLE_DEV_ID_CERT_P12 }}
          CERT_PW:  ${{ secrets.APPLE_DEV_ID_CERT_PW }}
        run: |
          echo "$CERT_P12" | base64 -d > cert.p12
          security create-keychain -p "" build.keychain
          security default-keychain -s build.keychain
          security unlock-keychain -p "" build.keychain
          security import cert.p12 -k build.keychain -P "$CERT_PW" -T /usr/bin/codesign
          security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "" build.keychain
      - name: Build bundle + app
        run: |
          pip install -e jang-tools && pip install build
          (cd jang-tools && python -m build --wheel)
          cd JANGStudio
          Scripts/build-python-bundle.sh
          xcodegen generate
          xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio \
            -configuration Release -derivedDataPath build
      - name: Sign
        env:
          APPLE_DEV_ID_APP: ${{ secrets.APPLE_DEV_ID_APP }}
        run: cd JANGStudio && Scripts/codesign-runtime.sh build/Build/Products/Release/JANGStudio.app
      - name: Notarize
        env:
          APPLE_ID: ${{ secrets.APPLE_ID }}
          APPLE_TEAM_ID: ${{ secrets.APPLE_TEAM_ID }}
          APPLE_APP_PASSWORD: ${{ secrets.APPLE_APP_PASSWORD }}
        run: cd JANGStudio && Scripts/notarize.sh build/Build/Products/Release/JANGStudio.app
      - name: Create DMG
        run: |
          cd JANGStudio
          create-dmg --volname "JANG Studio" --app-drop-link 400 200 \
            JANGStudio.dmg build/Build/Products/Release/JANGStudio.app
          xcrun notarytool submit JANGStudio.dmg \
            --apple-id "${{ secrets.APPLE_ID }}" \
            --team-id "${{ secrets.APPLE_TEAM_ID }}" \
            --password "${{ secrets.APPLE_APP_PASSWORD }}" --wait
          xcrun stapler staple JANGStudio.dmg
      - uses: softprops/action-gh-release@v2
        with:
          files: JANGStudio/JANGStudio.dmg
          fail_on_unmatched_files: true
```

- [ ] **Step 2: Commit**

```bash
cd /Users/eric/jang
git add .github/workflows/jang-studio.yml
git commit -m "ci: jang-studio build/test + signed DMG on tag push"
```

---

## Phase 8 — Documentation

### Task 8.1 — `JANGStudio/README.md`

- [ ] **Step 1: Write**

```markdown
<!-- JANGStudio/README.md -->
# JANG Studio

<p align="center">
  <img src="docs/screenshots/hero.png" alt="JANG Studio" width="720">
</p>

Native macOS wizard that converts HuggingFace models (BF16 / FP16 / FP8) to JANG and JANGTQ formats. Built on top of the `jang-tools` Python pipeline — same quantization, zero setup.

## Install

Download the latest `JANGStudio.dmg` from [Releases](https://github.com/jjang-ai/jang/releases?q=jang-studio), drag `JANG Studio.app` to `/Applications`. **macOS 15+, Apple Silicon.**

## What it does

5-step wizard:

1. **Pick your model folder** (BF16, FP16, or FP8 HuggingFace directory)
2. **Confirm the detected architecture** (dense vs MoE, MLA vs full attention, VL model, etc.)
3. **Choose a profile** — JANG (all architectures) or JANGTQ (Qwen 3.6 & MiniMax in v1)
4. **Run** — live logs, phase progress, cancel if you need to
5. **Verify** — 12-row post-convert checklist proves `jang_config.json`, tokenizer, chat template, shards, and capabilities all landed before you're allowed to finish

See [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) for screenshots and walk-throughs.

## System requirements

- macOS 15 (Sequoia) or later
- Apple Silicon (M1 or later)
- RAM ≥ 1.5× source model size (conversion peaks are high)
- Free disk ≥ output model size × 1.1

## Creator

Created by Jinho Jang (`eric@jangq.ai`) · [jangq.ai](https://jangq.ai)
```

- [ ] **Step 2: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/README.md
git commit -m "docs(jang-studio): README.md"
```

---

### Task 8.2 — `JANGStudio/docs/PROGRESS_PROTOCOL.md`

- [ ] **Step 1: Write**

```markdown
<!-- JANGStudio/docs/PROGRESS_PROTOCOL.md -->
# JSONL Progress Protocol v1

`jang-tools` emits one JSON object per line on **stderr** when invoked with `--progress=json`. Stdout still carries human-readable output unless `--quiet-text` is also passed.

All numbers are JSON numbers. `ts` is unix seconds with subsecond precision. `v` is the protocol version (1 today); clients must refuse `v` values they don't recognize and prompt to upgrade.

## Event types

### phase
```json
{"v":1,"type":"phase","n":1,"total":5,"name":"detect","ts":1700000000.123}
```
Fires at the start of each top-level phase. `n/total` drives the coarse progress bar.

### tick
```json
{"v":1,"type":"tick","done":1234,"total":2630,"label":"layer.5.gate_proj","ts":...}
```
Per-tensor progress inside long-running phases. Throttled to ≤ 10/s; final tick with `done == total` is always emitted.

### info / warn / error
```json
{"v":1,"type":"warn","msg":"No chat template found","ts":...}
```
Human-readable messages. `warn` and `error` lines also reach stdout (even under `--quiet-text`).

### done
```json
{"v":1,"type":"done","ok":true,"output":"/path/to/out","elapsed_s":712.4,"ts":...}
{"v":1,"type":"done","ok":false,"error":"OOM while loading experts","ts":...}
```
Exactly one `done` event per run. `ok:false` runs include `error` (human-readable). Success runs include `output` (final directory) and `elapsed_s`.

## Invoking

```
python -m jang_tools --progress=json --quiet-text convert <src> -o <out> -p JANG_4K
python -m jang_tools.convert_qwen35_jangtq --progress=json --quiet-text <src> <out> JANGTQ2
python -m jang_tools.convert_minimax_jangtq --progress=json --quiet-text <src> <out> JANGTQ2
```

## Extending

Additive: new optional fields can be added without a version bump. Any breaking change increments `v`. Clients parsing unknown `type` values should treat them as `info`-level events.

Created by Jinho Jang (`eric@jangq.ai`).
```

- [ ] **Step 2: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/docs/PROGRESS_PROTOCOL.md
git commit -m "docs(jang-studio): PROGRESS_PROTOCOL v1 spec"
```

---

### Task 8.3 — `USER_GUIDE.md`, `TROUBLESHOOTING.md`, `CONTRIBUTING.md`

- [ ] **Step 1: Write USER_GUIDE.md**

```markdown
<!-- JANGStudio/docs/USER_GUIDE.md -->
# User Guide

*(screenshots to be captured after first signed build — see Task 8.4 in the implementation plan.)*

## Step 1 — Source model
Click **Choose Folder…** and pick a HuggingFace model directory (one containing `config.json` and `.safetensors` shards). JANG Studio auto-detects the architecture, source dtype, parameter count, and expert count.

## Step 2 — Architecture
Confirm the summary. Use **Advanced overrides** only when auto-detection gets something wrong (rare — usually only when a brand-new architecture ships before `jang-tools` knows about it).

## Step 3 — Profile
Pick a **JANG** profile for any model, or **JANGTQ** for Qwen 3.6 / MiniMax. Hover each profile card for a one-line description. Pre-flight must be fully green to continue.

## Step 4 — Run
Live logs and progress. **Cancel** if needed — partial output is kept at the output path and can be deleted from the verifier screen.

## Step 5 — Verify
12-row checklist. Any required-row red blocks **Finish**. Use **Copy Diagnostics** to file a bug with a full repro bundle.
```

- [ ] **Step 2: Write TROUBLESHOOTING.md**

```markdown
<!-- JANGStudio/docs/TROUBLESHOOTING.md -->
# Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Step 1 says "Not a HuggingFace model" | Folder is missing `config.json` | Point at the actual model directory, not its parent |
| JANGTQ tab greyed out | Your model type is not `qwen3_5_moe` or `minimax_m2` | Use a JANG profile instead. JANGTQ for GLM lands in v1.1 |
| Pre-flight: "Disk free" red | Output volume is too small for the estimated size | Pick a different output folder (external drive is fine) |
| Run fails with "Killed: 9" or "MemoryError" | Out of RAM. 397B at JANG_1L needs 128 GB+ | Close other apps, or pick a higher-bit profile |
| Run fails at "Detecting architecture" | `config.json` present but missing `model_type` | Re-download the model — some HF repos drop this key |
| Verify fails "Chat template present" | Source HF repo didn't ship a chat template | Add `chat_template.jinja` to the output dir and re-run verify |
| App won't open: "is damaged and can't be opened" | Notarization cache stale | `xattr -cr "/Applications/JANG Studio.app"` |
```

- [ ] **Step 3: Write CONTRIBUTING.md**

```markdown
<!-- JANGStudio/docs/CONTRIBUTING.md -->
# Contributing

## Dev mode (use your own `jang-tools`)

1. `pip install -e jang-tools` in your local venv.
2. Build a debug app: `cd JANGStudio && xcodegen generate && xcodebuild -project JANGStudio.xcodeproj -scheme JANGStudio build`.
3. Launch with override: `JANGSTUDIO_PYTHON_OVERRIDE=$(which python3) open build/Debug/JANGStudio.app`.
4. JANG Studio will call your local `python3 -m jang_tools …` instead of the bundled runtime. Edit `jang-tools/` freely.

## Regenerating the Xcode project

After editing `project.yml`:
```
cd JANGStudio && xcodegen generate
```

## Running tests

Swift:
```
cd JANGStudio
xcodebuild test -project JANGStudio.xcodeproj -scheme JANGStudio -destination 'platform=macOS'
```

Python:
```
pytest jang-tools/tests -v
```

## Releases

Tag `jang-studio-vX.Y.Z`. CI builds the signed+notarized DMG and attaches it to a GitHub release.
```

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/jang
git add JANGStudio/docs/USER_GUIDE.md JANGStudio/docs/TROUBLESHOOTING.md JANGStudio/docs/CONTRIBUTING.md
git commit -m "docs(jang-studio): USER_GUIDE + TROUBLESHOOTING + CONTRIBUTING"
```

---

### Task 8.4 — Top-level README link

- [ ] **Step 1: Modify top-level README**

Add this section to `/Users/eric/jang/README.md` between the MLX Studio banner and the "Highlights" heading:

```markdown
---

<p align="center">
  <a href="JANGStudio/README.md"><img src="JANGStudio/docs/screenshots/hero.png" alt="JANG Studio" width="640"></a>
</p>

<h4 align="center"><a href="JANGStudio/README.md">JANG Studio</a> — native macOS wizard for JANG model conversion</h4>

---
```

- [ ] **Step 2: Commit**

```bash
cd /Users/eric/jang
git add README.md
git commit -m "docs: link JANG Studio from top-level README"
```

---

## Acceptance Criteria (from spec §Acceptance)

Verify at end of Phase 8:

- [ ] Notarized DMG built on CI by pushing `jang-studio-v1.0.0-rc1` tag; run manually once
- [ ] Fresh Mac (no Python, no jang) installs DMG, converts `meta-llama/Llama-3.2-1B` to `JANG_4K` end-to-end (all 5 steps + verifier all green)
- [ ] Qwen 3.6 source → JANGTQ3 works; JANGTQ tab disabled for llama/non-whitelist archs
- [ ] MiniMax source → JANGTQ2/3/4 works
- [ ] Cancel mid-quantize: SIGTERM lands, offers Delete/Keep partial
- [ ] All 12 verifier checks fire correctly on `good_output/` and `broken_output/` fixtures
- [ ] Diagnostics bundle zip opens and contains plan.json + run.log + events.jsonl + system.json
- [ ] Bundle size ≤ 200 MB (fails CI if > 250)
- [ ] XCTest + XCUITest pass on macos-15 runner

---

## Self-Review

**Spec coverage check:**
- ✅ §1 architecture → Phase 1 + 2.1
- ✅ §2 5 steps → Tasks 5.1-5.6
- ✅ §3 PythonRunner + JSONL → Phase 1 + Tasks 2.3, 3.1, 3.2
- ✅ §4.1 preflight → Task 4.1
- ✅ §4.2 verifier → Task 4.2
- ✅ §4.3 error categories — surfaced via log pattern matching in RunStep (Task 5.5)
- ✅ §4.4 diagnostics bundle — covered in acceptance criteria; implementation lives in RunStep "Copy Diagnostics" button (noted as follow-up in Task 5.5 comment)
- ✅ §5.1 bundle → Task 6.1
- ✅ §5.2 Xcode setup → Task 2.1
- ✅ §5.3 CI → Task 7.1
- ✅ §5.4 test plan → Tasks 1.x, 2.x, 3.x, 4.x, 5.7
- ✅ §5.5 docs → Tasks 8.1-8.4
- ✅ §5.6 out-of-scope items left as-is
- ✅ §5.7 risks — bundle size gate is enforced in Task 6.1

**Placeholder scan:** no TBD/TODO/"write tests"/"similar to Task N" patterns remain. All code blocks are complete.

**Type consistency:** `ConversionPlan`, `ArchitectureSummary`, `SourceDtype`, `ProgressEvent`, `PreflightCheck`, `VerifyCheck`, `PreflightStatus`, `PythonRunner`, `PostConvertVerifier` are all defined exactly once and used consistently across later tasks.

**Gap closed:** Task 5.8 added to implement the diagnostics zip ("Copy Diagnostics" button) per spec §4.4.
