"""M200 (iter 137): regression invariants for removed Settings-UI lies.

A Settings-UI lie is a field that: (a) appears in a settings-window
control (Stepper, TextField, Toggle), (b) persists to UserDefaults,
but (c) has NO downstream consumer that changes runtime behavior.
From `feedback_dont_lie_to_user.md`: the app must not show controls
that don't do anything.

iter-137 angle-I audit (Setting-actually-affects-behavior) found:
  - `defaultCalibrationSamples` — UI Stepper 64...1024 in Settings →
    Advanced. Persisted. Never reached convert subprocess: the
    `jang convert` argparse doesn't accept `--samples` (that flag is
    on the `profile` subcommand for TurboSmelt routing profile, not
    convert), and `CLIArgsBuilder.args(for:)` never emitted one.
    Flipping 64 → 1024 had zero measurable effect.
  - `calibrationJSONL` (ArchitectureOverrides field) — declared in
    ConversionPlan.swift; zero references anywhere else in the
    JANGStudio codebase. Pre-UI-surface lie (if any wizard step had
    exposed a picker, it would have been a lie; no picker yet, so
    technically dead code rather than a lie, but bundled for cleanup).

These tests pin the REMOVAL. Reintroducing the field without first
plumbing a downstream consumer (CLIArgsBuilder → `--samples` argv →
jang convert argparse → calibrate.py n_samples) would require
re-editing this invariant, which is the signal to add the plumbing
FIRST + prove end-to-end.

If a future quant method (AWQ, GPTQ, SmoothQuant) needs user-tunable
calibration samples, reintroduce the field via:
  1. Add `--samples N` to p_convert's argparse in
     `jang-tools/jang_tools/__main__.py`.
  2. Thread into calibrate.py's `n_samples` parameter.
  3. Add `calibrationSamples: Int` to ConversionPlan (not AppSettings
     — the plan is the per-conversion carrier).
  4. Emit `--samples <plan.calibrationSamples>` from CLIArgsBuilder.
  5. Expose in the Wizard's ProfileStep (not Settings — method-level
     knob, not app-level).
  6. Add a test that proves the CLI argv contains `--samples N`.
  7. Then update THIS invariant to document the new wired field.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
APP_SETTINGS = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Models" / "AppSettings.swift"
CONVERSION_PLAN = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Models" / "ConversionPlan.swift"
CLI_ARGS_BUILDER = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Runner" / "CLIArgsBuilder.swift"
SETTINGS_WINDOW = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "SettingsWindow.swift"


def _live_code(path: Path) -> str:
    """Return source with `//` and `///` single-line comments stripped.
    Preserves multi-line blocks like docstrings-in-code because Swift
    doesn't use `/** */`-style doc blocks much, and substring searches
    on live vs commented content are the main concern here.

    Same technique as iter-135 M198's comment-stripping for "no bad
    pattern in code" invariants — educational comments that MENTION
    the removed identifier must not false-positive the check."""
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_default_calibration_samples_not_declared_in_live_code():
    """M200 regression: the `defaultCalibrationSamples` field must not
    reappear in AppSettings.swift live code. The M200 removal comment
    MENTIONS the name in an educational `//` block; comment-strip
    prevents false-positive."""
    live = _live_code(APP_SETTINGS)
    assert "defaultCalibrationSamples" not in live, (
        "M200 regression: `defaultCalibrationSamples` is back in "
        "AppSettings.swift live code. Pre-M200 this was a Settings-UI "
        "lie — the Stepper persisted a value that never reached the "
        "convert subprocess (jang convert doesn't accept --samples). "
        "Do not reintroduce this field unless you FIRST plumbed a "
        "downstream consumer end-to-end. See M200 entry in "
        "AUDIT_CHECKLIST.md for the reintroduction protocol."
    )


def test_default_calibration_samples_not_referenced_in_settings_window():
    """Pin that SettingsWindow.swift's Stepper + observation-tracking
    references are gone. If the field comes back in AppSettings but
    no UI exposes it, it's just a dead field — but a UI reference
    without the backing field is a compile error, and a UI reference
    WITH the backing field is a lie returning. Both are caught by
    pinning the UI site."""
    live = _live_code(SETTINGS_WINDOW)
    assert "defaultCalibrationSamples" not in live, (
        "M200 regression: SettingsWindow.swift references "
        "`defaultCalibrationSamples` in live code. The Stepper UI was "
        "removed in M200 because it had no downstream consumer."
    )


def test_calibration_jsonl_not_declared_in_live_code():
    """M200 regression: ArchitectureOverrides.calibrationJSONL must
    not reappear. Pre-M200 it was dead (never read, never written by
    any UI). If a future wizard step needs user-supplied calibration
    data, add a URL picker + consumer in the same iter."""
    live = _live_code(CONVERSION_PLAN)
    assert "calibrationJSONL" not in live, (
        "M200 regression: `calibrationJSONL` is back in "
        "ConversionPlan.swift live code. Zero downstream consumers "
        "existed at removal time. Do not reintroduce without a UI "
        "picker AND a CLIArgsBuilder emission."
    )


def test_cli_args_builder_doesnt_emit_samples_flag():
    """Pin that CLIArgsBuilder.args(for:) doesn't emit `--samples`.
    If someone adds the flag emit without also adding the argparse
    on the Python side, the convert subprocess would crash with
    'unrecognized arguments: --samples 256'. Defense-in-depth against
    a partial reintroduction."""
    live = _live_code(CLI_ARGS_BUILDER)
    # Simple substring check — if this changes, either (a) the flag
    # was added correctly (update this invariant + prove argparse
    # side also accepts it), or (b) someone added it partially.
    assert "--samples" not in live, (
        "M200 regression: CLIArgsBuilder emits `--samples` but the "
        "jang convert argparse doesn't accept that flag. Running the "
        "converter would crash with 'unrecognized arguments: "
        "--samples N'. Either finish the plumbing on the Python side "
        "(p_convert.add_argument('--samples', ...) + calibrate.py "
        "threading) or revert the Swift-side emission."
    )


def test_jang_convert_argparse_status_documented():
    """Meta-pin: document in THIS test file that `jang convert` does
    not accept `--samples`. If that ever changes (future quant method
    adds calibration), update this assertion AND simultaneously
    reintroduce the Swift-side field with a paired test. The two
    sides must move together."""
    main_py = REPO_ROOT / "jang-tools" / "jang_tools" / "__main__.py"
    content = main_py.read_text(encoding="utf-8")
    # Find the p_convert block.
    start = content.find('p_convert = subparsers.add_parser("convert"')
    assert start != -1, "p_convert block not found in __main__.py"
    # Slice to the next p_* block or set_defaults.
    end = content.find("p_convert.set_defaults", start)
    assert end != -1
    block = content[start:end]
    # Current state (iter 137): no --samples, no -n flag on convert.
    # If this test fires because someone ADDED --samples, they should
    # ALSO have reintroduced the Swift side + added a downstream plumbing
    # test. Document that coupling here.
    for bad in ("--samples", "'-n',", '"-n",'):
        assert bad not in block, (
            f"M200 status drift: `jang convert` now accepts {bad!r}. "
            f"If this was intentional (new quant method needs user-"
            f"tunable calibration), you must ALSO: (1) reintroduce "
            f"`defaultCalibrationSamples` or a per-plan "
            f"`calibrationSamples` field in Swift, (2) wire it through "
            f"CLIArgsBuilder, (3) expose a UI control, (4) write an "
            f"end-to-end test asserting the argv contains `--samples "
            f"<value>`. Then update this invariant to reflect the new "
            f"wired state."
        )
