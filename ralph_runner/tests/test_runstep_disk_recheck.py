"""M204 (iter 139): RunStep final disk-space re-check invariant.

Pre-M204, PreflightRunner's disk-space gate ran in Step 3 (Profile) but
nothing re-verified disk space before Step 4 (Run) auto-spawned the
Python subprocess. Between profile-selection and start-convert, minutes
could pass while the user reads the preview card — during which a
concurrent download, Time Machine snapshot, or other process could
consume free space. Result: convert ran for 20+ minutes and failed
with ENOSPC mid-shard, leaving the user with partial output and no
warning up-front.

M204 adds a cheap final re-check inside RunStep.start() that runs
AFTER the Profile preflight but BEFORE spawning the subprocess. It
reuses PreflightRunner.estimateOutputBytes for math consistency across
the two gates.

This test pins the re-check shape via source inspection (same pattern
as M197 cross-language parity + M198 frontend invariants).
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
RUN_STEP = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "Steps" / "RunStep.swift"


def _live_code(path: Path) -> str:
    """Strip `//` + `///` single-line comments (iter-135 M198 technique)."""
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _start_fn_body() -> str:
    """Return the source of RunStep.start(). Slices from the `private
    func start() async` signature forward to the end of its braces."""
    content = RUN_STEP.read_text(encoding="utf-8")
    start = content.find("private func start() async")
    assert start != -1, "start() method not found in RunStep.swift"
    # Bound by the next `private func` at the same indent level. Simple.
    rest = content[start:]
    end = rest.find("\n    private func ", 1)
    if end == -1:
        end = len(rest)
    return rest[:end]


def test_runstep_imports_profiles_service():
    """Pin that RunStep injects ProfilesService. The disk re-check needs
    profile avgBits to compute expected output size; without the
    service injected, the re-check has no profile data to work from."""
    content = RUN_STEP.read_text(encoding="utf-8")
    assert "@Environment(ProfilesService.self)" in content, (
        "M204 regression: RunStep no longer injects ProfilesService. "
        "The final disk-space re-check needs it to call "
        "PreflightRunner.estimateOutputBytes(plan:, profiles:)."
    )


def test_start_fn_calls_estimate_output_bytes():
    """Pin that start() calls PreflightRunner.estimateOutputBytes
    before spawning the subprocess. The expected-output-size math
    must be consistent between Step-3 preflight and this re-check."""
    body = _live_code(RUN_STEP)
    # The call must appear in the file; scope further via start() slice.
    start_body = _start_fn_body()
    # Strip comments from the start body too before the check.
    start_body_live = "\n".join(
        line for line in start_body.split("\n")
        if not line.lstrip().startswith("//")
    )
    assert "PreflightRunner.estimateOutputBytes" in start_body_live, (
        "M204 regression: RunStep.start() does not call "
        "PreflightRunner.estimateOutputBytes. Pre-M204 there was no "
        "disk-space re-check between the Step-3 preflight and the "
        "subprocess spawn — a 10-minute gap during which the user's "
        "disk could fill, leading to ENOSPC mid-shard after 20+ "
        "minutes of wasted compute."
    )


def test_start_fn_checks_volume_available_capacity():
    """Pin the use of volumeAvailableCapacityForImportantUsage — the
    APFS-aware free-space key that accounts for purgeable caches.
    Using the plain volumeAvailableCapacityKey would over-count free
    space (counts purgeable which hasn't been purged yet)."""
    start_body = _start_fn_body()
    start_body_live = "\n".join(
        line for line in start_body.split("\n")
        if not line.lstrip().startswith("//")
    )
    assert "volumeAvailableCapacityForImportantUsage" in start_body_live, (
        "M204 regression: RunStep.start() disk re-check does not use "
        "volumeAvailableCapacityForImportantUsageKey. Use the important-"
        "usage variant so purgeable caches are accounted for (matches "
        "PreflightRunner.diskSpace behavior)."
    )


def test_start_fn_compares_free_vs_estimated_before_running():
    """Pin that the re-check's comparison happens BEFORE the transition
    to `.running` and BEFORE spawning PythonRunner. Running the check
    AFTER would burn CPU + potentially create partial output."""
    start_body = _start_fn_body()
    start_body_live = "\n".join(
        line for line in start_body.split("\n")
        if not line.lstrip().startswith("//")
    )
    # Find the disk comparison + the PythonRunner creation + the
    # `run = .running` assignment. Assert the comparison happens first.
    disk_compare_idx = start_body_live.find("free < estimated")
    runner_create_idx = start_body_live.find("PythonRunner(")
    running_assign_idx = start_body_live.find("coord.plan.run = .running")
    assert disk_compare_idx != -1, (
        "M204 regression: RunStep.start() re-check comparison "
        "`free < estimated` is missing. The estimate is computed but "
        "never compared against actual free disk space."
    )
    assert runner_create_idx != -1, "PythonRunner creation vanished"
    assert running_assign_idx != -1, "`run = .running` transition vanished"
    assert disk_compare_idx < running_assign_idx, (
        f"M204 regression: disk-space comparison (idx {disk_compare_idx}) "
        f"must happen BEFORE `coord.plan.run = .running` (idx "
        f"{running_assign_idx}). Running state should not be entered "
        f"if the disk re-check fails."
    )
    assert disk_compare_idx < runner_create_idx, (
        f"M204 regression: disk-space comparison (idx {disk_compare_idx}) "
        f"must happen BEFORE `PythonRunner(...)` creation (idx "
        f"{runner_create_idx}). The subprocess must not be spawned if "
        f"the re-check fails."
    )


def test_start_fn_surfaces_disk_drop_with_actionable_hint():
    """Pin that when the re-check fails, the log message tells the user
    what they need (N GB) vs what they have (M GB) and suggests an
    action. Matches the 'error remediation pattern' from
    feedback_remediation_pattern.md."""
    start_body = _start_fn_body()
    # Look for key phrases in the error message. Comments allowed here
    # because we want to find them regardless of whether they're in
    # the body of a string literal.
    assert "Need ~" in start_body or "need" in start_body.lower(), (
        "M204 regression: re-check failure log doesn't tell the user "
        "HOW MUCH space is needed. Without that number, the user can't "
        "judge whether 'free up space' is a minute's work or an hour's."
    )
    assert "free" in start_body.lower() and "gb" in start_body.lower(), (
        "M204 regression: re-check failure log doesn't report the "
        "current free-GB count. User needs both sides of the inequality "
        "to see the gap."
    )
    assert "Free up space" in start_body or "pick a different" in start_body or "retry" in start_body.lower(), (
        "M204 regression: re-check failure log doesn't suggest a next "
        "action. Per feedback_remediation_pattern.md, blocking error "
        "messages must be `<symptom>\\n→ <next-action>` — the user has "
        "to know what to do, not just what went wrong."
    )
