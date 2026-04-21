"""M225 (iter 150): TestInferenceSheet working-status label must be
honest about load-vs-generate phases.

§8 stage #5 (load) audit. Pre-M225:
  - `TestInferenceSheet.swift:120` showed a constant `"Generating..."`
    label whenever `vm.isGenerating == true`.
  - `InferenceRunner.generate()` is one-shot: spawns a fresh Python
    subprocess, loads the model from disk, generates, exits. Every
    `Send` click pays the full cold-load cost.
  - For a 30GB MoE model on M2 Max, load alone takes 15-30s.
  - User experience: click Send → see "Generating..." for 30s → see
    few generated tokens → conclude "model is slow at generation".
  - Reality: 95% of that time was LOADING, not generating.

M225 adds:
  - `lastLoadTimeS: Double?` recorded from `InferenceResult.loadTimeS`
    after every successful generate. Cites the number on the NEXT
    Send so the user knows what to expect.
  - `generateStartedAt: Date?` captured at the start of each Send.
    Drives an elapsed-seconds counter in the label.
  - `workingStatusLabel()` computed property that returns:
      first send: "Loading model + generating (Ns elapsed)… large
                   MoE models can take 30s+ on first run."
      later:      "Loading model + generating (Ns elapsed)… previous
                   run loaded in Ms. Each Send reloads the model."
  - TestInferenceSheet uses this label instead of the hardcoded
    "Generating..." string.

Does NOT solve the underlying cold-load cost (each Send reloads the
model from disk). That's a bigger architectural change (persistent
subprocess or streaming inference pipeline). M225 just makes the
current behavior HONEST so strangers don't misattribute the wait.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
SHEET = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "TestInferenceSheet.swift"
VM = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "TestInferenceViewModel.swift"


def _live_code(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_sheet_no_longer_hardcodes_generating_label():
    """Pin: TestInferenceSheet must NOT show a hardcoded "Generating..."
    string. It should use the ViewModel's workingStatusLabel() which
    distinguishes load vs. generate phases."""
    live = _live_code(SHEET)
    # The hardcoded literal was `Text("Generating...")`. Must be gone
    # from live code (comments that MENTION the old pattern are stripped
    # by _live_code).
    assert 'Text("Generating...")' not in live, (
        "M225 regression: TestInferenceSheet still hardcodes the "
        "`Text(\"Generating...\")` label. Pre-M225 this was misleading: "
        "users saw it during model LOAD (which dominates wall-clock for "
        "large MoE) and assumed the model was slow at generation. Use "
        "`vm.workingStatusLabel()` so the label distinguishes load vs "
        "generate."
    )


def test_sheet_uses_view_model_working_status_label():
    """Pin: the working-status label in TestInferenceSheet must come
    from the ViewModel's workingStatusLabel() call. Keeps the honest
    label logic in one place for testability + reuse."""
    live = _live_code(SHEET)
    assert "vm.workingStatusLabel()" in live, (
        "M225 regression: TestInferenceSheet no longer calls "
        "`vm.workingStatusLabel()`. The honest load-vs-generate label "
        "lives in the ViewModel; the sheet should render from it, not "
        "invent its own string."
    )


def test_view_model_has_working_status_label_method():
    """Pin the method + return type."""
    content = VM.read_text(encoding="utf-8")
    assert re.search(
        r"func\s+workingStatusLabel\(\s*\)\s*->\s*String",
        content,
    ), (
        "M225 regression: TestInferenceViewModel.workingStatusLabel() -> String "
        "method missing or signature changed. Expected a no-arg method "
        "returning the honest load-vs-generate label string."
    )


def test_view_model_captures_load_time_and_start_timestamp():
    """Pin the two pieces of state the label needs: lastLoadTimeS
    (from InferenceResult) and generateStartedAt (local timestamp).
    Without either, the label can't cite real numbers."""
    live = _live_code(VM)
    assert "lastLoadTimeS" in live, (
        "M225 regression: TestInferenceViewModel no longer captures "
        "lastLoadTimeS. Without it, subsequent-send labels can't cite "
        "the previous run's load time and strangers can't calibrate."
    )
    assert "generateStartedAt" in live, (
        "M225 regression: TestInferenceViewModel no longer captures "
        "generateStartedAt. Without it, the label can't show `(Ns "
        "elapsed)` and strangers can't tell if the app is stuck vs "
        "still making progress."
    )


def test_working_status_label_cites_elapsed_seconds():
    """Pin: the label must reference elapsed seconds when a
    generate is in progress. A silent spinner with no elapsed-time
    hint leaves strangers unsure whether the app is stuck."""
    live = _live_code(VM)
    # Either "(Ns elapsed)" or "elapsed)" patterns must appear in
    # the label construction.
    assert "elapsed" in live, (
        "M225 regression: workingStatusLabel() doesn't reference "
        "elapsed time. Without it, users watching a 30GB MoE load "
        "can't distinguish 'still loading' from 'app is frozen'."
    )


def test_working_status_label_references_first_run_vs_subsequent():
    """Pin: the label must differentiate first-send ("large MoE
    models can take 30s+") from subsequent-send ("previous run loaded
    in Ns"). Otherwise a user who's already done one Send sees a
    generic hint instead of a calibrated one."""
    live = _live_code(VM)
    # First-run hint: phrase about 30s+ loading.
    assert "30s" in live or "large MoE" in live, (
        "M225 regression: first-run label doesn't calibrate expectations "
        "for strangers running their first Send on a large MoE model. "
        "Add a 'can take 30s+' or 'large MoE' hint for lastLoadTimeS == nil."
    )
    # Subsequent-run hint: phrase about previous load time.
    assert "previous run" in live or "lastLoadTimeS" in live, (
        "M225 regression: subsequent-run label doesn't cite the "
        "previous load time. Strangers on their 2nd+ Send need a "
        "concrete number to calibrate."
    )
