"""M206 (iter 140): SourceStep cold-start guidance invariant.

Pre-M206, a first-time JANGStudio user on the empty-state SourceStep
saw only:
  - "No folder selected" (secondary foreground, muted)
  - "Choose Folder…" button

The explanation of WHAT to pick lived exclusively in the Section
header's InfoHint (hover-only tooltip). Strangers who don't discover
the (i) hover icon get zero instruction — a classic cold-start
friction point.

M206 adds always-visible guidance below the "No folder selected" row
when `coord.plan.sourceURL == nil`:
  1. What the folder should contain (config.json + .safetensors).
  2. A concrete example path.
  3. A link to huggingface.co for users who don't have a model locally.

These tests pin the guidance via source inspection. If a future iter
removes the guidance (e.g., "Choose Folder button is enough"), the
test fires and the rewrite forces the removal to be deliberate.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
SOURCE_STEP = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "Steps" / "SourceStep.swift"


def _live_code(path: Path) -> str:
    """Strip `//` + `///` single-line comments (iter-135 M198 pattern)."""
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_empty_state_shows_always_visible_guidance():
    """Pin that the empty-state (sourceURL == nil) shows content beyond
    the 'No folder selected' label. Without this, a stranger sees zero
    instruction on fresh launch."""
    live = _live_code(SOURCE_STEP)
    # The guidance block is gated by `if coord.plan.sourceURL == nil`.
    # Pre-M206 the nil branch was literally just the "No folder selected"
    # Text. Post-M206 there's an `if coord.plan.sourceURL == nil` block
    # with a VStack containing the guidance.
    assert "coord.plan.sourceURL == nil" in live, (
        "M206 regression: SourceStep no longer gates additional guidance "
        "on the `sourceURL == nil` empty state. Cold-start users need "
        "instruction beyond `No folder selected` + `Choose Folder…`."
    )


def test_guidance_names_required_files():
    """Pin the guidance mentions `config.json` AND `.safetensors` —
    the two concrete artifacts the user needs in their folder. A
    stranger who doesn't know HF's layout conventions needs to see
    these names to know what to pick."""
    live = _live_code(SOURCE_STEP)
    assert "config.json" in live, (
        "M206 regression: cold-start guidance must mention `config.json`."
    )
    assert ".safetensors" in live, (
        "M206 regression: cold-start guidance must mention `.safetensors`."
    )


def test_guidance_includes_concrete_example_path():
    """Pin that the guidance includes a concrete example path like
    `~/Downloads/Qwen3-...`. Abstract phrases like 'model directory'
    are useless without an example a stranger can pattern-match
    against a file on their disk."""
    live = _live_code(SOURCE_STEP)
    # Match `Example:` followed by a path-like string.
    assert re.search(r'Example:.*~/Downloads/', live), (
        "M206 regression: cold-start guidance must include a concrete "
        "example path (e.g. `~/Downloads/Qwen3-0.6B-Base/`). Abstract "
        "descriptions don't help a stranger know what to click."
    )


def test_guidance_links_to_huggingface():
    """Pin that the guidance includes a Link to huggingface.co for
    users who don't have a model locally. Without this, a first-time
    user whose machine has no HF models is stuck at step 0."""
    live = _live_code(SOURCE_STEP)
    assert "huggingface.co" in live, (
        "M206 regression: cold-start guidance must link to huggingface.co "
        "so users without a local model know where to get one."
    )
    assert "Link(" in live, (
        "M206 regression: the huggingface.co reference must be a SwiftUI "
        "Link (not just a Text) so it's clickable."
    )
