"""M210 (iter 142): ProfileStep auto-path must honor Settings.

Pre-M210 both `AppSettings.outputNamingTemplate` and
`AppSettings.defaultOutputParentPath` were Settings-UI lies:

  - `outputNamingTemplate` — persisted UI + `renderOutputName()` method,
    but ProfileStep.swift hardcoded `"\\(src.lastPathComponent)-\\(profile)"`
    interpolation at the auto-path site. Changing the template to
    e.g. `{basename}_{profile}_q` had zero effect on what dir got made.
  - `defaultOutputParentPath` — persisted UI with Choose/Clear buttons,
    but ProfileStep.swift always used `src.deletingLastPathComponent()`
    (the source's parent). User's configured default parent was never
    consulted.

M210 wires both through `autoOutputURL(for:profile:)`:
  1. Parent dir: use `settings.defaultOutputParentPath` when non-empty +
     valid dir, else fall back to source's parent.
  2. Folder name: use `settings.renderOutputName(basename:profile:family:)`
     to apply the template.

If either setting fails to take effect on a flipped value, the setting
is back to being a lie — this invariant fires and the remediation is
to re-wire, not to silence the test.

This source-inspection invariant complements AppSettingsTests's unit
tests of `renderOutputName` itself. Those tests verify the FUNCTION;
this test verifies the CONSUMER actually calls it.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
PROFILE_STEP = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Wizard" / "Steps" / "ProfileStep.swift"


def _live_code(path: Path) -> str:
    """Strip `//` + `///` single-line comments (iter-135 M198 pattern)."""
    lines = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        lines.append(line)
    return "\n".join(lines)


def test_profile_step_injects_app_settings():
    """Pin: ProfileStep must inject AppSettings via @Environment to
    access outputNamingTemplate + defaultOutputParentPath. Without
    the injection, the settings are unreachable from this view."""
    content = PROFILE_STEP.read_text(encoding="utf-8")
    assert "@Environment(AppSettings.self)" in content, (
        "M210 regression: ProfileStep no longer injects AppSettings. "
        "The auto-path code depends on it to honor Settings → General → "
        "Default output parent path + Output naming template."
    )


def test_profile_step_calls_render_output_name():
    """Pin: ProfileStep must use settings.renderOutputName(...) when
    computing the auto-generated folder name. Hardcoded interpolation
    like `"\\(src.lastPathComponent)-\\(profile)"` is the pre-M210
    lie — it ignores the user's template."""
    live = _live_code(PROFILE_STEP)
    assert "settings.renderOutputName" in live, (
        "M210 regression: ProfileStep no longer calls "
        "settings.renderOutputName. Pre-M210 the auto-path hardcoded "
        "the format; the template setting was a lie. Must route through "
        "renderOutputName so the user's template takes effect."
    )


def test_profile_step_honors_default_output_parent_path():
    """Pin: ProfileStep must reference settings.defaultOutputParentPath
    when resolving the parent directory. A consumer that always uses
    `src.deletingLastPathComponent()` ignores the user's configured
    default parent."""
    live = _live_code(PROFILE_STEP)
    assert "settings.defaultOutputParentPath" in live, (
        "M210 regression: ProfileStep no longer reads "
        "settings.defaultOutputParentPath. Users who configure a "
        "preferred output volume (e.g. /Volumes/ModelDrive/quantized/) "
        "would silently have that setting ignored."
    )


def test_profile_step_no_hardcoded_basename_profile_interpolation():
    """Pin that the pre-M210 bug pattern — literal string
    interpolation of `"\\(src.lastPathComponent)-\\(profile)"` — is
    NOT back in live code. Comment-strip first (the M146 and M210
    rationale comments mention the pre-fix shape as context).

    Simple substring check against the distinctive shape `lastPathComponent)-\\(`
    (close paren + dash + open parenthesis of a second Swift
    interpolation). That specific sequence is the pre-M210 bug and
    shouldn't appear in live code."""
    live = _live_code(PROFILE_STEP)
    # Direct literal-interpolation pattern `\(src.lastPathComponent)-\(`
    # is the old hardcoded site. Comment-stripped live code should be clean.
    assert "lastPathComponent)-\\(" not in live, (
        "M210 regression: ProfileStep re-introduced hardcoded "
        "`\\(src.lastPathComponent)-\\(profile)` interpolation. Route "
        "auto-path through autoOutputURL(for:profile:) which uses "
        "settings.renderOutputName — honoring the user's template."
    )


def test_profile_step_has_auto_output_url_helper():
    """Pin the extracted helper. Centralizing the auto-path logic in
    one function makes it reusable across the two sites (onAppear +
    profile-change regeneration) AND makes it the obvious place to
    extend for future settings like {output_pattern_v2}."""
    content = PROFILE_STEP.read_text(encoding="utf-8")
    # Swift signatures have `<label> <name>: Type`, so accept an
    # optional internal-name word between the label and the colon.
    assert re.search(
        r"func\s+autoOutputURL\(for(?:\s+\w+)?\s*:\s*URL\s*,\s*profile(?:\s+\w+)?\s*:\s*String\s*\)",
        content,
    ), (
        "M210 regression: ProfileStep.autoOutputURL(for:profile:) "
        "helper removed. Without the helper, the two auto-path sites "
        "(onAppear + profile-change) can drift — one honoring the "
        "settings, the other hardcoding. Consolidating is both DRY "
        "and a parity guarantee."
    )
