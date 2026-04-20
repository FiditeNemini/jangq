"""M198 (iter 135): retroactive invariants for JANGQuantizer.swiftpm fixes.

Audit across iters 111-197 (iter-134's meta-lesson "every security fix
needs a paired invariant") surfaced two gaps: M185 (iter 120) and M186
(iter 121) both fixed silent-failure / URL-injection bugs in the
JANGQuantizer.swiftpm frontend but neither had a paired regression
test. The package has no Tests/ directory today — rather than stand up
a whole SwiftPM test target for two invariants, we pin the fixed state
via source inspection from the Python ralph_runner suite (matches the
M197 parity invariant pattern — cross-language source-level check).

Invariants covered:
  1. M185 URL construction: `APIClient.listJobs` MUST use URLComponents +
     URLQueryItem, NOT string interpolation. String interpolation of
     user-supplied values into URL query strings is a classic injection
     vector (a user named `alice&phase=COMPLETED` would override the
     phase filter in the original code).
  2. M186 button-handler error surfacing: no `try?` in Task closures
     that are triggered by user Button actions. Silent-swallow leaves
     the user without feedback on network/auth/server failures.
  3. M185 SettingsView "Check Connection" button: explicit `do/catch`
     with `lastError` state, not silent `} catch { health = nil }`.

If this test fires:
  - Re-read the failure message; it names the file + line + bug class.
  - Revert to do/catch + user-visible state per M185/M186 (iter
    120/121) fix pattern.
  - The `try?` pattern is ONLY acceptable in background .task closures
    (mount-time health-check) where silent-null is the UX goal.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent  # /Users/eric/jang/
SWIFTPM_SRC = REPO_ROOT / "jang-server" / "frontend" / "JANGQuantizer.swiftpm" / "Sources"

API_CLIENT = SWIFTPM_SRC / "APIClient.swift"
QUEUE_VIEW = SWIFTPM_SRC / "QueueView.swift"
SETTINGS_VIEW = SWIFTPM_SRC / "SettingsView.swift"


def _fn_body(path: Path, signature: str) -> str:
    """Return the source of a function identified by its signature
    string. Slices from `signature` to the next top-level func /
    struct close-brace. Coarse but sufficient for this invariant."""
    content = path.read_text(encoding="utf-8")
    start = content.find(signature)
    assert start != -1, f"Signature {signature!r} not found in {path.name}"
    # Next top-level `func ` at the same indent level OR the closing
    # `}` of the struct. Simplest bound: slice ~50 lines forward.
    rest = content[start:]
    lines = rest.split("\n")
    end = min(80, len(lines))
    return "\n".join(lines[:end])


# ── M185: URL construction must use URLComponents ─────────────────────

def test_list_jobs_uses_url_components_not_interpolation():
    """M185 regression: APIClient.listJobs must build the query via
    URLComponents/URLQueryItem. String interpolation of user/phase
    into a URL is an injection vector."""
    body = _fn_body(API_CLIENT, "func listJobs(")
    assert "URLComponents" in body, (
        "M185 regression: APIClient.listJobs no longer uses URLComponents. "
        "A username containing `&`, `=`, `?`, `#`, `+`, or space would "
        "break the URL or inject extra query params. Revert to the "
        "URLComponents + URLQueryItem construction pattern."
    )
    assert "URLQueryItem" in body, (
        "M185 regression: APIClient.listJobs no longer uses URLQueryItem. "
        "Manual string concatenation misses URL encoding for special chars."
    )
    # Negative pin: no string interpolation of user/phase into a URL string.
    # The bug pattern was `var url = "/jobs?user=\(u)&phase=\(p)"`.
    # Disallow that shape.
    bad = re.search(r'"/jobs\?[^"]*\\\(', body)
    assert bad is None, (
        f"M185 regression: APIClient.listJobs re-introduced string "
        f"interpolation into the URL path. Matched: {bad.group(0)!r}. "
        f"Use URLComponents + URLQueryItem instead."
    )


# ── M186: button handlers must surface errors ─────────────────────────

def test_queue_view_cancel_retry_use_do_catch_not_try_optional():
    """M186 regression: JobCard's Cancel and Retry Button closures
    must use do/catch + state-surfaced error, not `try? await` which
    silently swallows failures."""
    content = QUEUE_VIEW.read_text(encoding="utf-8")
    # Only the M186 docstring comment is allowed to mention `try? await`
    # (it explains the pre-M186 bug). Live code should not contain it.
    # Strip comment-delimited lines before checking.
    code_lines = []
    for line in content.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("///") or stripped.startswith("//"):
            continue
        code_lines.append(line)
    code_only = "\n".join(code_lines)
    assert "try? await" not in code_only, (
        "M186 regression: QueueView contains `try? await` in live code. "
        "That pattern silently swallows network errors — user clicks "
        "Cancel/Retry, nothing visible happens. Use `do { try await X } "
        "catch { actionError = ... }` so the error surfaces."
    )


def test_queue_view_has_actionError_state_for_error_surface():
    """Pin the actionError @State variable that M186 introduced. Without
    it, even do/catch would have nowhere to surface the error."""
    content = QUEUE_VIEW.read_text(encoding="utf-8")
    assert re.search(r"@State\s+private\s+var\s+actionError\s*:", content), (
        "M186 regression: JobCard.actionError @State var missing. "
        "Cancel/Retry errors need a user-visible surface; actionError "
        "is the conventional name for that surface in this view."
    )
    # Pin the inline error-text view below the buttons.
    assert "actionError" in content and "Text(err)" in content, (
        "M186 regression: actionError is defined but not rendered. "
        "The Text(err) view below the action buttons is what the user "
        "actually sees — without it, errors are captured but invisible."
    )


# ── M185: SettingsView "Check Connection" button ──────────────────────

def test_settings_check_connection_uses_do_catch_with_last_error():
    """M185's SettingsView fix: the 'Check Connection' button must
    use do/catch and populate `lastError` on failure. Pre-M185 the
    catch was `{ health = nil }` — silent drop."""
    content = SETTINGS_VIEW.read_text(encoding="utf-8")
    # Find the Button("Check Connection") block.
    idx = content.find('Button("Check Connection")')
    assert idx != -1, "M185 regression: 'Check Connection' button removed"
    snippet = content[idx:idx + 600]
    assert "do {" in snippet and "catch {" in snippet, (
        "M185 regression: 'Check Connection' button no longer uses "
        "do/catch. User clicks + nothing happens on failure = iter-120 "
        "bug returning."
    )
    assert "lastError" in snippet, (
        "M185 regression: 'Check Connection' button doesn't populate "
        "lastError on failure. Without that, do/catch does nothing "
        "user-visible."
    )


def test_settings_has_last_error_rendering():
    """Pin that lastError is not only stored but ALSO rendered."""
    content = SETTINGS_VIEW.read_text(encoding="utf-8")
    assert re.search(r"@State\s+private\s+var\s+lastError\s*:", content), (
        "M185 regression: lastError @State var missing from SettingsView."
    )
    # lastError must appear in a Text(...) rendering context.
    assert re.search(r"if\s+let\s+err\s*=\s*lastError", content), (
        "M185 regression: lastError is defined but not rendered in a "
        "Text view. Without rendering, do/catch captures the error but "
        "the user sees nothing."
    )
