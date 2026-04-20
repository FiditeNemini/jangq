"""M194 (iter 131): HTTPException response-body redaction tests.

Response bodies cross the trust boundary FROM the server TO the client.
M193 (iter 130) redacted LOG sites (server-internal storage + access
logs). But HTTPException `detail=` strings are separate — they become
the response body visible to the calling client.

Pre-M194, three endpoints formatted `{e}` from HuggingFace client
exceptions directly into HTTPException:
  - POST /jobs       (L884)  raise HTTPException(404, f"Model '{req.model_id}' not found on HuggingFace: {e}")
  - POST /estimate   (L1155) raise HTTPException(404, f"Model not found: {e}")
  - GET  /recommend  (L1208) raise HTTPException(404, f"Model not found: {e}")

HF client exceptions sometimes embed the failing URL (with token in
query string) in their message. If that URL contains an HF token, a
secret-holding operator's token leaks to the ORIGINAL caller — which
may or may not be the same party that supplied the token. Same
redaction helper (redact_for_log) as M193, applied at a new boundary.

M194 also cleans up a small naming bug discovered during iter-130's
log audit: `check_rate_limit` previously shadowed the module-level
`log` logger with a local deque variable. No current bug (no log calls
inside the function), but a future maintainer would hit AttributeError
on `deque.info(...)`. Renamed to `ip_log`.

These tests pin the HTTPException raises via source inspection and
pin the renamed local via the rate-limiter's identifier usage.
"""
from __future__ import annotations

import re
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def test_model_not_found_in_jobs_post_is_redacted():
    """POST /jobs — the 'Model ... not found on HuggingFace' 404 body
    must wrap the HF exception with redact_for_log so tokens in HF
    client error messages don't flow to the API caller."""
    content = SERVER_PY.read_text(encoding="utf-8")
    idx = content.rfind("Model '")
    assert idx != -1, "M194 regression: 'Model ...' HTTPException removed"
    # Slice backwards + forwards to capture the raise line.
    snippet = content[idx - 100 : idx + 200]
    assert "redact_for_log" in snippet, (
        "M194 regression: HTTPException on POST /jobs model-not-found "
        "does NOT wrap the HF exception with redact_for_log. HF client "
        "errors can embed the failing URL (with token) in their message."
    )


def test_model_not_found_in_estimate_is_redacted():
    """POST /estimate — same class of leak as /jobs."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find all 'Model not found: {e}' raise sites.
    matches = [m.start() for m in re.finditer(r'raise HTTPException\(404, [^)]*Model not found', content)]
    assert matches, "M194 regression: '/estimate' Model-not-found HTTPException removed"
    for start in matches:
        snippet = content[start : start + 200]
        assert "redact_for_log" in snippet, (
            f"M194 regression: 'Model not found' HTTPException at offset "
            f"{start} does NOT wrap with redact_for_log. HF client "
            f"exceptions can carry tokens in URLs."
        )


def test_rate_limit_log_renamed_from_log_shadow():
    """Pin: check_rate_limit no longer shadows the module-level `log`
    logger with a local `log = deque()` assignment. Future maintainers
    adding debug logging inside the function would otherwise hit
    AttributeError on `deque.info(...)`."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find check_rate_limit function body.
    start = content.find("def check_rate_limit(")
    assert start != -1, "M194 regression: check_rate_limit removed"
    # Bound the function by the next top-level or class def.
    rest = content[start:]
    end = rest.find("\ndef ", 1)
    if end == -1:
        end = rest.find("\nclass ", 1)
    body = rest[: end if end != -1 else 1500]
    # The offending assignment (previously `log = _rate_limit_log.setdefault...`)
    # must NOT appear. The replacement (`ip_log = ...`) must.
    # Use word-boundary regex so "ip_log = _rate_limit_log.setdefault"
    # doesn't trivially match the "log = _rate_limit_log" substring.
    assert not re.search(r"\blog\s*=\s*_rate_limit_log\.setdefault", body), (
        "M194 regression: check_rate_limit re-introduced `log = "
        "_rate_limit_log.setdefault(...)` — this shadows the module-"
        "level `log` logger. Rename local back to `ip_log` or similar."
    )
    assert "ip_log" in body, (
        "M194 regression: check_rate_limit should use `ip_log` (or an "
        "equivalent non-shadowing name) for the per-IP deque."
    )


def test_redact_for_log_is_used_in_http_exception_bodies():
    """Generic pin: any `raise HTTPException(4\\d\\d, f"... {e}")`
    pattern where `{e}` is a caught exception from an external system
    (HF client, urllib, etc.) should wrap with redact_for_log.
    This test counts the HTTPException-with-interpolated-exception
    sites and asserts that the ones touching external-system errors
    are all wrapped.

    Scope: only the THREE known-external sites (HF model_info). Inline
    validators returning static messages (e.g. 'Invalid profile: ...')
    don't need redaction because the value is controlled input, not
    an exception from an external library.
    """
    content = SERVER_PY.read_text(encoding="utf-8")
    # The 3 HF-client sites. All must use redact_for_log.
    for search_phrase in (
        "not found on HuggingFace",
        "Model not found",
    ):
        for m in re.finditer(re.escape(search_phrase), content):
            # Slice ~300 chars around each occurrence.
            snippet = content[max(0, m.start() - 100) : m.start() + 200]
            # Skip snippets that are inside the M194 test rationale
            # block (we quote the phrase in the module docstring).
            if "raise HTTPException" not in snippet:
                continue
            assert "redact_for_log" in snippet, (
                f"M194 regression: HTTPException near '{search_phrase}' "
                f"(offset {m.start()}) does not wrap with redact_for_log. "
                f"HF client exceptions can carry tokens."
            )
