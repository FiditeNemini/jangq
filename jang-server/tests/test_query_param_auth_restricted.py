"""M192 (iter 129): pin that query-param auth is restricted to GET.

Pre-M192 `check_auth` accepted `?api_key=<token>` on ANY HTTP method.
Tokens passed in non-GET URLs leak to:
  1. Server access logs (uvicorn + nginx log the full URL).
  2. Browser history (curl-from-shared-workstation risk).
  3. Proxy/CDN logs (Cloudflare, nginx — log query strings by default).
  4. Terminal history (`curl` with the full URL saved to .bash_history).

Post-M192:
  - `Authorization: Bearer <key>` is accepted on all methods (preferred).
  - `?api_key=<key>` is accepted ONLY on GET (EventSource/SSE compat).
  - POST/DELETE with query-param auth and no header → 401.

These tests pin the check_auth implementation via source inspection.
Full HTTP-level tests would require spinning up the FastAPI app — out
of scope for this iter (matches M179/M189/M191 pattern).
"""
from __future__ import annotations

from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def _auth_fn_body() -> str:
    """Return the source of the check_auth function."""
    content = SERVER_PY.read_text(encoding="utf-8")
    start = content.find("async def check_auth")
    assert start != -1, "check_auth not found in server.py"
    # Find the next top-level `def` or `async def` AT column 0 (start of line)
    # to bound the function body.
    rest = content[start:]
    lines = rest.split("\n")
    end_idx = len(lines)
    for i, line in enumerate(lines[1:], start=1):
        if line.startswith("async def ") or line.startswith("def "):
            end_idx = i
            break
    return "\n".join(lines[:end_idx])


def test_check_auth_reads_query_param_only_on_get():
    """The query_params lookup must be gated behind a GET-method check."""
    body = _auth_fn_body()
    assert "request.query_params.get" in body, (
        "M192 regression: check_auth no longer reads query_params at all — "
        "but browser EventSource/SSE clients need it for GET /stream. "
        "Either restore the GET-gated branch or document that SSE moved "
        "to a different auth scheme (signed URL, cookie)."
    )
    # The branch that reads query_params must follow a method-equality check.
    # Look for `method == "GET"` or `.method == "GET"` near the query_params
    # line to pin the gate.
    assert 'request.method == "GET"' in body or 'method == "GET"' in body, (
        "M192 regression: check_auth reads `?api_key=` without checking "
        "the HTTP method. Query-param auth MUST be restricted to GET so "
        "POST/DELETE URLs don't carry tokens (tokens in POST URLs leak "
        "to logs/history/Referer/proxies)."
    )


def test_check_auth_non_get_rejects_missing_header():
    """Non-GET requests without a Bearer token must end up with token=''
    (not fall through to query_params). Source-inspect the else/elif
    chain to make sure there's no third fallback."""
    body = _auth_fn_body()
    # Count the number of `token = ` assignments. Should be exactly 3:
    #   (1) token = auth[7:]                    (Bearer branch)
    #   (2) token = request.query_params.get... (GET elif branch)
    #   (3) token = ""                           (else branch for non-GET)
    # Or 2 if the third is folded into an `if/else expression`, but then
    # the else branch must still set token to "" (fail-closed).
    token_assigns = body.count("token = ")
    assert token_assigns >= 2, (
        f"M192 regression: expected >=2 `token = ...` assignments in "
        f"check_auth (Bearer branch + fail-closed branch); found "
        f"{token_assigns}. Auth logic may have collapsed in a way that "
        f"re-admits query-param auth on non-GET."
    )
    # Fail-closed: the else branch must NOT read query_params unconditionally.
    # Check that between an `else:` (or the final branch) and the next
    # `if token not in API_KEYS`, there's a bare `token = ""` or the
    # post-elif fall-through sets token to a non-API-KEY value.
    assert '""' in body or "''" in body or "token = \"\"" in body, (
        "M192 regression: no fail-closed empty-string assignment for "
        "non-GET non-Bearer case. Missing Bearer on POST/DELETE must "
        "yield an empty token that fails the `in API_KEYS` check, not "
        "silently fall through to query-param auth."
    )


def test_check_auth_error_message_points_at_the_right_scheme():
    """The 401 message should name Authorization: Bearer as preferred
    so operators of misbehaving clients know how to fix them."""
    body = _auth_fn_body()
    assert "Bearer" in body, (
        "M192 regression: 401 response must name `Authorization: Bearer` "
        "so misbehaving clients know the preferred path."
    )


def test_check_auth_docstring_explains_risk():
    """Keep the why in the code. Future readers must see the risk
    without having to dig into git history."""
    body = _auth_fn_body()
    body_lower = body.lower()
    # Risks mentioned somewhere in the docstring: logs, history, referer,
    # or any subset. Accept 'log' + one other vector.
    assert "log" in body_lower, (
        "M192 regression: docstring doesn't mention log-leakage risk. "
        "Future maintainers tempted to re-add query-param auth on POST "
        "need to see WHY it was removed."
    )
    assert "eventsource" in body_lower or "sse" in body_lower, (
        "M192 regression: docstring doesn't mention EventSource/SSE — "
        "the one legitimate use case preserved on GET. Without this, "
        "a future maintainer might remove the GET branch entirely and "
        "break browser SSE clients."
    )


def test_docs_api_md_documents_get_only_restriction():
    """API.md must clearly state query-param auth is GET-only + explain why."""
    api_md = SERVER_PY.parent / "docs" / "API.md"
    content = api_md.read_text(encoding="utf-8")
    assert "GET requests only" in content or "GET only" in content or "GET-only" in content, (
        "M192 regression: API.md must clearly state query-param auth is "
        "GET-only. Without explicit docs, client authors will POST tokens "
        "in URLs and leak them."
    )


def test_docs_api_md_has_no_token_fragment():
    """API.md previously contained `hf_MGS...` (a partial real token
    prefix). Even a 3-char fragment confirms a real token was once
    there and narrows an attacker's search. Replace with a generic
    placeholder like '(empty)' or '<your_hf_token>'."""
    api_md = SERVER_PY.parent / "docs" / "API.md"
    content = api_md.read_text(encoding="utf-8")
    import re
    # Match `hf_` followed by 2+ alphanumerics — any such fragment is
    # suspicious in documentation. Generic placeholders like 'hf_...'
    # (literal "..." only) are fine; shape `hf_[real chars]` is not.
    fragments = re.findall(r"hf_[A-Za-z0-9]{2,}", content)
    assert not fragments, (
        f"M192 regression: API.md contains token-shaped fragments "
        f"{fragments}. Even short prefixes confirm the real token format "
        f"and narrow an attacker's search space. Replace with '(empty)' "
        f"or '<your_hf_token>' placeholders."
    )
