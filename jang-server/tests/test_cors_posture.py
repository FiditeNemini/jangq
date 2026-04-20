"""M191 (iter 128): jang-server CORS posture tests.

Pre-M191 the server shipped with wildcard CORS (allow_origins=["*"],
allow_methods=["*"], allow_headers=["*"]). That's permissive by default
and means any origin can fire a preflight and read responses. The
browser's allow_credentials=False default partially blocks cookie/auth
leaks, but response data still flows to malicious origins if any
endpoint forgets auth or leaks via error messages.

M191 tightens CORS to:
  - origin allowlist driven by JANG_CORS_ORIGINS env var (default
    localhost only); explicit opt-in "*" required for public APIs
  - methods restricted to GET/POST/DELETE/OPTIONS (the ones the server
    actually uses)
  - headers restricted to Content-Type + Authorization

These tests pin the middleware shape via source inspection. Full HTTP-
level CORS testing would require spinning up the FastAPI app — out of
scope for this iter (same pattern as M189 max-body-size tests).
"""
from __future__ import annotations

from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def test_cors_origins_env_var_defined():
    """JANG_CORS_ORIGINS must be readable by operators — no hardcoded values."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "JANG_CORS_ORIGINS" in content, (
        "M191 regression: JANG_CORS_ORIGINS env var missing — operators "
        "need to configure allowed origins without editing source."
    )


def test_cors_origins_default_is_not_wildcard():
    """Principle of least privilege: default MUST be restrictive."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find the default in the env.get call for JANG_CORS_ORIGINS.
    import re
    m = re.search(
        r'JANG_CORS_ORIGINS["\']\s*,\s*["\']([^"\']*)["\']',
        content,
    )
    assert m is not None, (
        "M191 regression: couldn't find JANG_CORS_ORIGINS default — the "
        "pattern 'os.environ.get(\"JANG_CORS_ORIGINS\", \"...default...\")' "
        "must be present so operators see there's a safe fallback."
    )
    default = m.group(1)
    assert default != "*", (
        "M191 regression: default JANG_CORS_ORIGINS is wildcard. Wildcard "
        "must be OPT-IN, not default. A production operator who doesn't "
        "set the env var should get a SAFE posture, not permissive."
    )
    assert "localhost" in default or "127.0.0.1" in default, (
        f"M191 regression: default JANG_CORS_ORIGINS={default!r} doesn't "
        f"include localhost — dev workflow broken or unsafe default picked."
    )


def test_cors_origins_list_is_parsed_from_env():
    """Env value must be comma-split, stripped, and passed to the middleware."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Expect a list-comprehension or equivalent that splits on ',' and strips.
    assert "split(\",\")" in content, (
        "M191 regression: JANG_CORS_ORIGINS must be comma-split so operators "
        "can list multiple origins."
    )
    assert "CORS_ORIGINS" in content, (
        "M191 regression: CORS_ORIGINS module-level constant missing — the "
        "parsed list needs a stable name for tests and introspection."
    )


def test_cors_methods_is_not_wildcard():
    """Restrict methods to the ones the server actually uses."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "CORS_METHODS" in content, (
        "M191 regression: CORS_METHODS module-level constant missing."
    )
    # Grab the CORS_METHODS definition line. It must NOT be ["*"].
    import re
    m = re.search(r"CORS_METHODS\s*=\s*(\[[^\]]*\])", content)
    assert m is not None, "M191 regression: couldn't parse CORS_METHODS = [...]"
    methods_literal = m.group(1)
    assert "*" not in methods_literal, (
        f"M191 regression: CORS_METHODS={methods_literal} still contains "
        f"wildcard. Restrict to the ones actually used (GET/POST/DELETE/OPTIONS)."
    )
    # Sanity: must include at least GET + POST (the routes this server has).
    for method in ("GET", "POST"):
        assert method in methods_literal, (
            f"M191 regression: CORS_METHODS missing {method!r} — routes "
            f"using this method would fail preflight."
        )


def test_cors_headers_is_not_wildcard():
    """Restrict headers to the ones the server actually reads."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "CORS_HEADERS" in content, (
        "M191 regression: CORS_HEADERS module-level constant missing."
    )
    import re
    m = re.search(r"CORS_HEADERS\s*=\s*(\[[^\]]*\])", content)
    assert m is not None, "M191 regression: couldn't parse CORS_HEADERS = [...]"
    headers_literal = m.group(1)
    assert "*" not in headers_literal, (
        f"M191 regression: CORS_HEADERS={headers_literal} still contains "
        f"wildcard. Restrict to Content-Type + Authorization (the ones "
        f"this server actually reads)."
    )
    # Sanity: Authorization must be allowed because check_auth reads it.
    assert "Authorization" in headers_literal, (
        "M191 regression: CORS_HEADERS missing 'Authorization' — browser "
        "clients would fail to send Bearer tokens cross-origin."
    )


def test_cors_middleware_uses_the_restricted_constants():
    """Pin that the middleware registration actually references the
    restricted constants, not inline literals. Prevents a future edit
    from accidentally re-introducing wildcards while leaving the
    constants (and these tests) green."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Look for the middleware registration block.
    start = content.find("app.add_middleware(")
    assert start != -1
    # Slice the next ~400 chars to capture the call + args.
    block = content[start:start + 500]
    assert "CORSMiddleware" in block, (
        "M191 regression: CORSMiddleware registration missing."
    )
    assert "allow_origins=CORS_ORIGINS" in block, (
        "M191 regression: allow_origins must reference CORS_ORIGINS, not "
        "a literal list, so the env-var allowlist is actually honored."
    )
    assert "allow_methods=CORS_METHODS" in block, (
        "M191 regression: allow_methods must reference CORS_METHODS."
    )
    assert "allow_headers=CORS_HEADERS" in block, (
        "M191 regression: allow_headers must reference CORS_HEADERS."
    )
