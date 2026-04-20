"""M189 (iter 126): request-body-size cap tests.

Pre-M189, jang-server had no upper bound on POST body size. An
attacker could send a 10 GB JSON body, exhausting RAM before
Pydantic validation rejects the wrong shape. JobRequest /
EstimateRequest payloads are at most a few KB in practice, so a
1 MB cap leaves 1000× headroom and stops memory-bomb requests cold.

These tests pin the middleware shape via source inspection. Full
HTTP-level integration testing would require spinning up the FastAPI
app — out of scope for this iter.
"""
from __future__ import annotations

from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def test_max_body_bytes_env_var_defined():
    """Env var name + sensible default must be defined."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "JANG_MAX_BODY_BYTES" in content, (
        "M189 regression: JANG_MAX_BODY_BYTES env var missing — operators "
        "need a tunable cap on max request body size."
    )
    assert "MAX_BODY_BYTES = int(os.environ.get" in content, (
        "M189 regression: MAX_BODY_BYTES module-level constant missing"
    )


def test_limit_body_size_middleware_registered():
    """The middleware must be registered via @app.middleware('http')."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "@app.middleware(\"http\")" in content, (
        "M189 regression: @app.middleware('http') decorator missing — "
        "needed to intercept ALL requests for body-size checking."
    )
    assert "async def limit_body_size" in content, (
        "M189 regression: limit_body_size middleware function missing"
    )


def test_middleware_rejects_with_413():
    """413 Payload Too Large is the correct HTTP status per RFC 9110."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find the middleware function body.
    start = content.find("async def limit_body_size")
    assert start != -1
    body = content[start:start + 2500]
    assert "status_code=413" in body, (
        "M189 regression: middleware must use 413 (Payload Too Large) "
        "per RFC 9110, not 400 or 500."
    )


def test_middleware_inspects_content_length_header():
    """Middleware must read Content-Length from request headers."""
    content = SERVER_PY.read_text(encoding="utf-8")
    start = content.find("async def limit_body_size")
    body = content[start:start + 2500]
    assert "content-length" in body.lower(), (
        "M189 regression: middleware must inspect Content-Length header"
    )


def test_default_cap_is_reasonable_for_jang_payloads():
    """The default 1 MB cap leaves 1000× headroom for typical
    JobRequest/EstimateRequest payloads (a few KB). If the default
    drops below 64 KB it'd reject legit requests; if it exceeds 100 MB
    it'd let through memory bombs."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Look for the default in the env.get call.
    import re
    m = re.search(r'JANG_MAX_BODY_BYTES.*?str\((\d+)\s*\*\s*(\d+)\s*\*\s*(\d+)\)', content)
    if m:
        default_bytes = int(m.group(1)) * int(m.group(2)) * int(m.group(3))
        assert 64 * 1024 <= default_bytes <= 100 * 1024 * 1024, (
            f"M189 regression: default body cap {default_bytes} bytes is "
            f"outside the reasonable range [64KB, 100MB]. JobRequest "
            f"payloads are a few KB; 100MB+ would let memory bombs through."
        )
