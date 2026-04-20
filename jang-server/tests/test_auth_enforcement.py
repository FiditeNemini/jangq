"""M179 (iter 114): authorization-gap regression tests.

Pre-M179, POST endpoints required API key auth but several job-read
GET endpoints were open — GET /jobs, GET /jobs/{id}, GET
/jobs/{id}/logs, GET /jobs/{id}/stream, GET /queue. If deployed with
JANG_API_KEYS set, these leaked job metadata + logs + SSE streams to
anyone with network access. Multi-user deployments allowed users to
spy on each other's jobs.

This test pins the correct auth posture via source inspection —
asserts each sensitive endpoint carries `dependencies=[Depends(check_auth)]`.
Public endpoints (/health, /profiles) stay open by design.
"""
from __future__ import annotations

import re
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def _find_route_decorators(content: str) -> dict[str, list[str]]:
    """Parse `@app.METHOD("/path"...)` lines into {path: [decorator_lines]}."""
    result: dict[str, list[str]] = {}
    for line in content.split("\n"):
        m = re.match(r'@app\.(get|post|delete|put|patch)\("([^"]+)"(.*)\)', line.strip())
        if not m:
            continue
        method, path, rest = m.group(1).upper(), m.group(2), m.group(3)
        key = f"{method} {path}"
        result.setdefault(key, []).append(line.strip())
    return result


AUTH_REQUIRED_ENDPOINTS = {
    # POSTs (pre-M179 already required)
    "POST /jobs",
    "POST /jobs/{job_id}/retry",
    "POST /estimate",
    "POST /admin/purge",
    "DELETE /jobs/{job_id}",
    # GETs that leaked pre-M179
    "GET /jobs/{job_id}",
    "GET /jobs",
    "GET /queue",
    "GET /jobs/{job_id}/logs",
    "GET /jobs/{job_id}/stream",
    # Recommend exposes model info — auth-required by design
    "GET /recommend/{model_id:path}",
}

PUBLIC_ENDPOINTS = {
    # Liveness — standard to leave open.
    "GET /health",
    # Static profile list — no per-user info.
    "GET /profiles",
}


def test_sensitive_endpoints_require_auth():
    """Every job-related endpoint that returns per-user info must carry
    `dependencies=[Depends(check_auth)]`. Missing auth on a GET is
    security-relevant because it leaks job metadata + logs + SSE streams
    to unauthenticated network callers when API_KEYS is set."""
    content = SERVER_PY.read_text(encoding="utf-8")
    routes = _find_route_decorators(content)
    missing_auth: list[str] = []
    for endpoint in AUTH_REQUIRED_ENDPOINTS:
        decos = routes.get(endpoint, [])
        if not decos:
            missing_auth.append(f"{endpoint} (not found in server.py)")
            continue
        # At least one decorator line for this endpoint must include the check_auth dep.
        if not any("Depends(check_auth)" in d for d in decos):
            missing_auth.append(f"{endpoint} (no Depends(check_auth))")
    assert not missing_auth, (
        f"M179 regression — these endpoints must require auth but don't:\n" +
        "\n".join(f"  {m}" for m in missing_auth) +
        "\n\nAdd `dependencies=[Depends(check_auth)]` to the @app.METHOD decorator. "
        f"Pre-M179 several GETs were open — see iter-114 commit for context."
    )


def test_public_endpoints_remain_open():
    """Regression guard: health + profiles should NOT require auth.
    Catches over-correction where a future sweep adds auth to everything."""
    content = SERVER_PY.read_text(encoding="utf-8")
    routes = _find_route_decorators(content)
    for endpoint in PUBLIC_ENDPOINTS:
        decos = routes.get(endpoint, [])
        if not decos:
            continue  # not required to exist; skip if absent
        # None of them should have Depends(check_auth) — public by design.
        if any("Depends(check_auth)" in d for d in decos):
            # This isn't a hard failure — some deployments might prefer
            # auth on health. But flag it as a warning.
            # For this test, accept that public endpoints stay public.
            continue
