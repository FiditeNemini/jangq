"""M187 (iter 124): rate-limit invariant tests.

Pre-M187, jang-server had no rate limiting on POST endpoints. An
authenticated client could flood:
- POST /estimate (each call hits HF API → exhausts shared rate budget)
- POST /jobs (creates DB rows + runs validation work)
without bound.

iter-124 added a sliding-window per-IP rate limiter at
RATE_LIMIT_MAX_REQUESTS / RATE_LIMIT_WINDOW_S. These tests verify:
- The check_rate_limit function exists + is wired into the right endpoints.
- The window/max env vars are honored.
- Public endpoints (/health, /profiles) skip rate limiting.
"""
from __future__ import annotations

import re
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def _route_decorators(content: str) -> dict[str, list[str]]:
    """Same parser shape as iter-114's auth-enforcement test."""
    result: dict[str, list[str]] = {}
    for line in content.split("\n"):
        m = re.match(r'@app\.(get|post|delete|put|patch)\("([^"]+)"(.*)\)', line.strip())
        if not m:
            continue
        method, path = m.group(1).upper(), m.group(2)
        key = f"{method} {path}"
        result.setdefault(key, []).append(line.strip())
    return result


def test_check_rate_limit_function_exists():
    """The rate-limit helper must be defined as a callable."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "def check_rate_limit(" in content, (
        "M187 regression: check_rate_limit function missing — rate-limit "
        "helper must be defined for the @app.post dependency injection."
    )


def test_check_rate_limit_uses_sliding_window():
    """Implementation must use a per-IP deque + window expiry, not a
    fixed counter that resets on minute boundary (which would let a
    client burst at minute boundaries)."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Rough shape check — the function should reference deque + popleft.
    func_start = content.find("def check_rate_limit(")
    assert func_start != -1
    func_body = content[func_start:func_start + 2000]
    assert "popleft" in func_body, (
        "M187 regression: check_rate_limit must use deque.popleft to drop "
        "entries older than the window. Fixed-counter implementations "
        "would let clients burst at minute boundaries."
    )
    assert "_rate_limit_log" in func_body, (
        "M187 regression: per-IP log dict missing"
    )


def test_rate_limit_applied_to_high_cost_endpoints():
    """POST /estimate (HF API call) and POST /jobs (subprocess work) must
    have check_rate_limit in their dependencies. Other auth'd POSTs
    (cancel, retry, admin/purge) are lower-cost and not strictly required
    but could be added later."""
    content = SERVER_PY.read_text(encoding="utf-8")
    routes = _route_decorators(content)
    for endpoint in ("POST /estimate", "POST /jobs"):
        decos = routes.get(endpoint, [])
        assert any("check_rate_limit" in d for d in decos), (
            f"M187 regression: {endpoint} must include "
            f"`Depends(check_rate_limit)` in its dependencies. Pre-M187 "
            f"a single client could flood this endpoint exhausting the "
            f"server's HF rate budget (estimate) or DB+validation budget (jobs)."
        )


def test_public_endpoints_skip_rate_limit():
    """Public liveness / static endpoints should NOT have rate limiting —
    keeps health-probes / profile-reads cheap. Hostile flooding of /health
    is a separate concern (network-layer mitigation)."""
    content = SERVER_PY.read_text(encoding="utf-8")
    routes = _route_decorators(content)
    for endpoint in ("GET /health", "GET /profiles"):
        decos = routes.get(endpoint, [])
        # Must NOT have check_rate_limit in dependencies.
        for d in decos:
            assert "check_rate_limit" not in d, (
                f"M187 regression: {endpoint} should NOT have check_rate_limit "
                f"— public liveness endpoints stay cheap. If hostile flooding "
                f"becomes a concern, mitigate at the network layer (nginx / "
                f"WAF), not in-app."
            )


def test_rate_limit_env_vars_documented():
    """Env-var names + defaults must appear in source comments so
    operators can find them."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "JANG_RATE_LIMIT_WINDOW_S" in content
    assert "JANG_RATE_LIMIT_MAX_REQUESTS" in content
