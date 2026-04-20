"""M188 (iter 125): SSE concurrent-connection cap tests.

Iter-124 M187 added per-IP rate limiting on the OPEN call rate, but
SSE streams are LONG-LIVED — a client can open at the rate limit and
accumulate thousands of open streams over time. Each consumes:
- A file descriptor (HTTP socket).
- An asyncio.Queue (modest memory).
- An entry in _sse_subscribers[job_id].

Process FD limits (typically 1024-4096 default on macOS/Linux) become
the real cap. Hitting them bricks the whole server.

iter-125 added per-IP + global concurrent-stream caps.
"""
from __future__ import annotations

import re
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def test_sse_connection_caps_defined():
    """Per-IP + global env-var caps must be defined with sensible defaults."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "JANG_SSE_MAX_PER_IP" in content, (
        "M188 regression: per-IP SSE cap env var missing — needed to "
        "bound a single client's concurrent stream count."
    )
    assert "JANG_SSE_MAX_GLOBAL" in content, (
        "M188 regression: global SSE cap env var missing — needed to "
        "bound total concurrent streams below process FD limit."
    )


def test_stream_job_checks_ip_count_before_accept():
    """The /jobs/{id}/stream endpoint must check both IP + global open-
    stream counts BEFORE accepting the connection. Increment under lock
    paired with decrement in the event_generator's finally."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find the stream_job function body.
    start = content.find("async def stream_job(")
    assert start != -1, "stream_job endpoint not found"
    # Look at the next ~3000 chars.
    body = content[start:start + 3000]
    # Must check global cap.
    assert "SSE_MAX_GLOBAL" in body, (
        "M188 regression: stream_job must check global SSE cap before "
        "accepting connection."
    )
    # Must check per-IP cap.
    assert "SSE_MAX_PER_IP" in body, (
        "M188 regression: stream_job must check per-IP SSE cap."
    )
    # Must increment the counter.
    assert "_sse_open_counts[ip]" in body, (
        "M188 regression: per-IP counter increment missing — check would "
        "be inert without tracking the open count."
    )


def test_event_generator_decrements_counter_in_finally():
    """The generator's finally block must decrement the per-IP counter
    AND drop the dict entry when count hits 0. Without these the per-IP
    count grows monotonically and locks the IP out forever after
    SSE_MAX_PER_IP closed streams."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find the event_generator finally block within stream_job. The
    # function body now includes the M188 cap-check preamble + the M180
    # subscriber-cleanup block + the M188 counter-decrement block, so
    # use a generous window — the decrement lives past 3000 chars.
    start = content.find("async def stream_job(")
    body = content[start:start + 5000]
    assert "_sse_open_counts[ip]" in body and "- 1" in body, (
        "M188 regression: per-IP counter decrement missing from "
        "event_generator finally — counter would grow monotonically."
    )
    # Drop the dict entry when count <= 0 (matches M180 ghost-key cleanup).
    assert re.search(r"_sse_open_counts\.pop\(ip", body), (
        "M188 regression: per-IP counter dict-entry cleanup missing — "
        "would leave ghost {ip: 0} entries (same M180 anti-pattern)."
    )


def test_sse_caps_documented_in_source():
    """Comments must explain the rationale (FD exhaustion). Future readers
    need to know WHY the caps exist before tuning them."""
    content = SERVER_PY.read_text(encoding="utf-8")
    assert "M188" in content, "M188 rationale comments missing from server.py"
    # At least one comment mentions file descriptors as the reason.
    assert "file descriptor" in content.lower() or "fd limit" in content.lower(), (
        "M188 rationale must mention file-descriptor exhaustion as the "
        "reason for the cap (otherwise future tuning could remove it)."
    )
