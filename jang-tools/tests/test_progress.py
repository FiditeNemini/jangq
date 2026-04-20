# jang-tools/tests/test_progress.py
"""Tests for jang_tools.progress — ProgressEmitter API and JSONL schema."""
import io
import json
import pytest

from jang_tools.progress import ProgressEmitter


def _drain(emitter: ProgressEmitter) -> list[dict]:
    """Parse the emitter's JSONL buffer into a list of event dicts."""
    raw = emitter._stderr.getvalue()
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def test_phase_event_shape():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.phase(1, 5, "detect")
    events = _drain(em)
    assert len(events) == 1
    ev = events[0]
    assert ev["v"] == 1
    assert ev["type"] == "phase"
    assert ev["n"] == 1
    assert ev["total"] == 5
    assert ev["name"] == "detect"
    assert isinstance(ev["ts"], (int, float))


def test_tick_event_shape():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.tick(1234, 2630, "layer.5.gate_proj")
    ev = _drain(em)[0]
    assert ev["type"] == "tick"
    assert ev["done"] == 1234
    assert ev["total"] == 2630
    assert ev["label"] == "layer.5.gate_proj"


def test_event_warn_and_info():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.event("warn", "No chat template found")
    em.event("info", "Detected qwen3_5_moe")
    evs = _drain(em)
    assert evs[0]["type"] == "warn"
    assert evs[0]["msg"] == "No chat template found"
    assert evs[1]["type"] == "info"
    em.event("error", "Database connection failed")
    assert _drain(em)[2]["type"] == "error"


def test_done_success_and_failure():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.done(ok=True, output="/tmp/out", elapsed_s=12.5)
    ev = _drain(em)[0]
    assert ev["type"] == "done"
    assert ev["ok"] is True
    assert ev["output"] == "/tmp/out"
    assert ev["elapsed_s"] == 12.5

    em2 = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em2.done(ok=False, error="OOM while loading experts")
    ev2 = _drain(em2)[0]
    assert ev2["ok"] is False
    assert ev2["error"] == "OOM while loading experts"


def test_text_mode_only_no_json():
    out, err = io.StringIO(), io.StringIO()
    em = ProgressEmitter(json_to_stderr=False, quiet_text=False, _stdout=out, _stderr=err)
    em.phase(1, 5, "detect")
    assert err.getvalue() == ""           # no JSONL
    assert "[1/5]" in out.getvalue()       # human-readable on stdout


def test_quiet_text_suppresses_stdout():
    out, err = io.StringIO(), io.StringIO()
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stdout=out, _stderr=err)
    em.phase(1, 5, "detect")
    assert out.getvalue() == ""            # suppressed
    assert err.getvalue() != ""            # JSONL still written


def test_tick_throttling_coalesces():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    # Fire 100 ticks back-to-back — should coalesce to ≤ a handful
    for i in range(100):
        em.tick(i, 100, f"t{i}")
    events = _drain(em)
    # Must always emit the final tick (done == total) but coalesce the rest
    assert len(events) < 50
    assert events[-1]["done"] == 99 or events[-1]["done"] == 100


def test_warn_never_throttled():
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    em.tick(1, 100, "a")
    em.event("warn", "something")
    em.tick(2, 100, "b")
    types = [e["type"] for e in _drain(em)]
    assert "warn" in types


# ──────────────────────────────────────────────────────────────────
# Iter 11: M62 env passthrough — JANG_TICK_THROTTLE_MS resolution
# ──────────────────────────────────────────────────────────────────

def test_tick_throttle_default_when_env_unset(monkeypatch):
    from jang_tools.progress import _resolve_tick_interval_s
    monkeypatch.delenv("JANG_TICK_THROTTLE_MS", raising=False)
    assert _resolve_tick_interval_s() == 0.1


def test_tick_throttle_reads_env(monkeypatch):
    from jang_tools.progress import _resolve_tick_interval_s
    monkeypatch.setenv("JANG_TICK_THROTTLE_MS", "250")
    assert _resolve_tick_interval_s() == 0.25


def test_tick_throttle_garbage_env_falls_back(monkeypatch):
    # Misconfigured env must NOT hang the emit loop (zero or negative interval
    # would mean the throttle never coalesces, which is worse than a too-long
    # interval). Any invalid value falls back to the 100 ms default.
    from jang_tools.progress import _resolve_tick_interval_s
    for val in ["", "not-a-number", "0", "-50", "   "]:
        monkeypatch.setenv("JANG_TICK_THROTTLE_MS", val)
        assert _resolve_tick_interval_s() == 0.1, f"should fall back for {val!r}"


def test_emitter_applies_env_throttle(monkeypatch):
    # End-to-end: with throttle=1ms, a tight loop of 100 ticks should emit
    # ALMOST all of them (vs the default 100ms throttle test above which keeps
    # <50). This verifies the per-instance throttle actually overrides the
    # module constant.
    monkeypatch.setenv("JANG_TICK_THROTTLE_MS", "1")
    em = ProgressEmitter(json_to_stderr=True, quiet_text=True, _stderr=io.StringIO(), _stdout=io.StringIO())
    # Add a 2 ms sleep between ticks so the 1 ms throttle lets each through.
    import time
    for i in range(10):
        em.tick(i, 10, f"t{i}")
        time.sleep(0.002)
    events = _drain(em)
    # All 10 should land, not coalesced to single digits.
    assert len(events) >= 8, f"1 ms throttle should preserve most ticks, got {len(events)}"
