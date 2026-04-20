# jang-tools/jang_tools/progress.py
"""Progress emitter for jang-tools — supports human text and JSONL.

JSONL schema v1 (one object per line on stderr):
    {"v":1,"type":"phase","n":1,"total":5,"name":"detect","ts":1700000000.123}
    {"v":1,"type":"tick","done":1234,"total":2630,"label":"layer.5","ts":...}
    {"v":1,"type":"info"|"warn"|"error","msg":"...","ts":...}
    {"v":1,"type":"done","ok":true,"output":"/path","elapsed_s":12.5,"ts":...}
    {"v":1,"type":"done","ok":false,"error":"...","ts":...}

See docs/PROGRESS_PROTOCOL.md for full spec.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, IO

PROTOCOL_VERSION = 1
_DEFAULT_TICK_MIN_INTERVAL_S = 0.1


def _resolve_tick_interval_s() -> float:
    """Resolve the tick throttle interval from JANG_TICK_THROTTLE_MS env, or
    fall back to the default 100 ms. Swift-side AppSettings mirrors the user
    setting into this env var for JANGStudio child processes (M62, iter 11).
    Accepts only positive integers — garbage / zero / negative falls back to
    the default so a misconfigured env can't hang emit loops.
    """
    raw = os.environ.get("JANG_TICK_THROTTLE_MS", "").strip()
    if not raw:
        return _DEFAULT_TICK_MIN_INTERVAL_S
    try:
        ms = int(raw)
    except ValueError:
        return _DEFAULT_TICK_MIN_INTERVAL_S
    if ms <= 0:
        return _DEFAULT_TICK_MIN_INTERVAL_S
    return ms / 1000.0


class ProgressEmitter:
    def __init__(
        self,
        json_to_stderr: bool,
        quiet_text: bool,
        _stdout: IO[str] | None = None,
        _stderr: IO[str] | None = None,
    ) -> None:
        self._json = json_to_stderr
        self._quiet = quiet_text
        self._stdout = _stdout if _stdout is not None else sys.stdout
        self._stderr = _stderr if _stderr is not None else sys.stderr
        self._last_tick_ts = 0.0
        self._tick_min_interval_s = _resolve_tick_interval_s()

    def _emit_json(self, payload: dict[str, Any]) -> None:
        if not self._json:
            return
        payload["v"] = PROTOCOL_VERSION
        payload.setdefault("ts", time.time())
        self._stderr.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self._stderr.flush()

    def _emit_text(self, line: str) -> None:
        if self._quiet:
            return
        self._stdout.write(line + "\n")
        self._stdout.flush()

    def phase(self, n: int, total: int, name: str) -> None:
        self._emit_json({"type": "phase", "n": n, "total": total, "name": name})
        self._emit_text(f"  [{n}/{total}] {name}")

    def tick(self, done: int, total: int, label: str = "") -> None:
        now = time.time()
        is_final = done >= total - 1
        if not is_final and (now - self._last_tick_ts) < self._tick_min_interval_s:
            return
        self._last_tick_ts = now
        payload: dict[str, Any] = {"type": "tick", "done": done, "total": total}
        if label:
            payload["label"] = label
        self._emit_json(payload)

    def event(self, level: str, message: str, **fields: Any) -> None:
        assert level in ("info", "warn", "error"), f"unknown level {level}"
        payload = {"type": level, "msg": message, **fields}
        self._emit_json(payload)
        if level in ("warn", "error"):
            self._emit_text(f"  [{level.upper()}] {message}")

    def done(
        self,
        ok: bool,
        output: str | None = None,
        elapsed_s: float | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"type": "done", "ok": ok}
        if output is not None:
            payload["output"] = output
        if elapsed_s is not None:
            payload["elapsed_s"] = elapsed_s
        if error is not None:
            payload["error"] = error
        self._emit_json(payload)


def make_noop() -> ProgressEmitter:
    """Emitter that writes nothing. Used when no progress flag is set."""
    return ProgressEmitter(json_to_stderr=False, quiet_text=True)
