"""Smoke test: JANGTQ scripts accept --progress=json and emit a done event."""
import json
import subprocess
import sys


def _run_module(mod: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", mod, "--progress=json", "--quiet-text", *args],
        capture_output=True, text=True, check=False,
    )


def test_qwen35_jangtq_emits_done_on_bad_input(tmp_path):
    r = _run_module("jang_tools.convert_qwen35_jangtq",
                    ["/tmp/nope_xyz", str(tmp_path / "out"), "JANGTQ2"])
    assert r.returncode != 0
    lines = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    assert any(e.get("type") == "done" and e.get("ok") is False for e in lines)


def test_minimax_jangtq_emits_done_on_bad_input(tmp_path):
    r = _run_module("jang_tools.convert_minimax_jangtq",
                    ["/tmp/nope_xyz", str(tmp_path / "out"), "JANGTQ2"])
    assert r.returncode != 0
    lines = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    assert any(e.get("type") == "done" and e.get("ok") is False for e in lines)
