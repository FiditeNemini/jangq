"""Smoke tests for the new --progress=json and --quiet-text global flags."""
import json
import subprocess
import sys


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "jang_tools", *args],
        capture_output=True, text=True, check=False,
    )


def test_version_still_works():
    r = _run(["--version"])
    assert r.returncode == 0
    assert "jang-tools" in r.stdout


def test_help_lists_progress_flag():
    r = _run(["--help"])
    assert "--progress" in r.stdout
    assert "--quiet-text" in r.stdout


def test_progress_json_emits_jsonl_on_stderr_for_inspect():
    # inspect on a nonexistent path should emit a "done ok:false" JSON line
    r = _run(["--progress=json", "inspect", "/tmp/definitely_does_not_exist_xyz"])
    assert r.returncode != 0
    lines = [json.loads(l) for l in r.stderr.splitlines() if l.strip().startswith("{")]
    assert any(ev.get("type") == "done" and ev.get("ok") is False for ev in lines)
