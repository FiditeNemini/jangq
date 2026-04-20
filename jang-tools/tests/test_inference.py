"""Tests for jang_tools.inference CLI.

These are UNIT tests that verify the CLI shape + error handling without
actually loading MLX weights (tests would be slow + need a real model).
End-to-end inference is tested separately via ralph_runner/audit.py.
"""
import json
import subprocess
import sys
from pathlib import Path


def test_cli_rejects_missing_model(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inference",
         "--model", str(tmp_path / "nope"),
         "--prompt", "Hello"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 2
    assert "model dir not found" in r.stderr.lower()


def test_cli_json_error_on_load_failure(tmp_path):
    """Empty dir — loader will raise; --json should emit a structured error."""
    d = tmp_path / "empty"
    d.mkdir()
    # Add an incomplete config to trigger load failure
    (d / "config.json").write_text('{"model_type":"nonexistent_arch_xyz"}')
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inference",
         "--model", str(d),
         "--prompt", "Hello",
         "--json",
         "--max-tokens", "5"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 3
    data = json.loads(r.stdout)
    assert "error" in data
    assert data["model"] == str(d)


def test_cli_help():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inference", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--model" in r.stdout
    assert "--prompt" in r.stdout
    assert "--max-tokens" in r.stdout
    assert "--image" in r.stdout
    assert "--video" in r.stdout
    assert "--json" in r.stdout
