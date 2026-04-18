"""Verify inspect-source returns valid JSON with expected keys."""
import json
import subprocess
import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_qwen"


def test_inspect_source_prints_valid_json():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(FIXTURE)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["model_type"] == "qwen3_5_moe"
    assert data["is_moe"] is True
    assert data["num_experts"] == 8
    assert data["dtype"] in ("bfloat16", "float16", "float8_e4m3fn", "unknown")
    assert "jangtq_compatible" in data
    assert data["jangtq_compatible"] is True   # qwen3_5_moe is in the v1 whitelist


def test_inspect_source_missing_config_errors(tmp_path):
    (tmp_path / "README").write_text("nope")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inspect-source", "--json", str(tmp_path)],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode != 0
    assert "config.json" in r.stderr.lower()
