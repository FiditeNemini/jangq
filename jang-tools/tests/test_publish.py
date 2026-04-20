"""Tests for jang_tools.publish.

We never actually hit HF here — use --dry-run to exercise the code path.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def fake_converted(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen3", "_name_or_path": "Qwen/Qwen3-0.6B"}))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.23, "block_size": 64, "bit_widths_used": [4]},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1000)
    return d


def test_cli_rejects_missing_token(fake_converted):
    env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model"],
        capture_output=True, text=True, check=False, env=env,
    )
    assert r.returncode == 2
    assert "HF_HUB_TOKEN" in r.stderr


def test_cli_dry_run(fake_converted, monkeypatch):
    monkeypatch.setenv("HF_HUB_TOKEN", "dummy_token")
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model", "--dry-run", "--json"],
        capture_output=True, text=True, check=False,
        env={**os.environ, "HF_HUB_TOKEN": "dummy_token"},
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["dry_run"] is True
    assert data["repo"] == "test/model"
    assert data["files_count"] >= 4   # config, jang_config, tokenizer_config, safetensors, README (generated)


def test_dry_run_generates_readme(fake_converted, monkeypatch):
    monkeypatch.setenv("HF_HUB_TOKEN", "dummy")
    assert not (fake_converted / "README.md").exists()
    subprocess.run(
        [sys.executable, "-m", "jang_tools", "publish",
         "--model", str(fake_converted), "--repo", "test/model", "--dry-run"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "HF_HUB_TOKEN": "dummy"},
    )
    assert (fake_converted / "README.md").exists()
    card = (fake_converted / "README.md").read_text()
    assert "license:" in card
    assert "base_model:" in card
