"""Tests for jang_tools.estimate_model."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

from jang_tools.estimate_model import predict, _predict_avg_bits


@pytest.fixture
def fake_model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "vocab_size": 151936,
    }))
    # Simulated 1 GB shard
    (d / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 1_000_000_000)
    return d


def test_predict_avg_bits_known_profiles():
    assert _predict_avg_bits("JANG_4K") == 4.0
    assert _predict_avg_bits("JANG_2S") < 4.0
    assert _predict_avg_bits("JANGTQ2") == 2.0
    assert _predict_avg_bits("JANGTQ4") == 4.0


def test_predict_avg_bits_rejects_unknown():
    with pytest.raises(ValueError):
        _predict_avg_bits("JANG_99X")


def test_predict_shape(fake_model_dir):
    r = predict(fake_model_dir, "JANG_4K")
    assert r["source_bytes"] == 1_000_000_000
    assert r["source_gb"] == 1.0
    # Output should be roughly 4/16 * 1 GB + overhead = ~0.26 GB
    assert 0.2 < r["predicted_output_gb"] < 0.3
    assert r["predicted_avg_bits"] == 4.0
    assert r["profile"] == "JANG_4K"


def test_cli_json(fake_model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "estimate-model",
         "--model", str(fake_model_dir),
         "--profile", "JANG_2S",
         "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["profile"] == "JANG_2S"
    assert data["predicted_avg_bits"] < 4.0


def test_cli_rejects_missing_model():
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "estimate-model",
         "--model", "/tmp/definitely_not_there_xyz",
         "--profile", "JANG_4K"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 2
