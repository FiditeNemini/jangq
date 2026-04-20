"""Unit tests for ralph_runner.audit — shape + registry.

MLX-dependent rows (a1-a5, a15) run only when mlx_lm is importable AND
a real model is available. For hermetic unit tests we stub and only test
dispatch + registry behavior.
"""
import json
import subprocess
import sys
from pathlib import Path
import pytest

from ralph_runner.audit import AUDIT_REGISTRY, audit_a7_size_estimate


@pytest.fixture
def mock_model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text('{"model_type":"qwen3"}')
    (d / "jang_config.json").write_text('{"format":"jang","format_version":"2.0","capabilities":{"arch":"qwen3"},"quantization":{"actual_bits_per_weight":4.23,"block_size":64,"bit_widths_used":[4]}}')
    (d / "tokenizer.json").write_text('{"model":{"type":"BPE"}}')
    (d / "tokenizer_config.json").write_text('{"tokenizer_class":"Qwen2Tokenizer"}')
    (d / "special_tokens_map.json").write_text('{}')
    # 1 MB fake shard
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1_000_000)
    return d


def test_registry_has_expected_rows():
    for row in ["a1", "a2", "a3", "a4", "a5", "a7", "a15"]:
        assert row in AUDIT_REGISTRY
    # a1 and a15 are required
    assert AUDIT_REGISTRY["a1"][2] is True
    assert AUDIT_REGISTRY["a15"][2] is True


def test_a7_passes_when_no_prediction(mock_model_dir):
    r = audit_a7_size_estimate(mock_model_dir, predicted_bytes=None)
    assert r["status"] == "pass"
    assert r["actual_gb"] == 0.001   # 1 MB


def test_a7_warns_on_large_drift(mock_model_dir):
    r = audit_a7_size_estimate(mock_model_dir, predicted_bytes=10_000_000)  # predicted 10 MB, actual 1 MB
    assert r["status"] == "warn"
    assert r["ratio"] == 0.1


def test_a7_passes_near_estimate(mock_model_dir):
    r = audit_a7_size_estimate(mock_model_dir, predicted_bytes=1_050_000)  # 5% over
    assert r["status"] == "pass"


def test_cli_rejects_missing_model(tmp_path):
    r = subprocess.run(
        [sys.executable, "-m", "ralph_runner.audit", "--model", str(tmp_path / "nope")],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 2
    data = json.loads(r.stdout)
    assert "error" in data


def test_cli_unknown_row_is_na(mock_model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "ralph_runner.audit",
         "--model", str(mock_model_dir),
         "--rows", "a7,a999_unknown",
         "--json"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0
    data = json.loads(r.stdout)
    assert data["rows"]["a999_unknown"]["status"] == "n/a"
    assert data["rows"]["a7"]["status"] in ("pass", "warn")
