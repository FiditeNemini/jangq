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


# M133 (iter 55): peer-helper sweep — `estimate_model.predict`'s no-safetensors
# fallback uses a flat `12 * h² * layers + 2 * h * vocab` formula that assumes
# a dense model and ignores num_experts. Meanwhile `recommend._estimate_params_billion`
# correctly multiplies MLP per-expert by num_experts for MoE. A 256-expert
# MoE whose source dir has no safetensors (unusual but possible: user points
# at a .bin-only snapshot or a corrupted dir) gets a prediction that's off
# by ~100x. Wizard then shows "predicted output: 0.3 GB" for a model that
# will actually produce 80+ GB — guaranteed user confusion.


@pytest.fixture
def moe_config_only_dir(tmp_path):
    """A model dir with ONLY a config.json — no .safetensors. Forces the
    fallback path in predict() to fire."""
    d = tmp_path / "moe-config-only"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "hidden_size": 3072,
        "num_hidden_layers": 48,
        "vocab_size": 151936,
        "intermediate_size": 3072,
        "num_experts": 256,
    }))
    # NO safetensors — _source_bytes returns 0, fallback fires.
    return d


def test_predict_fallback_accounts_for_moe_experts(moe_config_only_dir):
    """Pre-iter-55 the fallback used 12*h²*layers which misses 256-expert MLP
    (3*h*i*num_experts). A 256-expert 3072-hidden model has ~50B parameters;
    the old fallback estimated ~0.5B. Fix: mirror _estimate_params_billion's
    MoE-aware formula so the fallback doesn't silently misreport by 100x."""
    r = predict(moe_config_only_dir, "JANG_4K")
    # qwen3_5_moe with hidden=3072, layers=48, num_experts=256, intermediate=3072:
    # - attn: 4 * 3072² = 37.7 M per layer
    # - mlp per expert: 3 * 3072 * 3072 = 28.3 M
    # - mlp with 256 experts: 7.25 B per layer
    # - 48 layers: ~350 B total params (total expert + attn weights)
    # - vocab 151k: +0.9B embed/lm_head
    # Source bf16 (2 B/param) → ~700 GB.
    # JANG_4K @ 4 bits/weight → 4/16 * 700 GB * 1.05 overhead = ~184 GB.
    # Any sane prediction must be > 30 GB (10x the naive dense estimate).
    assert r["source_bytes"] > 0, "fallback should have produced a non-zero source_bytes"
    assert r["source_gb"] > 100, (
        f"256-expert MoE fallback produced source_gb={r['source_gb']}. "
        f"That would only be correct for a dense model; MoE adds ~100x more MLP "
        f"weights. Fallback must account for num_experts (M133)."
    )


def test_predict_fallback_still_works_for_dense_model(tmp_path):
    """Regression guard: a dense llama (no num_experts) should still produce
    a sane prediction from the fallback — don't over-engineer MoE path and
    break dense."""
    d = tmp_path / "dense-config-only"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "llama",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 128000,
        "intermediate_size": 11008,
    }))
    r = predict(d, "JANG_4K")
    # Dense 7B-class model, bf16 source ~14 GB. JANG_4K → ~3.7 GB.
    assert r["source_bytes"] > 0
    # Loose bounds — dense estimator needn't be perfect but must be in the
    # right order of magnitude.
    assert 5 < r["source_gb"] < 40, (
        f"dense llama 7B-class fallback produced source_gb={r['source_gb']}; "
        f"expected 5-40 GB range"
    )
