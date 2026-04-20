"""Tests for jang_tools.modelcard."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

from jang_tools.modelcard import generate_card


@pytest.fixture
def dense_model_dir(tmp_path):
    d = tmp_path / "dense"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "_name_or_path": "Qwen/Qwen3-0.6B-Base",
        "license": "apache-2.0",
    }))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.23, "block_size": 64, "bit_widths_used": [3, 4, 6, 8]},
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{m.content}}{% endfor %}",
    }))
    # Create a fake shard so size_gb is non-zero
    (d / "model-00001-of-00001.safetensors").write_bytes(b"x" * 1_000_000)
    return d


def test_card_has_frontmatter(dense_model_dir):
    card = generate_card(dense_model_dir)
    assert card.startswith("---")
    assert "license:" in card
    assert "base_model:" in card
    assert "quantization_config:" in card
    assert "family: jang" in card
    assert "profile: JANG_4K" in card


def test_card_has_usage_section(dense_model_dir):
    card = generate_card(dense_model_dir)
    assert "Quick start" in card
    assert "```python" in card
    # M45 (iter 20): symbol is `load_jang_model`, not `load_model`. The
    # `load_model` substring appears INSIDE `load_jang_model` so the old
    # assertion was vacuously true — a pure import-name regression would
    # never have been caught here. Assert the full correct symbol AND
    # assert the bare `load_model(` call is NOT present.
    assert "load_jang_model" in card
    assert "load_model(" not in card, \
        "card contains bare `load_model(` — would ImportError for adopters"


def test_cli_json_output(dense_model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(dense_model_dir), "--json"],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(r.stdout)
    assert data["license"] == "apache-2.0"
    assert data["quantization_config"]["family"] == "jang"
    assert data["quantization_config"]["profile"] == "JANG_4K"
    assert "card_markdown" in data
    assert data["card_markdown"].startswith("---")


def test_cli_writes_file(dense_model_dir, tmp_path):
    out = tmp_path / "README.md"
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "modelcard",
         "--model", str(dense_model_dir), "--output", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert out.exists()
    assert out.read_text().startswith("---")
