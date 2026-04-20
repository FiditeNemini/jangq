"""Tests for jang_tools.examples — capability-aware snippet generation."""
import json
import subprocess
import sys
from pathlib import Path
import pytest

from jang_tools.examples import detect_capabilities, render_snippet, SUPPORTED_LANGS

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


@pytest.fixture
def dense_model_dir(tmp_path):
    """A minimal valid JANG output for a dense chat model."""
    d = tmp_path / "dense"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "qwen3",
        "num_hidden_layers": 28,
        "_name_or_path": "Qwen/Qwen3-0.6B-Base",
    }))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "format_version": "2.0",
        "family": "jang", "profile": "JANG_4K",
        "capabilities": {"arch": "qwen3"},
        "quantization": {"actual_bits_per_weight": 4.23, "block_size": 64, "bit_widths_used": [3, 4, 6, 8]},
        "source_model": "Qwen/Qwen3-0.6B-Base",
    }))
    (d / "tokenizer_config.json").write_text(json.dumps({
        "chat_template": "{% for m in messages %}{{m.content}}{% endfor %}",
        "tokenizer_class": "Qwen2Tokenizer",
    }))
    return d


@pytest.fixture
def vl_model_dir(tmp_path):
    d = tmp_path / "vl"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "num_hidden_layers": 28}))
    (d / "jang_config.json").write_text(json.dumps({
        "format": "jang", "family": "jang", "profile": "JANG_4K",
        "quantization": {"actual_bits_per_weight": 4.0, "block_size": 64},
    }))
    (d / "preprocessor_config.json").write_text("{}")
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": "x"}))
    return d


def test_detect_dense_capabilities(dense_model_dir):
    caps = detect_capabilities(dense_model_dir)
    assert caps["model_type"] == "qwen3"
    assert caps["is_vl"] is False
    assert caps["is_video_vl"] is False
    assert caps["has_chat_template"] is True
    assert caps["profile"] == "JANG_4K"


def test_detect_vl_capabilities(vl_model_dir):
    caps = detect_capabilities(vl_model_dir)
    assert caps["is_vl"] is True
    assert caps["model_type"] == "qwen2_vl"


def test_render_python_dense(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "python")
    # Should use dense loader
    assert "from jang_tools.loader import load_model" in snippet
    assert "apply_chat_template" in snippet


def test_render_python_vl(vl_model_dir):
    snippet = render_snippet(vl_model_dir, "python")
    assert "load_jangtq_vlm" in snippet
    assert "image" in snippet.lower()


def test_render_swift_includes_path(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "swift")
    assert "JANGCore" in snippet
    assert str(dense_model_dir.resolve()) in snippet


def test_render_server_has_curl(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "server")
    assert "osaurus" in snippet.lower()
    assert "curl" in snippet
    assert "localhost:8080" in snippet


def test_render_hf(dense_model_dir):
    snippet = render_snippet(dense_model_dir, "hf")
    assert "pip install" in snippet or "Python" in snippet


def test_render_rejects_unknown_lang(dense_model_dir):
    with pytest.raises(ValueError):
        render_snippet(dense_model_dir, "rust")


def test_cli_examples_json(dense_model_dir):
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "examples",
         "--model", str(dense_model_dir), "--lang", "python", "--json"],
        capture_output=True, text=True, check=False,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert data["lang"] == "python"
    assert "from jang_tools.loader import load_model" in data["snippet"]


def test_cli_python_snippet_compiles(dense_model_dir):
    """Generated Python snippet must be syntactically valid."""
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "examples",
         "--model", str(dense_model_dir), "--lang", "python", "--json"],
        capture_output=True, text=True, check=True,
    )
    snippet = json.loads(r.stdout)["snippet"]
    # Don't EXEC — just compile. We can't actually import mlx in test env
    compile(snippet, "<eval>", "exec")  # raises SyntaxError if broken
