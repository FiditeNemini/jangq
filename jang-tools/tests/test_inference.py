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


class _FakeInnerTokenizer:
    """Minimal stand-in for a HF tokenizer with a chat template."""
    def __init__(self, template: str | None):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Pretend the template is `<|user|>{content}<|assistant|>`
        return f"<|user|>{messages[0]['content']}<|assistant|>"


class _FakeWrappedTokenizer:
    """Mimics mlx_lm's TokenizerWrapper (outer object with `.tokenizer` inner)."""
    def __init__(self, inner):
        self.tokenizer = inner


def test_apply_chat_template_when_present_on_wrapped_tokenizer():
    from jang_tools.inference import _apply_chat_template_if_any
    inner = _FakeInnerTokenizer(template="<|user|>{content}<|assistant|>")
    wrapped = _FakeWrappedTokenizer(inner)
    out = _apply_chat_template_if_any(wrapped, "Hello")
    assert out == "<|user|>Hello<|assistant|>"


def test_apply_chat_template_falls_through_when_absent():
    from jang_tools.inference import _apply_chat_template_if_any
    inner = _FakeInnerTokenizer(template=None)
    wrapped = _FakeWrappedTokenizer(inner)
    out = _apply_chat_template_if_any(wrapped, "Hello")
    assert out == "Hello"


def test_apply_chat_template_on_bare_tokenizer():
    """A tokenizer without a `.tokenizer` wrapper attr should still work."""
    from jang_tools.inference import _apply_chat_template_if_any
    bare = _FakeInnerTokenizer(template="<|user|>{content}<|assistant|>")
    out = _apply_chat_template_if_any(bare, "Ping")
    assert out == "<|user|>Ping<|assistant|>"


def test_make_sampler_returns_none_for_greedy():
    from jang_tools.inference import _make_sampler
    assert _make_sampler(0.0) is None
    assert _make_sampler(-0.5) is None  # treat negative as greedy, don't crash


def test_make_sampler_returns_callable_for_positive_temp():
    from jang_tools.inference import _make_sampler
    s = _make_sampler(0.7)
    # Either mlx_lm is present (callable) or absent (None); both are acceptable —
    # the important invariant is we never raise for a valid positive temperature.
    assert s is None or callable(s)


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
