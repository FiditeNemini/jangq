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
        # M121: record what kwargs the last apply_chat_template call received,
        # so tests can pin that enable_thinking is piped through correctly.
        self.last_call_kwargs: dict = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.last_call_kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        }
        # Reflect enable_thinking in the output so callers can see the toggle
        # took effect even if the tokenizer doesn't actually know the kwarg.
        thinking_tag = "[THINKING]" if kwargs.get("enable_thinking", True) else "[NO-THINK]"
        return f"<|user|>{thinking_tag}{messages[0]['content']}<|assistant|>"


class _FakeWrappedTokenizer:
    """Mimics mlx_lm's TokenizerWrapper (outer object with `.tokenizer` inner)."""
    def __init__(self, inner):
        self.tokenizer = inner


def test_apply_chat_template_when_present_on_wrapped_tokenizer():
    from jang_tools.inference import _apply_chat_template_if_any
    inner = _FakeInnerTokenizer(template="<|user|>{content}<|assistant|>")
    wrapped = _FakeWrappedTokenizer(inner)
    out = _apply_chat_template_if_any(wrapped, "Hello")
    # Default enable_thinking=True → template marks the thinking path.
    assert out == "<|user|>[THINKING]Hello<|assistant|>"


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
    assert out == "<|user|>[THINKING]Ping<|assistant|>"


# M121: reasoning-model smoke-test flag — enable_thinking=False short-circuits
# the <think>...</think> block that eats 100+ tokens on GLM-5.1/Qwen3.6/
# MiniMax M2.7. Pre-fix, TestInferenceSheet smoke tests on reasoning models
# showed partial thinking with no answer. Now opt-in via --no-thinking CLI.

def test_apply_chat_template_pipes_enable_thinking_false():
    """When enable_thinking=False, the kwarg must arrive at the template."""
    from jang_tools.inference import _apply_chat_template_if_any
    inner = _FakeInnerTokenizer(template="<|user|>{content}<|assistant|>")
    wrapped = _FakeWrappedTokenizer(inner)
    out = _apply_chat_template_if_any(wrapped, "2+2?", enable_thinking=False)
    # Output reflects the no-think branch of our fake template.
    assert out == "<|user|>[NO-THINK]2+2?<|assistant|>"
    # And the kwarg actually reached apply_chat_template.
    assert inner.last_call_kwargs.get("enable_thinking") is False
    assert inner.last_call_kwargs.get("add_generation_prompt") is True


def test_apply_chat_template_default_keeps_thinking_on():
    """Regression guard: omitting the kwarg preserves existing behavior."""
    from jang_tools.inference import _apply_chat_template_if_any
    inner = _FakeInnerTokenizer(template="<|user|>{content}<|assistant|>")
    wrapped = _FakeWrappedTokenizer(inner)
    out = _apply_chat_template_if_any(wrapped, "Explain quantum mechanics.")
    # No explicit toggle → kwarg defaults to True (existing behavior).
    assert "[THINKING]" in out
    assert inner.last_call_kwargs.get("enable_thinking") is True


def test_apply_chat_template_no_thinking_survives_template_error(monkeypatch):
    """If the template RAISES on the enable_thinking kwarg (ancient tokenizer
    that strictly rejects unknown kwargs), we must still produce something
    usable rather than silently dropping the prompt to raw form."""
    from jang_tools.inference import _apply_chat_template_if_any

    class _StrictTokenizer:
        chat_template = "strict"

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected kwarg enable_thinking")
            return f"<|user|>{messages[0]['content']}<|assistant|>"

    out = _apply_chat_template_if_any(_StrictTokenizer(), "Hi", enable_thinking=False)
    # Retry without the kwarg must succeed — not fall all the way to raw "Hi".
    assert out == "<|user|>Hi<|assistant|>"


def test_cli_help_lists_no_thinking_flag():
    """Surface the flag in --help so wizard users can discover it via the CLI."""
    r = subprocess.run(
        [sys.executable, "-m", "jang_tools", "inference", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "--no-thinking" in r.stdout


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
