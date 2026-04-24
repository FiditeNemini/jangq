"""Thin wrapper around DSV4's custom `encoding_dsv4.py`."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

def _load_encoding_module(encoding_dir: Path | None = None):
    d = encoding_dir
    if d is None:
        env = os.environ.get("DSV4_ENCODING_DIR")
        if not env:
            raise RuntimeError(
                "DSV4 encoding_dsv4.py module path not set. Either pass "
                "encoding_dir=Path('<source>/encoding') or set the "
                "DSV4_ENCODING_DIR env var to the directory containing "
                "encoding_dsv4.py from your DeepSeek-V4-Flash bundle."
            )
        d = Path(env)
    f = d / "encoding_dsv4.py"
    if not f.exists():
        raise FileNotFoundError(
            f"encoding_dsv4.py not found at {f}. Set DSV4_ENCODING_DIR env "
            "var or ensure the DeepSeek-V4-Flash source is downloaded."
        )
    spec = importlib.util.spec_from_file_location("encoding_dsv4", str(f))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["encoding_dsv4"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = None


def _get():
    global _mod
    if _mod is None:
        _mod = _load_encoding_module()
    return _mod


def apply_chat_template(
    messages: list[dict],
    *,
    thinking_mode: str = "thinking",
    context: list[dict] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | None = None,
) -> str:
    """Apply DSV4's chat template to messages → prompt string.

    Tools are defined inline on the system/developer message via its
    `tools` field (OpenAI format). DSV4's encoder reads them from there.
    """
    m = _get()
    return m.encode_messages(
        messages,
        thinking_mode=thinking_mode,
        context=context,
        drop_thinking=drop_thinking,
        add_default_bos_token=add_default_bos_token,
        reasoning_effort=reasoning_effort,
    )


def parse_completion(raw_text: str, *, thinking_mode: str = "thinking") -> dict:
    """Parse DSV4's raw completion → structured assistant message.

    Returns: {"role": "assistant", "reasoning_content": str,
              "content": str, "tool_calls": list}.
    """
    m = _get()
    return m.parse_message_from_completion_text(raw_text, thinking_mode=thinking_mode)
