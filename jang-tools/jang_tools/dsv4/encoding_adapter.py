"""Thin wrapper around DSV4's custom `encoding_dsv4.py`."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

def _default_encoding_dirs() -> list[Path]:
    """Shallow-search standard local DSV4 model roots for encoding_dsv4.py."""
    dirs: list[Path] = []

    def add(path: Path) -> None:
        if path not in dirs:
            dirs.append(path)

    roots = [Path.home() / "models"]
    # Accept either the canonical name or the historical typo'd one.
    extra_root = (
        os.environ.get("VMLX_MODELS_DIR")
        or os.environ.get("VLLM_MODELS_DIR")
        or os.environ.get("VMLINUX_MODELS_DIR")
    )
    if extra_root:
        roots.insert(0, Path(extra_root).expanduser())
    volumes = Path("/Volumes")
    if volumes.exists():
        try:
            roots.extend(p for p in volumes.iterdir() if p.is_dir())
        except Exception:
            pass

    for root in roots:
        try:
            if not root.exists():
                continue
            add(root / "Sources" / "DeepSeek-V4-Flash" / "encoding")
            for pattern in (
                "DeepSeek-V4-Flash*/encoding",
                "JANGQ/DeepSeek-V4-Flash*/encoding",
                "*/DeepSeek-V4-Flash*/encoding",
                "*/*DeepSeek-V4-Flash*/encoding",
            ):
                for match in root.glob(pattern):
                    add(match)
        except Exception:
            continue
    return dirs

def _load_encoding_module(encoding_dir: Path | None = None):
    d = encoding_dir
    if d is None:
        env = os.environ.get("DSV4_ENCODING_DIR")
        if env:
            d = Path(env)
        else:
            for candidate in _default_encoding_dirs():
                if (candidate / "encoding_dsv4.py").exists():
                    d = candidate
                    break
        if d is None:
            raise RuntimeError(
                "DSV4 encoding_dsv4.py module path not set. Either pass "
                "encoding_dir=Path('<source>/encoding') or set the "
                "DSV4_ENCODING_DIR env var to the directory containing "
                "encoding_dsv4.py from your DeepSeek-V4-Flash bundle."
            )
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
