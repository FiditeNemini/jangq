"""Generate capability-aware code snippets for converted JANG/JANGTQ models.

Reads the model's config.json + jang_config.json + tokenizer_config.json
to detect capabilities, then renders a Jinja template.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from ._json_utils import read_json_object as _read_json_object

SUPPORTED_LANGS = ("python", "swift", "server", "hf")

_TEMPLATE_MAP = {
    "python": "python-snippet.py.jinja",
    "swift": "swift-snippet.swift.jinja",
    "server": "server-snippet.sh.jinja",
    "hf": "hf-snippet.md.jinja",
}


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("jang_tools"),
        autoescape=select_autoescape(default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )


# M93 (iter 29): model_types that are DEFINITIONALLY text-only. Rule 11 of
# feedback_readme_standards.md: "MiniMax is text-only — never include VLM code".
# Enforced here so a stray preprocessor_config.json in a MiniMax output dir
# (e.g. from copy-paste or a prior VL convert) can't silently flip is_vl=True
# and cause the template to emit VLM imports/snippets that won't work at runtime.
_TEXT_ONLY_MODEL_TYPES = frozenset({
    "minimax_m2",      # MiniMax M2 / M2.5 / M2.7 — text-only MoE
    "minimax",         # bare alias, mapped to minimax_m2 in capabilities.py
    "minimax_m2_5",    # explicit alias
})


def detect_capabilities(model_dir: Path) -> dict[str, Any]:
    """Inspect a converted model dir for capability flags needed by templates."""
    # M126 (iter 73): route all 3 reads through _read_json_object so a
    # corrupted post-convert artifact surfaces with the failing file's
    # path + purpose in the error. Pre-iter-73 the outer cmd_examples
    # try/except emitted `ERROR: JSONDecodeError: Expecting value: …`
    # without naming the file — user had 3+ candidates to check.
    cfg_path = model_dir / "config.json"
    cfg = _read_json_object(cfg_path, purpose="config.json")
    jang_cfg_path = model_dir / "jang_config.json"
    jang_cfg: dict[str, Any] = (
        _read_json_object(jang_cfg_path, purpose="jang_config.json")
        if jang_cfg_path.exists()
        else {}
    )
    tok_cfg_path = model_dir / "tokenizer_config.json"
    tok_cfg: dict[str, Any] = (
        _read_json_object(tok_cfg_path, purpose="tokenizer_config.json")
        if tok_cfg_path.exists()
        else {}
    )

    model_type = cfg.get("model_type") or cfg.get("text_config", {}).get("model_type", "")
    quant = jang_cfg.get("quantization", {})
    has_jinja = (model_dir / "chat_template.jinja").exists()
    has_inline_template = bool(tok_cfg.get("chat_template"))
    has_chat_template = (
        has_jinja
        or has_inline_template
        or (model_dir / "chat_template.json").exists()
    )

    tool_parser = cfg.get("tool_call_parser") or cfg.get("tool_choice_parser")
    reasoning_parser = cfg.get("reasoning_parser") or cfg.get("thinking_parser")

    # Compute actual_bits from whichever field is available
    actual_bits_raw = (
        quant.get("actual_bits_per_weight")
        or quant.get("actual_bits")
    )
    if actual_bits_raw is None:
        bw_list = quant.get("bit_widths_used") or [4]
        actual_bits_raw = sum(bw_list) / max(len(bw_list), 1)

    return {
        "model_path": str(model_dir.resolve()),
        "model_name": model_dir.name,
        "family": jang_cfg.get("family") or (
            "jangtq" if "tq_codebook" in quant else "jang"
        ),
        "profile": jang_cfg.get("profile", "JANG_4K"),
        "actual_bits": actual_bits_raw,
        "block_size": quant.get("block_size", 64),
        "base_model": (
            cfg.get("_name_or_path")
            or jang_cfg.get("source_model")
            or model_type
        ),
        "license": cfg.get("license"),
        "model_type": model_type,
        # M93: force text-only for known-text-only model types regardless of
        # what files are in the dir. A stray preprocessor_config.json (copy
        # residue, bad convert, etc.) must NOT flip is_vl=True on MiniMax
        # and cause the template to emit VLM snippets that break at runtime.
        "is_vl": (
            model_type not in _TEXT_ONLY_MODEL_TYPES
            and (model_dir / "preprocessor_config.json").exists()
        ),
        "is_video_vl": (
            model_type not in _TEXT_ONLY_MODEL_TYPES
            and (model_dir / "video_preprocessor_config.json").exists()
        ),
        "has_chat_template": has_chat_template,
        "has_tool_parser": bool(tool_parser),
        "tool_parser": tool_parser or "",
        "has_reasoning": bool(reasoning_parser) or bool(cfg.get("enable_thinking")),
        # M90 (iter 27): `feedback_readme_standards.md` rule 10 requires
        # BOTH `reasoning` AND `thinking` tags for Qwen3.5 models. Before
        # this, only `has_reasoning` was emitted — the template omitted a
        # dedicated `thinking` tag. `has_thinking` specifically tracks the
        # runtime `enable_thinking` config (what users toggle on/off during
        # inference), while `has_reasoning` is the broader capability flag.
        "has_thinking": bool(cfg.get("enable_thinking")),
    }


def render_snippet(model_dir: Path, lang: str) -> str:
    if lang not in SUPPORTED_LANGS:
        raise ValueError(f"lang must be one of {SUPPORTED_LANGS}, got {lang}")
    caps = detect_capabilities(model_dir)
    # hf template embeds an abbreviated python snippet
    if lang == "hf":
        short_tpl = _env().get_template("python-snippet.py.jinja")
        caps["snippet_python_short"] = short_tpl.render(**caps)
    tpl = _env().get_template(_TEMPLATE_MAP[lang])
    return tpl.render(**caps)


def cmd_examples(args) -> None:
    """CLI entry: python -m jang_tools examples --model <dir> --lang <lang> [--json]"""
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)
    try:
        snippet = render_snippet(model_dir, args.lang)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    if args.json:
        print(json.dumps({"lang": args.lang, "snippet": snippet, "model": str(model_dir)}, indent=None))
    else:
        print(snippet)


def register(subparsers) -> None:
    p = subparsers.add_parser("examples", help="Generate usage snippets for a converted model")
    p.add_argument("--model", required=True, help="Path to converted JANG/JANGTQ model dir")
    p.add_argument("--lang", required=True, choices=SUPPORTED_LANGS, help="Target language")
    p.add_argument("--json", action="store_true", help="Emit single JSON line on stdout")
    p.set_defaults(func=cmd_examples)
