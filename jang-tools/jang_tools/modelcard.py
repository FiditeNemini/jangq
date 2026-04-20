"""Generate a HuggingFace-compatible model card for a converted JANG/JANGTQ model."""
from __future__ import annotations
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from .examples import detect_capabilities, render_snippet


def _env() -> Environment:
    return Environment(
        loader=PackageLoader("jang_tools"),
        autoescape=select_autoescape(default=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _size_gb(model_dir: Path) -> float:
    total = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))
    return round(total / 1_000_000_000, 2) if total else 0.0


def generate_card(model_dir: Path) -> str:
    caps = detect_capabilities(model_dir)
    caps["size_gb"] = _size_gb(model_dir)
    caps["date"] = dt.date.today().isoformat()
    # Render Python snippet to embed inside the card
    caps["snippet_python"] = render_snippet(model_dir, "python")
    tpl = _env().get_template("model-card.md.jinja")
    return tpl.render(**caps)


def cmd_modelcard(args) -> None:
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)
    try:
        card = generate_card(model_dir)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    if args.json:
        caps = detect_capabilities(model_dir)
        caps["size_gb"] = _size_gb(model_dir)
        out = {
            "license": caps.get("license") or "apache-2.0",
            "base_model": caps["base_model"],
            "quantization_config": {
                "family": caps["family"],
                "profile": caps["profile"],
                "actual_bits": caps["actual_bits"],
                "block_size": caps["block_size"],
                "size_gb": caps["size_gb"],
            },
            "card_markdown": card,
        }
        print(json.dumps(out, indent=None))
    else:
        if args.output:
            Path(args.output).write_text(card)
            print(f"wrote {args.output}")
        else:
            print(card)


def register(subparsers) -> None:
    p = subparsers.add_parser("modelcard", help="Generate HuggingFace model card for a converted model")
    p.add_argument("--model", required=True, help="Path to converted JANG/JANGTQ model dir")
    p.add_argument("--output", help="Write card to file (default stdout)")
    p.add_argument("--json", action="store_true", help="Emit JSON with metadata + card markdown")
    p.set_defaults(func=cmd_modelcard)
