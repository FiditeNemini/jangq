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


def generate_card(model_dir: Path) -> tuple[str, bool]:
    """Render the model card for ``model_dir``.

    Returns ``(card_markdown, license_unknown)``. ``license_unknown`` is
    True when ``config.json`` had no ``license`` field — callers use
    this signal to emit a user-visible warning (CLI stderr, Swift UI
    banner). Pre-M202 (iter 138) the template silently defaulted
    missing licenses to ``apache-2.0``; for most upstream HF models
    (Qwen/Llama/etc.) that's a fabrication because they put the license
    in the README YAML, not config.json. Now we render ``other`` in the
    YAML frontmatter (HF's standard marker for custom/unknown) and
    require the publisher to correct it before HF upload.
    """
    caps = detect_capabilities(model_dir)
    caps["size_gb"] = _size_gb(model_dir)
    caps["date"] = dt.date.today().isoformat()
    # Render Python snippet to embed inside the card
    caps["snippet_python"] = render_snippet(model_dir, "python")
    # M202 (iter 138): honest-license pipeline. If `detect_capabilities`
    # didn't find `license` in config.json (common: HF convention puts
    # the license in README YAML, not config.json), mark it as the
    # HF-standard ``other`` tag and signal to the caller that a
    # warning is warranted.
    license_unknown = caps.get("license") is None
    if license_unknown:
        caps["license"] = "other"
    tpl = _env().get_template("model-card.md.jinja")
    card = tpl.render(license_unknown=license_unknown, **caps)
    return card, license_unknown


def cmd_modelcard(args) -> None:
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)
    try:
        card, license_unknown = generate_card(model_dir)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    # M91 (iter 28): warn that the card is a skeleton. Goes to stderr so it
    # doesn't pollute the --json stdout payload or the plain-text card print.
    # Swift GenerateModelCardSheet shows a user-visible banner instead; this
    # stderr note helps CLI users + Ralph harness tail-reads.
    print(
        "NOTE: generated card is a skeleton (metadata + Python snippet). "
        "For HF publishing, add per-subject MMLU scores, JANG-vs-MLX comparisons, "
        "and a Korean section per feedback_readme_standards.md.",
        file=sys.stderr,
    )
    # M202 (iter 138): license-unknown warning. Pre-M202 the template
    # silently defaulted missing licenses to apache-2.0 — every Qwen/
    # Llama/etc source (which keep license in README YAML, not
    # config.json) got a fabricated apache-2.0 tag. Now we emit
    # ``other`` in the YAML AND warn the user to set it explicitly.
    if license_unknown:
        print(
            "NOTE: no `license` key in config.json; YAML frontmatter uses "
            "`license: other` as a placeholder. Before HF publishing, edit "
            "the frontmatter to match the SOURCE model's license (Qwen "
            "License, Llama-3 Community, apache-2.0, mit, etc.) and add "
            "`license_name` / `license_link` if using `other`.",
            file=sys.stderr,
        )
    if args.json:
        caps = detect_capabilities(model_dir)
        caps["size_gb"] = _size_gb(model_dir)
        out = {
            # M202 (iter 138): parity with generate_card — "other" + a
            # separate `license_unknown` signal instead of silently
            # fabricating apache-2.0. Consumers (Swift PublishToHF sheet,
            # Ralph harness parsers) should check license_unknown and
            # prompt the user to set it explicitly.
            "license": caps.get("license") or "other",
            "license_unknown": license_unknown,
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
