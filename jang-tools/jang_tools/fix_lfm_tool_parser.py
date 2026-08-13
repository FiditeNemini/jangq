"""Restore mlx_lm tool-call detection on LFM2.5 (Liquid) bundles.

Problem
-------
`mlx_lm` picks a tool parser by **string-searching the chat template text**
(`tokenizer_utils._infer_tool_parser`): the marker `<|tool_list_start|>` selects
the `pythonic` parser. The LFM2.5 template renders its tool list as plain text
("List of tools: [") and only emits `<|tool_call_start|>[fn(...)]<|tool_call_end|>`
at call time, so that literal never appears in the template source. Detection
therefore returns None and mlx_lm does no tool parsing at all — even though
`mlx_lm/tool_parsers/pythonic.py` documents exactly LFM's call format.

LiquidAI's upstream fix is to add a Jinja comment containing the literal.

Fix applied here (both, belt and braces)
----------------------------------------
1. `tool_parser_type: "pythonic"` in tokenizer_config.json. This is read FIRST
   (`tokenizer_config.get("tool_parser_type", _infer_tool_parser(...))`), so it
   is robust even if mlx_lm changes its heuristics later.
2. A whitespace-trimmed Jinja comment carrying the literal, appended to every
   template surface the bundle ships (chat_template.jinja and/or the embedded
   `chat_template` in tokenizer_config.json). Uses `{#- ... -#}` rather than
   upstream's bare `{# ... #}` so it provably renders to nothing.

Idempotent, and verifies rendering is byte-identical before/after.

    python -m jang_tools.fix_lfm_tool_parser <bundle_dir> [more...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HINT = "{#- <|tool_list_start|> detection hint for mlx_lm -#}"
MARKER = "<|tool_list_start|>"
PARSER = "pythonic"


def _add_hint(template: str) -> str:
    if MARKER in template:
        return template
    return template + "\n" + HINT


def _render_probe(tmpl_dir: Path) -> list[str]:
    """Render a few shapes so we can prove the hint changed nothing."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tmpl_dir))
    msgs = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "w",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    out = []
    for kw in ({}, {"tools": tools}):
        for agp in (True, False):
            try:
                out.append(tok.apply_chat_template(
                    msgs, add_generation_prompt=agp, tokenize=False, **kw))
            except Exception as e:  # noqa: BLE001
                out.append(f"<error {type(e).__name__}>")
    return out


def fix(bundle: Path) -> dict:
    tc_path = bundle / "tokenizer_config.json"
    jinja_path = bundle / "chat_template.jinja"
    if not tc_path.exists():
        raise SystemExit(f"{bundle}: no tokenizer_config.json")

    before = _render_probe(bundle)

    tc = json.loads(tc_path.read_text())
    changed = []

    if tc.get("tool_parser_type") != PARSER:
        tc["tool_parser_type"] = PARSER
        changed.append("tool_parser_type")

    if isinstance(tc.get("chat_template"), str):
        new = _add_hint(tc["chat_template"])
        if new != tc["chat_template"]:
            tc["chat_template"] = new
            changed.append("embedded chat_template")

    tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n")

    if jinja_path.exists():
        t = jinja_path.read_text()
        new = _add_hint(t)
        if new != t:
            jinja_path.write_text(new)
            changed.append("chat_template.jinja")

    after = _render_probe(bundle)
    if before != after:
        raise SystemExit(
            f"{bundle}: ABORT — rendering changed after adding the hint. "
            f"The comment must be render-neutral."
        )

    # Confirm mlx_lm now resolves the parser.
    from mlx_lm.tokenizer_utils import _infer_tool_parser
    tmpl = (jinja_path.read_text() if jinja_path.exists()
            else tc.get("chat_template") or "")
    return {
        "bundle": bundle.name,
        "changed": changed or ["(already correct)"],
        "inferred": _infer_tool_parser(tmpl),
        "explicit": json.loads(tc_path.read_text()).get("tool_parser_type"),
        "render_identical": True,
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    for d in argv[1:]:
        r = fix(Path(d).expanduser())
        print(f"  {r['bundle']:34s} inferred={r['inferred']:9s} "
              f"explicit={r['explicit']:9s} render_identical={r['render_identical']} "
              f"changed={', '.join(r['changed'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
