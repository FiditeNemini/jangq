#!/usr/bin/env python3
"""Hy3 reasoning/tool parser contract.

Small parser reference for future `../vmlx` work. It avoids model loading and
focuses only on Hy3's text surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALLS_START = "<tool_calls>"
TOOL_CALLS_END = "</tool_calls>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_SEP = "<tool_sep>"
ARG_KEY_START = "<arg_key>"
ARG_KEY_END = "</arg_key>"
ARG_VALUE_START = "<arg_value>"
ARG_VALUE_END = "</arg_value>"


@dataclass
class ParsedHy3Output:
    content: str
    reasoning_content: str | None
    tool_calls: list[dict[str, Any]]


def _take_between(text: str, start: str, end: str, offset: int = 0) -> tuple[str | None, int]:
    i = text.find(start, offset)
    if i < 0:
        return None, offset
    j = text.find(end, i + len(start))
    if j < 0:
        return None, offset
    return text[i + len(start):j], j + len(end)


def extract_reasoning(text: str) -> tuple[str, str | None]:
    reasoning, end = _take_between(text, THINK_START, THINK_END)
    if reasoning is None:
        return text, None
    without = text[: text.find(THINK_START)] + text[end:]
    return without, reasoning


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    block, end = _take_between(text, TOOL_CALLS_START, TOOL_CALLS_END)
    if block is None:
        return text, []
    tool_calls: list[dict[str, Any]] = []
    offset = 0
    while True:
        raw, offset2 = _take_between(block, TOOL_CALL_START, TOOL_CALL_END, offset)
        if raw is None:
            break
        offset = offset2
        if TOOL_SEP not in raw:
            continue
        name, args_blob = raw.split(TOOL_SEP, 1)
        args: dict[str, Any] = {}
        arg_offset = 0
        while True:
            key, key_end = _take_between(args_blob, ARG_KEY_START, ARG_KEY_END, arg_offset)
            if key is None:
                break
            value, value_end = _take_between(args_blob, ARG_VALUE_START, ARG_VALUE_END, key_end)
            if value is None:
                break
            arg_offset = value_end
            args[key.strip()] = _coerce_value(value.strip())
        tool_calls.append({"type": "function", "function": {"name": name.strip(), "arguments": args}})
    without = text[: text.find(TOOL_CALLS_START)] + text[end:]
    return without, tool_calls


def _coerce_value(value: str) -> Any:
    if not value:
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_hy3_output(text: str) -> ParsedHy3Output:
    text, reasoning = extract_reasoning(text)
    text, tool_calls = parse_tool_calls(text)
    return ParsedHy3Output(content=text.strip(), reasoning_content=reasoning, tool_calls=tool_calls)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", default="<think></think>Hello")
    args = ap.parse_args()
    parsed = parse_hy3_output(args.text)
    print(json.dumps(parsed.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

