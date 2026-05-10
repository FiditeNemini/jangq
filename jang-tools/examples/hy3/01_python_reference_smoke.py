"""Smoke client for a Hy3-preview OpenAI-compatible reference server.

Use after a vLLM or SGLang server is already running. This script does not load
the 295B model locally.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://127.0.0.1:8010"
DEFAULT_MODEL = "hy3-preview"


TESTS = [
    {
        "name": "direct_math",
        "messages": [{"role": "user", "content": "What is 2+2? Answer with only the number."}],
        "needles": ["4"],
        "reasoning_effort": "no_think",
        "max_tokens": 16,
    },
    {
        "name": "low_reasoning",
        "messages": [{"role": "user", "content": "If x + 5 = 12, what is x? Answer briefly."}],
        "needles": ["7"],
        "reasoning_effort": "low",
        "max_tokens": 96,
    },
    {
        "name": "tool_parser_surface",
        "messages": [{"role": "user", "content": "Return a JSON object with key answer and value Paris."}],
        "needles": ["paris"],
        "reasoning_effort": "no_think",
        "max_tokens": 96,
    },
]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    results = []
    started = time.time()

    for test in TESTS:
        payload = {
            "model": args.model,
            "messages": test["messages"],
            "temperature": 0.0,
            "max_tokens": test["max_tokens"],
            "extra_body": {
                "chat_template_kwargs": {
                    "reasoning_effort": test["reasoning_effort"],
                }
            },
        }
        try:
            response = post_json(endpoint, payload, timeout=args.timeout)
            text = extract_text(response)
            passed = all(needle in text.lower() for needle in test["needles"])
            error = None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            response = {}
            text = ""
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "name": test["name"],
                "passed": passed,
                "reasoning_effort": test["reasoning_effort"],
                "needles": test["needles"],
                "text": text,
                "error": error,
                "usage": response.get("usage") if isinstance(response, dict) else None,
            }
        )

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "passed": all(item["passed"] for item in results),
        "duration_sec": round(time.time() - started, 3),
        "tests": results,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise
