"""Run the ZAYA coherence gate against a real OpenAI-compatible server.

This script does not load ZAYA itself. ZAYA requires Zyphra's custom runtime
branch today, so the honest gate is: launch that runtime, call it through the
OpenAI chat-completions API, and write a JSON report before upload.

Example server:
    vllm serve /Users/eric/jang/models/Zyphra/ZAYA1-8B --port 8010 \
      --mamba-cache-dtype float32 --dtype bfloat16 \
      --reasoning-parser qwen3 --enable-auto-tool-choice \
      --tool-call-parser zaya_xml

Example gate:
    python3 03_coherence_gate.py \
      --server http://127.0.0.1:8010 \
      --model /Users/eric/jang/models/Zyphra/ZAYA1-8B \
      --output /Users/eric/jang/models/Zyphra/zaya_coherence_report.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SERVER = "http://127.0.0.1:8010"
DEFAULT_MODEL = "/Users/eric/jang/models/Zyphra/ZAYA1-8B"
DEFAULT_OUTPUT = "/Users/eric/jang/models/Zyphra/zaya_coherence_report.json"


TESTS = [
    {
        "name": "arithmetic_2_plus_2",
        "messages": [
            {"role": "system", "content": "Answer directly and briefly."},
            {"role": "user", "content": "What is 2 + 2? Answer with only the number."},
        ],
        "needles": ["4"],
        "max_tokens": 16,
    },
    {
        "name": "capital_france",
        "messages": [
            {"role": "system", "content": "Answer directly and briefly."},
            {"role": "user", "content": "What is the capital of France? Answer in one word."},
        ],
        "needles": ["paris"],
        "max_tokens": 16,
    },
    {
        "name": "simple_python",
        "messages": [
            {"role": "system", "content": "Answer with code only."},
            {"role": "user", "content": "Write a Python function add(a, b) that returns their sum."},
        ],
        "needles": ["def", "return"],
        "max_tokens": 96,
    },
]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
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
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    endpoint = args.server.rstrip("/") + "/v1/chat/completions"
    results: list[dict[str, Any]] = []
    started = time.time()

    for test in TESTS:
        payload = {
            "model": args.model,
            "messages": test["messages"],
            "temperature": args.temperature,
            "max_tokens": test["max_tokens"],
        }
        try:
            response = post_json(endpoint, payload, timeout=args.timeout)
            text = extract_text(response)
            lowered = text.lower()
            passed = all(needle in lowered for needle in test["needles"])
            error = None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            response = {}
            text = ""
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        results.append(
            {
                "name": test["name"],
                "passed": passed,
                "needles": test["needles"],
                "text": text,
                "error": error,
                "usage": response.get("usage") if isinstance(response, dict) else None,
            }
        )

    report = {
        "model": args.model,
        "server": args.server,
        "endpoint": endpoint,
        "passed": all(item["passed"] for item in results),
        "duration_sec": round(time.time() - started, 3),
        "tests": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
