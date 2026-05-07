"""Smoke client for the official Zyphra ZAYA vLLM runtime.

Start the server with the Zyphra vLLM and transformers branches:

    vllm serve /Users/eric/jang/models/Zyphra/ZAYA1-8B --port 8010 \
      --mamba-cache-dtype float32 --dtype bfloat16 \
      --reasoning-parser qwen3 --enable-auto-tool-choice \
      --tool-call-parser zaya_xml

Do not enable prefix caching for this model. The official vLLM ZAYA class
asserts it off because CCA inner state has to be restored with KV.

Run:
    python3 01_python_vllm_smoke.py --base-url http://127.0.0.1:8010
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


DEFAULT_MODEL = "/Users/eric/jang/models/Zyphra/ZAYA1-8B"


def post_json(base_url: str, path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e


def chat_once(base_url: str, model: str, prompt: str, enable_thinking: bool) -> tuple[str, float, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 256,
        "temperature": 0.2,
        "extra_body": {"enable_thinking": enable_thinking},
    }
    t0 = time.time()
    result = post_json(base_url, "/v1/chat/completions", payload)
    return prompt, time.time() - t0, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    prompts = [
        ("What is 17 + 28? Answer with only the number.", False),
        ("Give one sentence on why prefix cache restore is risky for CCA models.", False),
        ("Solve: if x + 5 = 12, what is x?", True),
    ]

    print(f"base_url={args.base_url}")
    print(f"model={args.model}")
    print("issuing concurrent chat requests")

    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [
            pool.submit(chat_once, args.base_url, args.model, prompt, thinking)
            for prompt, thinking in prompts
        ]
        for fut in as_completed(futures):
            prompt, dt, result = fut.result()
            choice = result["choices"][0]
            msg = choice.get("message", {})
            usage = result.get("usage", {})
            print("\n---")
            print(f"prompt: {prompt}")
            print(f"finish_reason: {choice.get('finish_reason')}")
            print(f"usage: {usage}")
            print(f"latency_s: {dt:.2f}")
            print(msg.get("content", ""))

    print("\nsmoke complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
