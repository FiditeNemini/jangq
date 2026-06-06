#!/usr/bin/env python3
"""Run a native-vs-manual-vs-sink-off MiMo generation probe.

This is intentionally a harness, not a fix. It keeps all generation inputs the
same and only toggles MiMo sink runtime environment between child processes so
the model is unloaded between variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Answer in concise English. Write three bullet points explaining what a "
    "prefix cache does for a local language model server."
)


def _cjk_count(text: str) -> int:
    return sum(
        1
        for ch in text
        if ("\u4e00" <= ch <= "\u9fff")
        or ("\u3400" <= ch <= "\u4dbf")
        or ("\u3040" <= ch <= "\u30ff")
        or ("\uac00" <= ch <= "\ud7af")
    )


def _child(args: argparse.Namespace) -> int:
    from mlx_lm import generate, load
    from jang_tools.mimo_v2 import mlx_register  # noqa: F401

    model, tokenizer = load(
        args.model,
        lazy=args.lazy,
        tokenizer_config={"trust_remote_code": True},
        model_config={"trust_remote_code": True},
    )
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    prompt_tokens = tokenizer.encode(prompt)
    started = time.time()
    generate_kwargs = {"max_tokens": args.max_tokens, "verbose": False}
    if args.temp > 0:
        from mlx_lm.sample_utils import make_sampler

        generate_kwargs["sampler"] = make_sampler(temp=args.temp)
    text = generate(model, tokenizer, prompt, **generate_kwargs)
    payload = {
        "mode": args.mode,
        "sink_disabled": os.environ.get("JANG_MIMO_DISABLE_SINK") in {"1", "true", "yes"},
        "elapsed_sec": round(time.time() - started, 3),
        "cjk_count": _cjk_count(text),
        "prompt_tokens": len(prompt_tokens),
        "prompt": args.prompt,
        "output": text,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _run_variant(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    env = os.environ.copy()
    env.pop("JANG_MIMO_DISABLE_SINK", None)
    env.pop("JANG_MIMO_MANUAL_SINK_SDPA", None)
    if mode == "sink_disabled":
        env["JANG_MIMO_DISABLE_SINK"] = "1"
    elif mode == "manual_sink_sdpa":
        env["JANG_MIMO_MANUAL_SINK_SDPA"] = "1"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--mode",
        mode,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--temp",
        str(args.temp),
    ]
    if args.lazy:
        cmd.append("--lazy")
    if args.enable_thinking:
        cmd.append("--enable-thinking")
    proc = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "mode": mode,
            "sink_disabled": mode == "sink_disabled",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    result = None
    for line in reversed([line for line in proc.stdout.splitlines() if line.strip()]):
        try:
            result = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if result is None:
        return {
            "mode": mode,
            "sink_disabled": mode == "sink_disabled",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "parse_error": "no_json_result_line",
        }
    if proc.stderr.strip():
        result["stderr"] = proc.stderr
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--lazy", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", default="normal")
    args = parser.parse_args(argv)

    if args.child:
        return _child(args)

    artifact = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temp": args.temp,
        "enable_thinking": args.enable_thinking,
        "variants": [
            _run_variant(args, "native_sink_sdpa"),
            _run_variant(args, "manual_sink_sdpa"),
            _run_variant(args, "sink_disabled"),
        ],
    }
    text = json.dumps(artifact, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
