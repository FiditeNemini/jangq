"""03 — DSML tool-call parsing demo.

DeepSeek-V4-Flash uses DSML format: ｜DSML｜ markers (fullwidth pipes,
NOT ASCII pipes). Tool calls look like:

    <｜DSML｜invoke name="get_weather">
      <｜DSML｜parameter name="location" string="true">San Francisco</｜DSML｜parameter>
      <｜DSML｜parameter name="units" string="false">"celsius"</｜DSML｜parameter>
    </｜DSML｜invoke>

This script:
  1. Asks the model with a tool-augmented system prompt
  2. Parses the response with parse_dsml_tool_calls
  3. Verifies parsed names + arguments

Run: python3 03_tool_calling.py [bundle_path]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"

TOOLS_SYSTEM = """You have access to these tools. Call them using DSML format.

Tool: get_weather
  description: Returns the current weather at a location.
  parameters:
    - location (string, required): city or place name
    - units (string, optional): 'celsius' or 'fahrenheit'

When you decide to call a tool, emit:
<｜DSML｜invoke name="get_weather">
  <｜DSML｜parameter name="location" string="true">CITY HERE</｜DSML｜parameter>
</｜DSML｜invoke>"""

USER_PROMPT = "What's the weather in Tokyo today?"


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    os.environ.setdefault("DSV4_LONG_CTX", "1")
    import mlx.core as mx
    mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)

    from jang_tools.load_jangtq import load_jangtq_model
    from jang_tools.dsv4.runtime import generate, GenerateOptions
    from jang_tools.dsv4.test_chat import parse_dsml_tool_calls

    print(f"=== Loading DSV4-Flash from {bundle.name} ===", flush=True)
    t0 = time.time()
    model, tok = load_jangtq_model(str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s\n", flush=True)

    msgs = [
        {"role": "system",  "content": TOOLS_SYSTEM},
        {"role": "user",    "content": USER_PROMPT},
    ]
    print(f"USER: {USER_PROMPT}\n")
    t1 = time.time()
    res = generate(model, tok, str(bundle), messages=msgs,
                   opts=GenerateOptions(mode="chat", max_tokens=256))
    dt = time.time() - t1

    print(f"RAW OUTPUT ({len(res.content)} chars):")
    print(f"  {res.content!r}\n")

    calls = parse_dsml_tool_calls(res.content)
    print(f"PARSED {len(calls)} tool call(s):")
    for c in calls:
        print(f"  • {c['name']}({json.dumps(c['arguments'], ensure_ascii=False)})")

    print(f"\nt={dt:.1f}s tokens={res.n_tokens} finish={res.finish_reason}")

    if not calls:
        print("\nWARN — model didn't emit a tool call. May need a stronger system prompt.")
    elif calls[0]["name"] == "get_weather" and "Tokyo" in str(calls[0]["arguments"].get("location","")):
        print("\nPASS — tool call parsed and matches request")
    else:
        print("\nWARN — tool call parsed but unexpected name/args")


if __name__ == "__main__":
    main()
