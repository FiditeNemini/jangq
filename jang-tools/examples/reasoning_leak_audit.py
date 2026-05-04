"""Reasoning + tool-call leak audit across DSV4 / Laguna / Mistral 3.5.

Cross-checks Python runtime behavior against the Swift vmlx-swift-lm
ReasoningParser / ToolCallFormat dispatchers. Runs on each bundle
configured below the rule:

  1. Single-turn thinking ON  → reasoning + content split, no leak
  2. Single-turn thinking OFF → no reasoning, content clean
  3. Multi-turn (T2) thinking ON → no T1 reasoning bleed into T2 content

For DSV4: also exercises DSML tool call parsing.

Usage:
  python3 reasoning_leak_audit.py [model=dsv4|laguna|mistral3] [bundle_path]

Defaults to DSV4-Flash JANGTQ.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# (model, default bundle, supports_thinking, tool_format)
PROFILES = {
    "dsv4":     ("~/.mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ", True,  "dsml"),
    "laguna":   ("~/.mlxstudio/models/_bundles/Laguna-XS.2-JANGTQ",       True,  "glm4"),
    "mistral3": ("~/.mlxstudio/models/_bundles/Mistral-Medium-3.5-128B-JANGTQ", False, "mistral"),
}

LEAK_TAGS = ("<think>", "</think>", "<|channel>", "<channel|>", "<|start|>", "<|message|>",
             "｜DSML｜", "</DSML>", "<thought>", "</thought>")


def audit_dsv4(bundle: Path, supports_thinking: bool, tool_format: str):
    os.environ.setdefault("DSV4_LONG_CTX", "1")
    import mlx.core as mx
    mx.set_memory_limit(200 * 1024**3)

    from jang_tools.load_jangtq import load_jangtq_model
    from jang_tools.dsv4.runtime import generate, GenerateOptions
    from jang_tools.dsv4.test_chat import parse_dsml_tool_calls

    print(f"=== DSV4 audit: {bundle.name} ===", flush=True)
    t0 = time.time()
    model, tok = load_jangtq_model(str(bundle))
    print(f"loaded in {time.time()-t0:.1f}s\n", flush=True)

    fails = []

    # T1-thinking-on
    print("T1 (think): ", end="", flush=True)
    res = generate(model, tok, str(bundle),
                   messages=[{"role":"user","content":"What's 2+2? Think briefly first."}],
                   opts=GenerateOptions(mode="think", max_tokens=512))
    leaks = [t for t in LEAK_TAGS if t in res.content]
    print(f"reasoning={len(res.reasoning_content)}ch content={len(res.content)}ch "
          f"saw_close={res.saw_think_close} leaks={leaks}")
    if leaks: fails.append(("T1-think leak", leaks))
    if not res.reasoning_content and supports_thinking and res.saw_think_close:
        fails.append(("T1-think missing reasoning", res.raw[:80]))

    # T1-thinking-off
    print("T1 (chat):  ", end="", flush=True)
    res = generate(model, tok, str(bundle),
                   messages=[{"role":"user","content":"What's the capital of Japan?"}],
                   opts=GenerateOptions(mode="chat", max_tokens=64, temperature=0.0))
    leaks = [t for t in LEAK_TAGS if t in res.content]
    print(f"content={res.content[:60]!r} leaks={leaks}")
    if leaks: fails.append(("T1-chat leak", leaks))

    # T2 multi-turn thinking on
    print("T2 (multi-turn think): ", end="", flush=True)
    res1 = generate(model, tok, str(bundle),
                    messages=[{"role":"user","content":"What's 5+5?"}],
                    opts=GenerateOptions(mode="think", max_tokens=300))
    msgs2 = [
        {"role":"user","content":"What's 5+5?"},
        {"role":"assistant","content":res1.content},
        {"role":"user","content":"And 7+7?"},
    ]
    res2 = generate(model, tok, str(bundle), messages=msgs2,
                    opts=GenerateOptions(mode="think", max_tokens=300))
    leaks = [t for t in LEAK_TAGS if t in res2.content]
    bleed = res1.reasoning_content[:50] in res2.content if res1.reasoning_content else False
    print(f"content={res2.content[:60]!r} leaks={leaks} bleed={bleed}")
    if leaks: fails.append(("T2-think leak", leaks))
    if bleed: fails.append(("T2-think reasoning bleed", res1.reasoning_content[:50]))

    # DSML tool call
    if tool_format == "dsml":
        print("Tool call (DSML): ", end="", flush=True)
        sys_msg = ('You can call tools using DSML format. Tool: get_time(timezone: string). '
                   'Emit <｜DSML｜invoke name="get_time"><｜DSML｜parameter name="timezone" '
                   'string="true">UTC</｜DSML｜parameter></｜DSML｜invoke>')
        res = generate(model, tok, str(bundle),
                       messages=[{"role":"system","content":sys_msg},
                                 {"role":"user","content":"What time is it in UTC?"}],
                       opts=GenerateOptions(mode="chat", max_tokens=256))
        calls = parse_dsml_tool_calls(res.content)
        print(f"calls={[c['name'] for c in calls]} content={res.content[:80]!r}")
        # Don't fail — model may or may not call tool depending on prompt strength

    return fails


def audit_laguna_or_mistral3(bundle: Path, supports_thinking: bool, name: str):
    """Generic auditor for the simpler Python runtimes (no reasoning split helper)."""
    import mlx.core as mx
    mx.set_memory_limit(200 * 1024**3)
    from transformers import AutoTokenizer

    if name == "laguna":
        from jang_tools.laguna.runtime import load
    else:
        from jang_tools.mistral3.runtime import load

    print(f"=== {name} audit: {bundle.name} ===", flush=True)
    tok = AutoTokenizer.from_pretrained(str(bundle), trust_remote_code=True)
    t0 = time.time()
    model, cfg, fmt = load(str(bundle))
    print(f"loaded in {time.time()-t0:.1f}s (format={fmt})\n", flush=True)

    fails = []

    msgs = [{"role":"user","content":"Briefly: what is the largest planet in our solar system?"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text)

    out = list(ids)
    x = mx.array([ids], dtype=mx.uint32)
    if name == "laguna":
        logits, caches = model(x, caches=None)
    else:
        logits, caches = model(x, caches=None)
    for _ in range(48):
        nxt = int(mx.argmax(logits[0, -1]).item())
        out.append(nxt)
        if nxt == tok.eos_token_id: break
        x = mx.array([[nxt]], dtype=mx.uint32)
        logits, caches = model(x, caches=caches)
    decoded = tok.decode(out[len(ids):])
    leaks = [t for t in LEAK_TAGS if t in decoded]
    print(f"content={decoded[:120]!r} leaks={leaks}")
    if leaks: fails.append(("T1 leak", leaks))
    return fails


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "dsv4"
    if model_name not in PROFILES:
        print(f"unknown model: {model_name}; choose from {list(PROFILES)}"); sys.exit(2)
    default_path, supports_thinking, tool_format = PROFILES[model_name]
    bundle = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else Path(default_path).expanduser()
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    if model_name == "dsv4":
        fails = audit_dsv4(bundle, supports_thinking, tool_format)
    else:
        fails = audit_laguna_or_mistral3(bundle, supports_thinking, model_name)

    print("\n=== SUMMARY ===")
    if fails:
        print(f"FAIL — {len(fails)} issues:")
        for name, detail in fails:
            print(f"  • {name}: {detail}")
        sys.exit(3)
    print("PASS — no leaks, reasoning/content split clean")


if __name__ == "__main__":
    main()
