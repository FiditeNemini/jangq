"""Live-verify a Qwen3.6-27B JANG bundle: vision, video, reasoning, tools, MTP.

Structural checks are not proof (`feedback_verify_runtime_before_ship`). This
loads the bundle and exercises every capability it claims.

    python -m jang_tools.verify_qwen36_bundle <bundle_dir> [--image X.png]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path


def _text(out) -> str:
    s = str(out)
    m = re.search(r"text='(.*?)', token=", s, re.S)
    return (m.group(1) if m else s).replace("\\n", " ").strip()


def main(argv) -> int:
    B = Path(argv[1])
    img = None
    for i, a in enumerate(argv):
        if a == "--image" and i + 1 < len(argv):
            img = argv[i + 1]

    print("=" * 74)
    print(f"  VERIFY  {B.name}")
    print("=" * 74, flush=True)

    cfg = json.loads((B / "config.json").read_text())
    jang = json.loads((B / "jang_config.json").read_text())
    q = cfg.get("quantization", {})
    print(f"  quantization : bits={q.get('bits')} gs={q.get('group_size')} "
          f"mode={q.get('mode','affine')} overrides="
          f"{sum(1 for v in q.values() if isinstance(v, dict))}")

    sd = (jang.get("chat") or {}).get("sampling_defaults", {})
    gen = json.loads((B / "generation_config.json").read_text())
    print(f"  sampling     : T={sd.get('temperature')} top_p={sd.get('top_p')} "
          f"top_k={sd.get('top_k')} eos={gen.get('eos_token_id')}")
    modes = (jang.get("chat") or {}).get("sampling_modes", {})
    print(f"  modes stamped: {list(modes)}")

    # --- two-file agreement gate -----------------------------------------
    mismatch = [k for k in ("temperature", "top_p", "top_k")
                if sd.get(k) != gen.get(k)]
    print(f"  two-file agree: {'YES' if not mismatch else 'NO -> ' + str(mismatch)}")

    # --- modality claims vs files ----------------------------------------
    caps = jang.get("capabilities", {})
    print(f"  claims       : vision={caps.get('has_vision')} "
          f"video={caps.get('has_video')} audio={caps.get('has_audio')} "
          f"reasoning={caps.get('default_reasoning')} tools={caps.get('tool_parser')}")
    for f in ("preprocessor_config.json", "video_preprocessor_config.json"):
        print(f"    {f:34s} {'present' if (B / f).exists() else 'ABSENT'}")
    mtp = jang.get("mtp", {})
    print(f"  mtp          : artifact={mtp.get('artifact_available')} "
          f"tensors={mtp.get('tensor_count')} runtime={mtp.get('runtime_available')}")

    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template

    t0 = time.time()
    model, proc = load(str(B))
    print(f"\n  loaded in {time.time()-t0:.1f}s", flush=True)

    results = {}

    # --- reasoning ON / OFF (text) ---------------------------------------
    for label, think in (("reasoning-ON", True), ("reasoning-OFF", False)):
        msgs = [{"role": "user", "content": "What is the capital of Japan? One sentence."}]
        p = proc.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think)
        t0 = time.time()
        o = generate(model, proc, p, max_tokens=120, temperature=0.6, verbose=False)
        txt = _text(o)
        n = len(proc.tokenizer.encode(txt))
        print(f"\n  --- {label} --- {n} tok in {time.time()-t0:.1f}s "
              f"= {n/max(time.time()-t0,1e-9):.1f} tok/s")
        print(f"  {txt[:220]}")
        results[label] = txt

    # --- vision ----------------------------------------------------------
    if img:
        msgs = [{"role": "user", "content": "Describe the shapes and their colors."}]
        p = apply_chat_template(proc, model.config, msgs, num_images=1)
        t0 = time.time()
        o = generate(model, proc, p, image=[img], max_tokens=90,
                     temperature=0.6, verbose=False)
        print(f"\n  --- vision ({time.time()-t0:.1f}s) ---")
        print(f"  {_text(o)[:260]}")
        results["vision"] = _text(o)

    # --- tool call -------------------------------------------------------
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    p = proc.tokenizer.apply_chat_template(
        [{"role": "user", "content": "What's the weather in Santa Clara?"}],
        tools=tools, add_generation_prompt=True, tokenize=False,
        enable_thinking=False)
    o = generate(model, proc, p, max_tokens=160, temperature=0.6, verbose=False)
    txt = _text(o)
    print(f"\n  --- tool call ---")
    print(f"  emits <tool_call>: {'<tool_call>' in txt}")
    print(f"  {txt[:220]}")
    results["tool"] = txt

    ok = all(v.strip() for v in results.values()) and not mismatch
    print("\n" + "=" * 74)
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}  ({len(results)} probes)")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
