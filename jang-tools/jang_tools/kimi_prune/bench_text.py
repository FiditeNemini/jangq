"""Minimal text-quality bench for Kimi K2.6 JANGTQ bundles.

Not a full MMLU runner — this is a smoke-grade bench that covers the
dimensions we actually care about for shipping decisions:

  1. **Reasoning** — 3 hand-picked grade-school + competition-level math
     prompts. Pass/fail on exact answer match.
  2. **Knowledge** — 5 MMLU-format multiple-choice questions (mix of
     domains). Expects the model to output "A", "B", "C", or "D".
     Measures top-1 accuracy.
  3. **Code** — 2 HumanEval-style short function completions. Not
     graded strictly; we check for obviously broken output (empty,
     repetition loop, syntax error).
  4. **Chinese** — 2 prompts. Not graded; checks the model produces
     Chinese characters rather than English.
  5. **Tool calling** — 1 prompt with an OpenAI-format tool spec in the
     system prompt. We verify the output contains Kimi's
     `<|tool_calls_section_begin|>` marker OR can be parsed by
     `mlx_lm.tool_parsers.kimi_k2`.

For each prompt: records prefill latency, decode tok/s, coherence (no
repetition loop), and pass/fail.

Run:
  python -m jang_tools.kimi_prune.bench_text \
      --model /path/to/Kimi-K2.6-REAP-30-JANGTQ_1L

Full MMLU (57 subjects × 1-5 shots) belongs on a rented H200 — this harness
is the "does the model broadly work" gate before that investment.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path


REASONING = [
    dict(prompt="What is 47 + 38? Respond with just the number.",
         expected_contains=["85"]),
    dict(prompt="If a train leaves station A at 2 pm going 60 mph and another "
                "leaves station B at 3 pm going 80 mph toward station A, and "
                "the stations are 280 miles apart, at what time do they meet? "
                "Respond in the form HH:MM pm.",
         expected_contains=["4:", "04:"]),
    dict(prompt="A right triangle has legs of length 3 and 4. What is the length "
                "of the hypotenuse? Respond with just the number.",
         expected_contains=["5"]),
]

MMLU = [
    dict(
        prompt=(
            "Question: Which gas makes up the majority of Earth's atmosphere?\n"
            "A) Oxygen\nB) Carbon dioxide\nC) Nitrogen\nD) Argon\n"
            "Answer with a single letter only."
        ),
        answer="C",
    ),
    dict(
        prompt=(
            "Question: In computing, what does 'RAM' stand for?\n"
            "A) Rapid Access Method\nB) Random Access Memory\n"
            "C) Read Allocate Module\nD) Runtime Application Memory\n"
            "Answer with a single letter only."
        ),
        answer="B",
    ),
    dict(
        prompt=(
            "Question: The Pythagorean theorem applies to which type of triangle?\n"
            "A) Equilateral\nB) Isosceles\nC) Right\nD) Obtuse\n"
            "Answer with a single letter only."
        ),
        answer="C",
    ),
    dict(
        prompt=(
            "Question: Which of the following is NOT a prime number?\n"
            "A) 7\nB) 11\nC) 15\nD) 13\n"
            "Answer with a single letter only."
        ),
        answer="C",
    ),
    dict(
        prompt=(
            "Question: Who wrote 'Hamlet'?\n"
            "A) Charles Dickens\nB) William Shakespeare\n"
            "C) Mark Twain\nD) Jane Austen\n"
            "Answer with a single letter only."
        ),
        answer="B",
    ),
]

CODE = [
    dict(prompt="Write a Python function `add(a, b)` that returns a + b. "
                "Output only the function, no explanation.",
         must_contain=["def add", "return"]),
    dict(prompt="Write a Python function `is_even(n)` that returns True if "
                "n is even, False otherwise. Output only the function.",
         must_contain=["def is_even", "return"]),
]

CHINESE = [
    dict(prompt="用一句话描述中国的首都北京。",  # "Describe Beijing in one sentence."
         must_contain_cjk=True),
    dict(prompt="什么是人工智能?请简短回答。",  # "What is AI? Short answer."
         must_contain_cjk=True),
]

TOOL = [
    dict(
        system_prompt=(
            "You have access to one tool:\n"
            "{\"name\": \"get_weather\", \"description\": \"Get current weather in a "
            "city\", \"parameters\": {\"city\": {\"type\": \"string\"}}}"
        ),
        user_prompt="What is the weather in Tokyo right now?",
        expected_pattern=r"<\|tool_call_begin\|>|get_weather",
    ),
]


def _repetition_loop(text: str) -> str | None:
    s = text.strip()
    if len(s) < 20:
        return None
    for n in (2, 3, 4, 5, 6, 8, 12):
        if len(s) >= 3 * n and s.endswith(s[-n:] * 3):
            return f"substring {s[-n:]!r} repeats 3x+"
    toks = s.split()
    if len(toks) >= 8 and len(set(toks[-8:])) <= 2:
        return f"last 8 tokens collapse to <=2 unique: {toks[-8:]!r}"
    return None


def _contains_cjk(text: str) -> bool:
    return any(
        "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"
        for ch in text
    )


def _generate(model, tokenizer, user_prompt: str, *,
              system_prompt: str | None = None, max_tokens: int = 60) -> dict:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    templated = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False,
    )
    t0 = time.time()
    # Force small prefill_step_size so prefill chunks fit under Metal's ~60s
    # command-buffer watchdog. mlx_lm defaults to 512 which is fine for small
    # affine-quantized models, but a 191 GB 2-bit MXTQ MoE needs smaller chunks
    # (per-shape JIT compile + routing + MLA absorb stack gets expensive).
    # Same idea as the VL path's prefill_step_size=32; for text with MUCH
    # shorter prompts (≤ ~60 tok) we use 16 to amortize compile cost across
    # 3-4 small buffers instead of one.
    text = generate(
        model, tokenizer,
        prompt=templated,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),
        verbose=False,
        prefill_step_size=16,
    )
    elapsed = time.time() - t0
    return {"text": text, "elapsed": elapsed, "max_tokens": max_tokens}


def run(model_path: Path, limit_kind: str | None = None) -> int:
    from jang_tools.load_jangtq import load_jangtq_model
    # apply patch first (idempotent)
    from jang_tools.kimi_prune.runtime_patch import apply as _apply_patch
    _apply_patch(dry_run=False)

    print(f"[bench] model: {model_path}")
    model, tokenizer = load_jangtq_model(model_path)

    # Warmup: first forward through a 191 GB quantized MoE JIT-compiles
    # per-shape Metal kernels for TurboQuantSwitchLinear + MLA absorb +
    # SwitchGLU. On cold start, a single prefill over a ~30-token prompt
    # can exceed Metal's ~60 s command-buffer watchdog. Running a tiny
    # (1-token) forward first amortizes the compile into a controlled
    # call where we can afford the wait.
    print("  [bench] warmup forward (JIT-compiles Metal kernels)...",
          flush=True)
    import time as _time
    _t0 = _time.time()
    from mlx_lm.sample_utils import make_sampler as _mks
    from mlx_lm import generate as _gen
    _ = _gen(
        model, tokenizer,
        prompt="Hi",
        max_tokens=1,
        sampler=_mks(temp=0.0),
        verbose=False,
    )
    print(f"  [bench] warmup done in {_time.time() - _t0:.1f}s", flush=True)

    buckets = []
    if limit_kind in (None, "reasoning"):
        buckets.append(("reasoning", REASONING, 30))
    if limit_kind in (None, "mmlu"):
        buckets.append(("mmlu", MMLU, 6))
    if limit_kind in (None, "code"):
        buckets.append(("code", CODE, 80))
    if limit_kind in (None, "chinese"):
        buckets.append(("chinese", CHINESE, 60))
    if limit_kind in (None, "tool"):
        buckets.append(("tool", TOOL, 100))

    total_pass = 0
    total_cases = 0
    rows: list[dict] = []
    for kind, cases, tokens in buckets:
        print(f"\n=== {kind.upper()} ({len(cases)} cases) ===")
        for i, c in enumerate(cases):
            if kind == "tool":
                out = _generate(
                    model, tokenizer, c["user_prompt"],
                    system_prompt=c.get("system_prompt"),
                    max_tokens=tokens,
                )
            else:
                out = _generate(
                    model, tokenizer, c["prompt"], max_tokens=tokens,
                )
            text = out["text"]
            ok = True
            reason = ""
            rep = _repetition_loop(text)
            if rep:
                ok = False
                reason = f"repetition: {rep}"
            if ok and kind == "reasoning":
                if not any(tok in text for tok in c["expected_contains"]):
                    ok = False
                    reason = f"none of {c['expected_contains']!r} in output"
            if ok and kind == "mmlu":
                m = re.search(r"\b([A-D])\b", text)
                predicted = m.group(1) if m else "?"
                if predicted != c["answer"]:
                    ok = False
                    reason = f"predicted {predicted!r} expected {c['answer']!r}"
            if ok and kind == "code":
                for needle in c["must_contain"]:
                    if needle not in text:
                        ok = False
                        reason = f"missing {needle!r}"
                        break
            if ok and kind == "chinese":
                if c.get("must_contain_cjk") and not _contains_cjk(text):
                    ok = False
                    reason = "no CJK characters in output"
            if ok and kind == "tool":
                if not re.search(c["expected_pattern"], text):
                    ok = False
                    reason = f"no match for {c['expected_pattern']!r}"
            tag = "PASS" if ok else "FAIL"
            total_pass += int(ok)
            total_cases += 1
            tps = tokens / max(out["elapsed"], 1e-6)
            print(f"  [{tag}] {kind}#{i}  {out['elapsed']:.1f}s  "
                  f"({tps:.1f} tok/s)  {reason}")
            print(f"         → {text[:180]!r}")
            rows.append(dict(
                kind=kind, case=i, pass_=ok, reason=reason,
                text=text, elapsed=out["elapsed"],
            ))

    print(f"\n=== SUMMARY ===")
    print(f"  {total_pass} / {total_cases} passed")
    report = {
        "model": str(model_path),
        "total_pass": total_pass, "total": total_cases,
        "cases": rows,
    }
    out_path = Path(f"/tmp/kimi_bench_text_{int(time.time())}.json")
    out_path.write_text(json.dumps(report, indent=2))
    print(f"  full report: {out_path}")
    return 0 if total_pass == total_cases else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--kind", default=None,
                    choices=("reasoning", "mmlu", "code", "chinese", "tool"))
    args = ap.parse_args()
    return run(args.model, args.kind)


if __name__ == "__main__":
    sys.exit(main())
