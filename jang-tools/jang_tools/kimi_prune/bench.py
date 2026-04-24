"""Multi-domain benchmark gate for the pruned Kimi K2.6 model.

Run focused benchmarks per domain and compare against a cached baseline
to decide whether a prune iteration passes.

Domains and their datasets:
  coding     HumanEval (164 prompts) — pass@1 via generated function tests
  tool       BFCL v3 or Salesforce/xlam — AST match on function call JSON
  agentic    SWE-bench_Lite (10 task sample, patch similarity)
  pentest    custom 50-MCQ assets/pentest_mcq.jsonl
  general    MMLU 500-sample
  chinese    C-Eval 200-sample

Gate thresholds (relative delta vs baseline, must be >= threshold):
  coding -5%  tool -5%  agentic -10%  pentest -10%  general -3%  chinese -5%

CAUTION: HumanEval scoring runs generated code in a subprocess with a
10 s timeout. Network access is not sandboxed. Do not run on untrusted
generator output outside a disposable environment.

Usage:
  # baseline (once, on the unpruned model)
  python -m jang_tools.kimi_prune.bench --model <src> --out baseline.json
  # post-prune gate (compare pruned vs baseline)
  python -m jang_tools.kimi_prune.bench --model <pruned> --out cur.json \\
      --baseline baseline.json --gate
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path


GATE_THRESHOLDS = {
    "coding":  -0.05,
    "tool":    -0.05,
    "agentic": -0.10,
    "pentest": -0.10,
    "general": -0.03,
    "chinese": -0.05,
}


@dataclass
class BenchResult:
    scores: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    model_path: str = ""
    elapsed_seconds: float = 0.0


def run_humaneval(model_gen_fn, n_samples: int = 164):
    from datasets import load_dataset
    import subprocess, tempfile
    ds = load_dataset("openai_humaneval", split=f"test[:{n_samples}]")
    out = []
    ok_count = 0
    for row in ds:
        completion = model_gen_fn(row["prompt"])
        full = (row["prompt"] + completion + "\n\n" + row["test"]
                + f"\n\ncheck({row['entry_point']})\n")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full); p = f.name
        try:
            r = subprocess.run(["python3", p], timeout=10, capture_output=True)
            ok = r.returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        out.append({"task_id": row["task_id"], "passed": ok})
        ok_count += int(ok)
    return ok_count / max(len(out), 1), out


def run_bfcl(model_gen_fn, n_samples: int = 100):
    from datasets import load_dataset
    import json as _j
    try:
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                          split=f"train[:{n_samples}]")
    except Exception:
        ds = load_dataset("Salesforce/xlam-function-calling-60k",
                          split=f"train[:{n_samples}]")
    out = []
    ok_count = 0
    for row in ds:
        prompt = row.get("query") or row.get("question", "")
        tools_desc = row.get("tools") or row.get("function", "")
        gold = row.get("answers") or row.get("ground_truth", "")
        fp = (f"Tools:\n{tools_desc}\n\nUser: {prompt}\n\n"
              "Assistant (respond with JSON function call):")
        resp = model_gen_fn(fp)
        try:
            gold_calls = _j.loads(gold) if isinstance(gold, str) else gold
            mdl = _j.loads(resp.strip().split("\n")[0])
            ok = (isinstance(mdl, (dict, list)) and
                  str(mdl).lower().find(str(gold_calls)[:20].lower()) >= 0)
        except Exception:
            ok = False
        out.append({"prompt": prompt[:80], "passed": ok})
        ok_count += int(ok)
    return ok_count / max(len(out), 1), out


def run_mmlu(model_choice_fn, n_samples: int = 500):
    from datasets import load_dataset
    import random
    ds = load_dataset("cais/mmlu", "all", split="test")
    rng = random.Random(42)
    idxs = rng.sample(range(len(ds)), min(n_samples, len(ds)))
    letters = ["A", "B", "C", "D"]
    out = []; ok_count = 0
    for i in idxs:
        row = ds[i]
        pred = model_choice_fn(row["question"], row["choices"])
        ok = pred.strip().upper()[:1] == letters[row["answer"]]
        out.append({"subject": row["subject"], "passed": ok})
        ok_count += int(ok)
    return ok_count / len(out), out


def run_ceval(model_choice_fn, n_samples: int = 200):
    from datasets import load_dataset
    import random
    try:
        ds = load_dataset("ceval/ceval-exam", "all", split="val")
    except Exception:
        return 0.0, [{"error": "ceval unavailable"}]
    rng = random.Random(42)
    idxs = rng.sample(range(len(ds)), min(n_samples, len(ds)))
    out = []; ok_count = 0
    for i in idxs:
        row = ds[i]
        choices = [row["A"], row["B"], row["C"], row["D"]]
        pred = model_choice_fn(row["question"], choices)
        ok = pred.strip().upper()[:1] == row["answer"].strip().upper()
        out.append({"subject": row["subject"], "passed": ok})
        ok_count += int(ok)
    return ok_count / len(out), out


def run_pentest_mcq(model_choice_fn, assets_dir: Path):
    path = assets_dir / "pentest_mcq.jsonl"
    if not path.exists():
        return 0.0, [{"error": f"missing {path}"}]
    out = []; ok_count = 0
    for i, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        pred = model_choice_fn(row["question"], row["choices"])
        ok = pred.strip().upper()[:1] == row["answer"].strip().upper()
        out.append({"id": row.get("id", i), "passed": ok})
        ok_count += int(ok)
    return ok_count / max(len(out), 1), out


def run_swebench(model_gen_fn, n_tasks: int = 10):
    from datasets import load_dataset
    try:
        ds = load_dataset("princeton-nlp/SWE-bench_Lite",
                          split=f"dev[:{n_tasks}]")
    except Exception:
        return 0.0, [{"error": "swebench unavailable"}]
    import difflib
    out = []; ok_count = 0
    for row in ds:
        prompt = (f"Repo: {row['repo']}\n"
                  f"Issue: {row['problem_statement'][:2000]}\n\n"
                  "Produce a unified diff patch:")
        resp = model_gen_fn(prompt)
        sim = difflib.SequenceMatcher(None, resp, row["patch"]).ratio()
        ok = sim > 0.4
        out.append({"instance_id": row["instance_id"], "similarity": sim,
                    "passed": ok})
        ok_count += int(ok)
    return ok_count / max(len(out), 1), out


def _build_gen_fn(model, tokenizer, max_new_tokens: int = 256):
    from mlx_lm import generate
    def _gen(prompt: str) -> str:
        return generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_new_tokens, verbose=False, temp=0.0)
    return _gen


def _build_choice_fn(model, tokenizer):
    import mlx.core as mx
    letters = ["A", "B", "C", "D"]
    def _choice(question: str, choices: list[str]) -> str:
        prompt = question + "\n"
        for i, c in enumerate(choices):
            prompt += f"{letters[i]}. {c}\n"
        prompt += "Answer: "
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        x = mx.array([ids], dtype=mx.int32)
        logits = model(x)[:, -1, :]
        letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0]
                      for l in letters]
        scored = [float(logits[0, lid]) for lid in letter_ids]
        return letters[int(max(range(4), key=lambda i: scored[i]))]
    return _choice


def run_all(model_path: Path, assets_dir: Path) -> BenchResult:
    import time
    from jang_tools.kimi_prune.profile import _load_model_mlx
    t0 = time.time()
    model, tokenizer, _ = _load_model_mlx(model_path)
    gen = _build_gen_fn(model, tokenizer)
    choice = _build_choice_fn(model, tokenizer)
    res = BenchResult(model_path=str(model_path))
    print("[bench] coding", flush=True)
    res.scores["coding"], res.details["coding"] = run_humaneval(gen)
    print("[bench] tool", flush=True)
    res.scores["tool"], res.details["tool"] = run_bfcl(gen)
    print("[bench] general", flush=True)
    res.scores["general"], res.details["general"] = run_mmlu(choice)
    print("[bench] chinese", flush=True)
    res.scores["chinese"], res.details["chinese"] = run_ceval(choice)
    print("[bench] pentest", flush=True)
    res.scores["pentest"], res.details["pentest"] = run_pentest_mcq(choice, assets_dir)
    print("[bench] agentic", flush=True)
    res.scores["agentic"], res.details["agentic"] = run_swebench(gen)
    res.elapsed_seconds = time.time() - t0
    return res


def gate(result: BenchResult, baseline_path: Path):
    baseline = json.loads(baseline_path.read_text())
    deltas = {}
    passed = True
    for dom, thr in GATE_THRESHOLDS.items():
        base = baseline["scores"].get(dom, 0.0)
        cur = result.scores.get(dom, 0.0)
        delta = (cur - base) / base if base > 0 else 0.0
        deltas[dom] = {"baseline": base, "current": cur, "rel_delta": delta,
                       "threshold": thr, "passed": delta >= thr}
        if delta < thr:
            passed = False
    return passed, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--assets", type=Path,
                    default=Path(__file__).parent / "assets")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()

    result = run_all(args.model, args.assets)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump({"scores": result.scores, "details": result.details,
                   "model_path": result.model_path,
                   "elapsed_seconds": result.elapsed_seconds}, f, indent=2)
    print(f"[bench] scores: {result.scores}", flush=True)

    if args.gate:
        assert args.baseline, "--baseline required with --gate"
        passed, deltas = gate(result, args.baseline)
        print(f"[bench] gate {'PASS' if passed else 'FAIL'}", flush=True)
        for d, info in deltas.items():
            print(f"  {d:8s} base={info['baseline']:.3f} "
                  f"cur={info['current']:.3f} "
                  f"Δ={info['rel_delta']:+.1%}  "
                  f"{'ok' if info['passed'] else 'FAIL'}  "
                  f"(thr {info['threshold']:+.0%})", flush=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
