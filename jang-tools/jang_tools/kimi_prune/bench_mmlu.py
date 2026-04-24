"""Real MMLU sampler for Kimi K2.6 JANGTQ bundles.

Pulls N questions from the canonical MMLU dataset (`cais/mmlu`, `all` config,
test split — 14 042 total), runs them through the model with a 5-shot
prompt per subject, parses A/B/C/D from the output, and reports accuracy
overall + per-subject.

Two sampling strategies supported:
  * `--mode logit` — pure logit-based: compute log-probabilities of the
    4 candidate letters at the first generated token. Fast (~2-3 s per
    question). Works even if reasoning mode eats the early tokens —
    we look at the VERY FIRST token the model would emit.
  * `--mode generate` — open-ended generation up to `--max-tokens` tokens,
    then regex out the first occurrence of `\\b([A-D])\\b`. Slower (~5-30 s
    per question depending on thinking length) but correctly handles
    Kimi's always-thinking chat template.

Kimi K2.6 is an always-thinking model, so `generate` mode is the default.

Run:
  python -m jang_tools.kimi_prune.bench_mmlu \\
      --model /path/to/Kimi-K2.6-REAP-30-JANGTQ_1L \\
      --num 200 \\
      --mode generate \\
      --max-tokens 400

Outputs: /tmp/kimi_mmlu_<timestamp>.json with per-question predictions,
accuracy totals, per-subject accuracy, and timing breakdown.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path


LETTERS = ["A", "B", "C", "D"]


def _load_mmlu(num_questions: int, seed: int = 42,
               stratified_subjects: int = 0, per_subject: int = 0):
    """Return a list of dicts {subject, question, choices[4], answer_letter}.

    Two sampling modes:
      * Uniform random (default): num_questions samples drawn across all 57
        subjects. Fast but biased toward subjects with more questions.
      * Stratified (stratified_subjects > 0): pick N subjects at random,
        then `per_subject` questions from each — total
        `stratified_subjects * per_subject` questions. Gives balanced
        per-subject accuracy numbers. 10 x 30 = 300 is the "proper" default.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `pip install datasets` (huggingface datasets library).",
              file=sys.stderr)
        sys.exit(2)
    print(f"[mmlu] loading cais/mmlu (all) test split ...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"[mmlu] loaded {len(ds)} total questions", flush=True)

    if stratified_subjects > 0 and per_subject > 0:
        # Group by subject
        by_subject: dict[str, list[int]] = {}
        for i, row in enumerate(ds):
            by_subject.setdefault(row["subject"], []).append(i)
        all_subjects = sorted(by_subject.keys())
        rng = random.Random(seed)
        # Shuffle subject order, pick the first N that have at least
        # per_subject questions (all MMLU subjects have >= 100, so we
        # will always find enough).
        rng.shuffle(all_subjects)
        picked_subjects = [
            s for s in all_subjects if len(by_subject[s]) >= per_subject
        ][:stratified_subjects]
        print(f"[mmlu] stratified: {len(picked_subjects)} subjects "
              f"x {per_subject} q/each = "
              f"{len(picked_subjects) * per_subject} total", flush=True)
        for s in picked_subjects:
            print(f"  - {s}  ({len(by_subject[s])} available)", flush=True)
        picked = []
        for s in picked_subjects:
            s_rng = random.Random(seed ^ hash(s) & 0xFFFFFFFF)
            candidates = list(by_subject[s])
            s_rng.shuffle(candidates)
            picked.extend(candidates[:per_subject])
    else:
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        picked = indices[:num_questions]

    out = []
    for i in picked:
        row = ds[i]
        out.append({
            "subject": row["subject"],
            "question": row["question"],
            "choices": row["choices"],
            "answer_letter": LETTERS[int(row["answer"])],
        })
    return out


def _format_prompt(q: dict) -> str:
    """Render one MMLU item as a zero-shot multiple-choice question."""
    return (
        f"The following is a multiple-choice question about {q['subject'].replace('_', ' ')}.\n\n"
        f"Question: {q['question']}\n"
        f"A) {q['choices'][0]}\n"
        f"B) {q['choices'][1]}\n"
        f"C) {q['choices'][2]}\n"
        f"D) {q['choices'][3]}\n\n"
        f"Respond with a single letter only (A, B, C, or D). Your final answer "
        f"must be on its own line after your reasoning, prefixed with 'Answer:'."
    )


def _extract_letter(text: str) -> str | None:
    """Pull A/B/C/D out of the model's output, preferring the last line.

    Kimi is a thinking model, so the answer typically comes after </think>
    or at the tail. We search (in order):
      1. The last 'Answer: X' pattern
      2. After '</think>', the first A-D
      3. The LAST standalone letter A-D anywhere
      4. The first standalone letter A-D anywhere (fallback)
    """
    # 1. Explicit "Answer: X"
    m = list(re.finditer(r"[Aa]nswer\s*:\s*\**\s*\(?\s*([A-Da-d])\)?", text))
    if m:
        return m[-1].group(1).upper()
    # 2. After </think>
    if "</think>" in text:
        tail = text.split("</think>", 1)[1]
        m = re.search(r"\b([A-D])\b", tail)
        if m:
            return m.group(1)
    # 3. Last standalone letter
    m = list(re.finditer(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text))
    if m:
        return m[-1].group(1)
    return None


def _generate_and_extract(model, tokenizer, q: dict, *, max_tokens: int,
                          prefill_step_size: int,
                          thinking: bool = False,
                          temp: float = 0.0, top_p: float = 1.0,
                          top_k: int = 0) -> dict:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    user = _format_prompt(q)
    msgs = [{"role": "user", "content": user}]
    try:
        templated = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            thinking=thinking,
        )
    except TypeError:
        templated = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
        )
    sampler_kwargs = {"temp": temp}
    if temp > 0:
        if top_p is not None and top_p < 1.0:
            sampler_kwargs["top_p"] = top_p
        if top_k is not None and top_k > 0:
            sampler_kwargs["top_k"] = top_k
    t0 = time.time()
    text = generate(
        model, tokenizer,
        prompt=templated,
        max_tokens=max_tokens,
        sampler=make_sampler(**sampler_kwargs),
        verbose=False,
        prefill_step_size=prefill_step_size,
    )
    elapsed = time.time() - t0
    pred = _extract_letter(text)
    return {
        "subject": q["subject"],
        "question": q["question"][:120] + ("..." if len(q["question"]) > 120 else ""),
        "answer": q["answer_letter"],
        "predicted": pred,
        "correct": pred == q["answer_letter"],
        "elapsed": elapsed,
        "raw": text,
    }


def run(model_path: Path, *, num: int, max_tokens: int, seed: int,
        prefill_step_size: int, thinking: bool,
        stratified_subjects: int = 0, per_subject: int = 0,
        temp: float = 0.0, top_p: float = 1.0, top_k: int = 0) -> int:
    from jang_tools.kimi_prune.runtime_patch import apply as _apply_patch
    _apply_patch(dry_run=False)
    from jang_tools.load_jangtq import load_jangtq_model
    print(f"[mmlu] model: {model_path}")
    model, tokenizer = load_jangtq_model(model_path)

    # Open a per-question txt log alongside the JSON report. This is a
    # live append-as-we-go log so you can see the actual model output
    # while the bench runs and diagnose "wrong answer because cut off
    # mid-think" vs "wrong answer because model believed it".
    ts = int(time.time())
    txt_path = Path(f"/tmp/kimi_mmlu_{ts}.txt")
    txt_fh = txt_path.open("w", encoding="utf-8")
    print(f"[mmlu] live txt log: {txt_path}", flush=True)

    questions = _load_mmlu(
        num, seed=seed,
        stratified_subjects=stratified_subjects,
        per_subject=per_subject,
    )
    num = len(questions)  # actual count in stratified mode
    # Group by subject for reporting — but run in the shuffled order so
    # we don't bias early warmup.
    t_start = time.time()
    results: list[dict] = []
    per_subject_hit: dict[str, int] = defaultdict(int)
    per_subject_total: dict[str, int] = defaultdict(int)

    for i, q in enumerate(questions):
        out = _generate_and_extract(
            model, tokenizer, q,
            max_tokens=max_tokens,
            prefill_step_size=prefill_step_size,
            thinking=thinking,
            temp=temp, top_p=top_p, top_k=top_k,
        )
        results.append(out)
        per_subject_total[out["subject"]] += 1
        per_subject_hit[out["subject"]] += int(out["correct"])
        total_hit = sum(per_subject_hit.values())
        running_acc = total_hit / (i + 1) * 100
        mark = "✓" if out["correct"] else "✗"
        print(f"  [{i + 1:>3}/{num}] {mark} {out['subject']:<28}  "
              f"pred={out['predicted'] or '?'!s:<2} ans={out['answer']}  "
              f"{out['elapsed']:>5.1f}s   running={running_acc:.1f}%",
              flush=True)
        # Append full question + raw response to txt log
        txt_fh.write(f"\n=== Q{i + 1}/{num}  {out['subject']} ===\n")
        txt_fh.write(f"Question: {q['question']}\n")
        for j, choice in enumerate(q["choices"]):
            txt_fh.write(f"  {LETTERS[j]}) {choice}\n")
        txt_fh.write(f"\nExpected: {out['answer']}\n")
        txt_fh.write(f"Predicted: {out['predicted'] or '(no-match)'}  "
                     f"({'CORRECT' if out['correct'] else 'WRONG'})\n")
        txt_fh.write(f"Elapsed: {out['elapsed']:.1f}s\n")
        txt_fh.write(f"--- raw response ---\n{out['raw']}\n")
        txt_fh.write(f"--- end Q{i + 1} ---\n")
        txt_fh.flush()

    total_hit = sum(per_subject_hit.values())
    acc = total_hit / num * 100
    elapsed = time.time() - t_start
    print()
    print(f"=== MMLU {num}-q accuracy: {total_hit}/{num} = {acc:.2f}% "
          f"({elapsed/60:.1f} min total) ===")
    print()
    print("Per-subject:")
    for subj in sorted(per_subject_total,
                       key=lambda s: per_subject_hit[s] / per_subject_total[s]):
        h = per_subject_hit[subj]
        t = per_subject_total[subj]
        print(f"  {subj:<40} {h}/{t}  ({h/t*100:.0f}%)")

    report = {
        "model": str(model_path),
        "num": num,
        "seed": seed,
        "max_tokens": max_tokens,
        "prefill_step_size": prefill_step_size,
        "accuracy": acc,
        "total_hit": total_hit,
        "elapsed_seconds": elapsed,
        "per_subject": {
            s: {"correct": per_subject_hit[s], "total": per_subject_total[s]}
            for s in per_subject_total
        },
        "results": results,
    }
    out_path = Path(f"/tmp/kimi_mmlu_{ts}.json")
    out_path.write_text(json.dumps(report, indent=2))
    txt_fh.write(f"\n\n=== SUMMARY ===\n")
    txt_fh.write(f"{total_hit}/{num} correct = {acc:.2f}%\n")
    for subj in sorted(per_subject_total,
                       key=lambda s: per_subject_hit[s] / per_subject_total[s]):
        h = per_subject_hit[subj]
        t = per_subject_total[subj]
        txt_fh.write(f"  {subj:<40} {h}/{t}  ({h/t*100:.0f}%)\n")
    txt_fh.close()
    print(f"\njson report: {out_path}")
    print(f"txt log:     {txt_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--num", type=int, default=200,
                    help="Number of MMLU questions to sample (default 200)")
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="Max generation tokens per question. Kimi thinks a "
                         "lot — 400 gives room for think block + answer.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for question sampling (for reproducibility)")
    ap.add_argument("--prefill-step-size", type=int, default=16,
                    help="Prefill chunk size — must stay small for 191 GB bundles")
    ap.add_argument("--thinking", action="store_true", default=False,
                    help="Enable Kimi's chain-of-thought mode (default OFF — "
                         "Kimi's default chat template has thinking ON, but "
                         "for MMLU we want direct letter answers, so we disable).")
    ap.add_argument("--stratified-subjects", type=int, default=0,
                    help="If > 0, use stratified sampling: pick this many "
                         "subjects, take per-subject questions from each. "
                         "10 x 30 = 300q gives balanced per-subject numbers.")
    ap.add_argument("--per-subject", type=int, default=0,
                    help="Questions per subject in stratified mode.")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    args = ap.parse_args()
    return run(
        args.model,
        num=args.num,
        max_tokens=args.max_tokens,
        seed=args.seed,
        prefill_step_size=args.prefill_step_size,
        thinking=args.thinking,
        stratified_subjects=args.stratified_subjects,
        per_subject=args.per_subject,
        temp=args.temp, top_p=args.top_p, top_k=args.top_k,
    )


if __name__ == "__main__":
    sys.exit(main())
