"""Logit-based MMLU bench for JANGTQ bundles.

For each question: build `prompt = question + choices + Answer:`, run one
forward pass, pick the letter with the highest log-probability at the
next-token position. No generation, no max-tokens, no extraction regex.

This is how the HuggingFace `lm-evaluation-harness` scores MMLU — the
industry standard. It's also ~10x faster than generative scoring and
doesn't get confused by CoT-trained models that reason before answering.

Stratified args are identical to bench_mmlu.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

LETTERS = ["A", "B", "C", "D"]
_materialize = getattr(mx, "async_eval")


def _load_mmlu(stratified_subjects, per_subject, seed=42):
    from datasets import load_dataset
    print("[mmlu] loading cais/mmlu (all) test split ...", flush=True)
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subject = {}
    for i, row in enumerate(ds):
        by_subject.setdefault(row["subject"], []).append(i)
    all_subjects = sorted(by_subject.keys())
    rng = random.Random(seed)
    rng.shuffle(all_subjects)
    picked_subjects = [
        s for s in all_subjects if len(by_subject[s]) >= per_subject
    ][:stratified_subjects]
    print(
        f"[mmlu] stratified: {len(picked_subjects)} subjects "
        f"x {per_subject} q/each = {len(picked_subjects) * per_subject}",
        flush=True,
    )
    picked = []
    for s in picked_subjects:
        s_rng = random.Random(seed ^ (hash(s) & 0xFFFFFFFF))
        candidates = list(by_subject[s])
        s_rng.shuffle(candidates)
        picked.extend(candidates[:per_subject])
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


def _format_prompt(q):
    return (
        "The following is a multiple choice question about "
        f"{q['subject'].replace('_', ' ')}. Respond with a single letter "
        "(A, B, C, or D).\n\n"
        f"Question: {q['question']}\n"
        f"A. {q['choices'][0]}\n"
        f"B. {q['choices'][1]}\n"
        f"C. {q['choices'][2]}\n"
        f"D. {q['choices'][3]}\n"
        "Answer:"
    )


def _letter_token_ids(tokenizer):
    ids = {}
    for L in LETTERS:
        for candidate in (f" {L}", L, f"\n{L}"):
            tokens = tokenizer.encode(candidate, add_special_tokens=False)
            if len(tokens) == 1:
                ids[L] = tokens[0]
                break
        else:
            tokens = tokenizer.encode(f" {L}", add_special_tokens=False)
            ids[L] = tokens[0] if tokens else None
    return ids


def _forward_last_logits(model, tokenizer, prompt, prefill_step_size=16):
    from mlx_lm.models.cache import make_prompt_cache
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    x = mx.array([ids], dtype=mx.int32)
    cache = make_prompt_cache(model)
    T = x.shape[1]
    offset = 0
    final_logits = None
    while offset < T:
        end = min(offset + prefill_step_size, T)
        out = model(x[:, offset:end], cache=cache)
        if end == T:
            _materialize(out, [c.state for c in cache])
            final_logits = out[:, -1, :]
        else:
            _materialize([c.state for c in cache])
            mx.clear_cache()
        offset = end
    _materialize(final_logits)
    return final_logits


def run(model_path, *, stratified_subjects, per_subject,
        prefill_step_size, seed):
    from jang_tools.kimi_prune.runtime_patch import apply as _apply_patch
    _apply_patch(dry_run=False)
    from jang_tools.load_jangtq import load_jangtq_model
    print(f"[mmlu-logit] model: {model_path}")
    model, tokenizer = load_jangtq_model(model_path)

    letter_ids = _letter_token_ids(tokenizer)
    print(f"[mmlu-logit] letter token IDs: {letter_ids}")
    missing = [L for L, tid in letter_ids.items() if tid is None]
    if missing:
        print(f"ERROR: couldn't resolve token IDs for {missing}", file=sys.stderr)
        return 2

    questions = _load_mmlu(stratified_subjects, per_subject, seed=seed)
    total = len(questions)

    ts = int(time.time())
    txt_path = Path(f"/tmp/kimi_mmlu_logit_{ts}.txt")
    txt_fh = txt_path.open("w", encoding="utf-8")
    print(f"[mmlu-logit] live log: {txt_path}")

    t_start = time.time()
    results = []
    hit = 0
    per_subject_hit = defaultdict(int)
    per_subject_total = defaultdict(int)

    for i, q in enumerate(questions):
        prompt = _format_prompt(q)
        t0 = time.time()
        logits = _forward_last_logits(
            model, tokenizer, prompt, prefill_step_size=prefill_step_size,
        )
        lgt = logits[0]
        scores = {L: float(lgt[letter_ids[L]].item()) for L in LETTERS}
        predicted = max(scores, key=scores.get)
        elapsed = time.time() - t0
        correct = predicted == q["answer_letter"]
        hit += int(correct)
        per_subject_total[q["subject"]] += 1
        per_subject_hit[q["subject"]] += int(correct)
        running = hit / (i + 1) * 100
        mark = "✓" if correct else "✗"
        print(
            f"  [{i + 1:>3}/{total}] {mark} {q['subject']:<34}  "
            f"pred={predicted} ans={q['answer_letter']}  "
            f"{elapsed:>4.1f}s  running={running:.1f}%  "
            f"  A={scores['A']:.1f} B={scores['B']:.1f} "
            f"C={scores['C']:.1f} D={scores['D']:.1f}",
            flush=True,
        )
        results.append({
            "subject": q["subject"],
            "question": q["question"][:120] + ("..." if len(q["question"]) > 120 else ""),
            "answer": q["answer_letter"],
            "predicted": predicted,
            "correct": correct,
            "scores": scores,
            "elapsed": elapsed,
        })
        txt_fh.write(
            f"\n=== Q{i + 1}/{total} {q['subject']} ({'PASS' if correct else 'FAIL'}) ===\n"
            f"Question: {q['question']}\n"
        )
        for j, c in enumerate(q["choices"]):
            txt_fh.write(f"  {LETTERS[j]}. {c}\n")
        txt_fh.write(
            f"Expected: {q['answer_letter']}  Predicted: {predicted}\n"
            f"Scores: A={scores['A']:.2f} B={scores['B']:.2f} "
            f"C={scores['C']:.2f} D={scores['D']:.2f}\n"
        )
        txt_fh.flush()

    elapsed_total = time.time() - t_start
    acc = hit / total * 100
    print()
    print(f"=== MMLU-logit {total}-q: {hit}/{total} = {acc:.2f}% "
          f"({elapsed_total/60:.1f} min) ===")
    print("\nPer-subject:")
    for s in sorted(per_subject_total,
                    key=lambda x: per_subject_hit[x] / per_subject_total[x]):
        h = per_subject_hit[s]
        t = per_subject_total[s]
        print(f"  {s:<40} {h}/{t}  ({h/t*100:.0f}%)")

    report = {
        "model": str(model_path),
        "num": total,
        "accuracy": acc,
        "hit": hit,
        "elapsed_seconds": elapsed_total,
        "per_subject": {
            s: {"correct": per_subject_hit[s], "total": per_subject_total[s]}
            for s in per_subject_total
        },
        "results": results,
    }
    out_path = Path(f"/tmp/kimi_mmlu_logit_{ts}.json")
    out_path.write_text(json.dumps(report, indent=2))
    txt_fh.write(f"\n=== SUMMARY ===\n{hit}/{total} = {acc:.2f}%\n")
    txt_fh.close()
    print(f"\njson: {out_path}\ntxt:  {txt_path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--stratified-subjects", type=int, default=30)
    ap.add_argument("--per-subject", type=int, default=10)
    ap.add_argument("--prefill-step-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    return run(
        args.model,
        stratified_subjects=args.stratified_subjects,
        per_subject=args.per_subject,
        prefill_step_size=args.prefill_step_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    sys.exit(main())
