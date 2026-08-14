"""KL / top-1 agreement of a Qwen3.6-27B bundle against a reference bundle.

The judge for every quality claim in this pipeline. Runs a fixed prompt set
through both models (teacher-forced on the reference's greedy continuation) and
reports mean KL(ref || quant) per token and top-1 agreement. Text-only prompts
so results are comparable across profiles; rendered through the real chat
template with thinking ON (the default) plus a coding slice.

    python -m jang_tools.qwen36_kl_eval <ref_bundle> <test_bundle> [--tokens 64]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

PROMPTS = [
    "Prove that the square root of 2 is irrational.",
    "Write a Python function that merges overlapping intervals and state its complexity.",
    "A bag has 4 red, 6 blue and 5 green marbles. Two are drawn without replacement. What is P(same colour)?",
    "Explain the difference between a B-tree and an LSM tree for storage engines.",
    "Implement binary search and explain the classic off-by-one pitfall.",
    "Summarise the causes of the 1873 financial panic in two paragraphs.",
]


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    ref_p, test_p = Path(argv[1]), Path(argv[2])
    n_tok = 64
    for i, a in enumerate(argv):
        if a == "--tokens":
            n_tok = int(argv[i + 1])

    from mlx_vlm import load

    print(f"  reference: {ref_p.name}")
    print(f"  test     : {test_p.name}")

    ref, proc = load(str(ref_p))

    # 1) build teacher-forced sequences with the REFERENCE (greedy)
    seqs = []
    t0 = time.time()
    for p in PROMPTS:
        prompt = proc.tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True,
            tokenize=False, enable_thinking=True)
        ids = proc.tokenizer.encode(prompt)
        toks = list(ids)
        for _ in range(n_tok):
            logits = ref.language_model(mx.array([toks]))
            if hasattr(logits, "logits"):
                logits = logits.logits
            nxt = int(mx.argmax(logits[0, -1]).item())
            toks.append(nxt)
        seqs.append((len(ids), toks))
    print(f"  rollouts built in {time.time()-t0:.0f}s")

    # 2) reference logprobs on those sequences
    def logprobs(model, seqs):
        out = []
        for start, toks in seqs:
            logits = model.language_model(mx.array([toks]))
            if hasattr(logits, "logits"):
                logits = logits.logits
            lp = logits[0, start - 1:-1].astype(mx.float32)   # predicts toks[start:]
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            mx.eval(lp)
            out.append(lp)
        return out

    ref_lp = logprobs(ref, seqs)
    del ref
    mx.clear_cache()

    test, _ = load(str(test_p))
    test_lp = logprobs(test, seqs)
    del test
    mx.clear_cache()

    tot_kl, tot_pos, agree = 0.0, 0, 0
    for (start, toks), rl, tl in zip(seqs, ref_lp, test_lp):
        p = mx.exp(rl)
        kl = (p * (rl - tl)).sum(axis=-1)
        mx.eval(kl)
        tot_kl += float(kl.sum().item())
        n = rl.shape[0]
        tot_pos += n
        agree += int((mx.argmax(rl, axis=-1) == mx.argmax(tl, axis=-1)).sum().item())

    print(f"\n  positions      : {tot_pos}")
    print(f"  mean KL (nats) : {tot_kl / tot_pos:.5f}")
    print(f"  top-1 agreement: {100.0 * agree / tot_pos:.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
