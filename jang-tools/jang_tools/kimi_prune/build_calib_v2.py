"""Build mixed-domain calibration corpus v2 — rebalanced for MMLU + coding.

Target mix (weights by domain, scaled to --target-tokens):

  coding        24%   Python/TS/Rust/Go/C++ real code
  agentic       20%   tool-use traces (OpenAI/Kimi/MiniMax format)
  general       20%   world knowledge prose
  academic_mc   10%   MMLU-train + ARC + OpenBookQA + SciQ (MC format)
  science        8%   physics/chem/bio/math/medicine prose
  chinese        8%   CN-language prose + code
  cybersec       5%   reduced from 13% (Kimi v1); no more overweighting
  systems        3%
  longctx        2%

Changes from v1:
  * NEW academic_mc bucket (0 -> 10%). Explicitly teaches REAP which
    experts handle structured-knowledge MC retrieval the way MMLU eval
    probes them. MMLU contamination avoided by using the TRAIN split
    (never overlaps MMLU test set).
  * Science split out of general (was 0 explicit) -> 8%.
  * Cybersec 13% -> 5% (Kimi v1 was overweighted).
  * Coding 22% -> 24% (slight bump — top use case).
  * Agentic held at 20%.
  * Chinese 8% -> 8% held.

Usage:
  python -m jang_tools.kimi_prune.build_calib_v2 \\
      --out <path/to/data-drive>/kimi_calib/corpus_v2.jsonl \\
      --target-tokens 5_000_000 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

# Reuse helpers from v1
from jang_tools.kimi_prune.build_calib import (
    Source, _conv_text, _field, _join_fields, _code_solution,
    _hash, CHARS_PER_TOK,
)


def _mc_text(row: dict) -> str | None:
    """Format a multiple-choice question as `Question ... A B C D Answer: X`.

    Works for MMLU-train, ARC, OpenBookQA, SciQ (after per-dataset field
    normalization via the _mc_* wrappers below).
    """
    q = row.get("question") or row.get("query") or row.get("prompt")
    choices = row.get("choices") or row.get("options")
    answer = row.get("answer") or row.get("answerKey") or row.get("correct_answer")
    if not (q and choices and answer is not None):
        return None
    if isinstance(choices, dict):
        # ARC / OpenBookQA format: {"text": [...], "label": [...]}
        texts = choices.get("text") or []
        labels = choices.get("label") or []
        if not texts or len(texts) != len(labels):
            return None
        # answer is a label like "A"/"B"/etc
        ans_letter = str(answer).strip()
        lines = [f"{lbl}. {t}" for lbl, t in zip(labels, texts)]
    elif isinstance(choices, list):
        if len(choices) < 2:
            return None
        letters = ["A", "B", "C", "D", "E"]
        lines = [f"{letters[i]}. {c}" for i, c in enumerate(choices[:5])]
        # MMLU: answer is int; convert
        if isinstance(answer, int):
            ans_letter = letters[answer]
        else:
            # assume letter string already
            ans_letter = str(answer).strip()
    else:
        return None
    return f"Question: {q}\n" + "\n".join(lines) + f"\nAnswer: {ans_letter}"


def _sciq_mc(row: dict) -> str | None:
    """SciQ rows have `question`, 3 `distractor1/2/3` + 1 `correct_answer`."""
    q = row.get("question")
    if not q:
        return None
    correct = row.get("correct_answer")
    d1 = row.get("distractor1")
    d2 = row.get("distractor2")
    d3 = row.get("distractor3")
    if not all(isinstance(x, str) for x in (correct, d1, d2, d3)):
        return None
    # Deterministic letter assignment via hash so it's reproducible
    opts = [correct, d1, d2, d3]
    # Shuffle deterministically based on question text for reproducibility
    seed = int(hashlib.blake2b(q.encode(), digest_size=4).hexdigest(), 16)
    rng = random.Random(seed)
    order = list(range(4))
    rng.shuffle(order)
    letters = ["A", "B", "C", "D"]
    lines = []
    ans_letter = None
    for out_i, in_i in enumerate(order):
        lines.append(f"{letters[out_i]}. {opts[in_i]}")
        if in_i == 0:  # correct was at original idx 0
            ans_letter = letters[out_i]
    support = row.get("support", "")
    prefix = f"Context: {support}\n\n" if support else ""
    return f"{prefix}Question: {q}\n" + "\n".join(lines) + f"\nAnswer: {ans_letter}"


SOURCES_V2: list[Source] = [
    # === coding 24% =====================================================
    Source("coding", 0.08, "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train",
           _join_fields("problem", "solution"), max_records=75_000),
    Source("coding", 0.05, "m-a-p/CodeFeedback-Filtered-Instruction", None, "train",
           _join_fields("query", "answer"), max_records=60_000),
    Source("coding", 0.04, "nickrosh/Evol-Instruct-Code-80k-v1", None, "train",
           _join_fields("instruction", "output"), max_records=60_000),
    Source("coding", 0.04, "HuggingFaceH4/CodeAlpaca_20K", None, "train",
           _join_fields("prompt", "completion"), max_records=20_000),
    Source("coding", 0.03, "iamtarun/python_code_instructions_18k_alpaca", None, "train",
           _join_fields("instruction", "input", "output"), max_records=18_000),

    # === agentic 20% ====================================================
    Source("agentic", 0.07, "NousResearch/hermes-function-calling-v1", None, "train",
           _conv_text, max_records=50_000),
    Source("agentic", 0.05, "glaiveai/glaive-function-calling-v2", None, "train",
           _field("chat"), max_records=30_000),
    Source("agentic", 0.03, "lilacai/glaive-function-calling-v2-sharegpt", None, "train",
           _conv_text, max_records=20_000),
    Source("agentic", 0.02, "THUDM/AgentInstruct", None, "os",
           _conv_text, max_records=3_000),
    Source("agentic", 0.01, "THUDM/AgentInstruct", None, "webshop",
           _conv_text, max_records=3_000),
    Source("agentic", 0.02, "princeton-nlp/SWE-bench_oracle", None, "train",
           _join_fields("problem_statement", "patch"), max_records=10_000),

    # === general 20% ====================================================
    Source("general", 0.08, "allenai/tulu-3-sft-mixture", None, "train",
           _conv_text, max_records=80_000),
    Source("general", 0.05, "teknium/OpenHermes-2.5", None, "train",
           _conv_text, max_records=60_000),
    Source("general", 0.04, "HuggingFaceH4/ultrachat_200k", None, "train_sft",
           _conv_text, max_records=40_000),
    Source("general", 0.03, "ccdv/arxiv-summarization", None, "train",
           _field("abstract"), max_records=30_000),

    # === academic_mc 10%  (NEW) =========================================
    # MMLU train split — never overlaps MMLU test (different questions).
    Source("academic_mc", 0.05, "cais/mmlu", "all", "auxiliary_train",
           _mc_text, max_records=40_000),
    Source("academic_mc", 0.02, "allenai/ai2_arc", "ARC-Challenge", "train",
           _mc_text, max_records=2_000),
    Source("academic_mc", 0.015, "allenai/openbookqa", "main", "train",
           _mc_text, max_records=5_000),
    Source("academic_mc", 0.015, "allenai/sciq", None, "train",
           _sciq_mc, max_records=13_000),

    # === science 8% (prose / not MC) ====================================
    Source("science", 0.03, "ccdv/arxiv-summarization", None, "train",
           _field("article"), max_records=10_000),
    Source("science", 0.025, "pubmed_qa", "pqa_labeled", "train",
           _join_fields("question", "long_answer"), max_records=1_000),
    Source("science", 0.025, "camel-ai/physics", None, "train",
           _join_fields("topic", "message_1", "message_2"), max_records=20_000),

    # === chinese 8% =====================================================
    Source("chinese", 0.04, "silk-road/alpaca-data-gpt4-chinese", None, "train",
           _join_fields("instruction_zh", "input_zh", "output_zh"), max_records=40_000),
    Source("chinese", 0.02, "wangrui6/Zhihu-KOL", None, "train",
           _join_fields("INSTRUCTION", "RESPONSE"), max_records=20_000),
    Source("chinese", 0.02, "YeungNLP/firefly-train-1.1M", None, "train",
           _join_fields("input", "target"), max_records=30_000),

    # === cybersec 5% (reduced from 13%) ================================
    Source("cybersec", 0.03, "CyberNative/Code_Vulnerability_Security_DPO", None, "train",
           _join_fields("system", "question", "chosen"), max_records=15_000),
    Source("cybersec", 0.02, "Trendyol/cybersecurity-instruction-datasets", None, "train",
           _join_fields("instruction", "input", "output"), max_records=15_000),

    # === systems 3% =====================================================
    Source("systems", 0.015, "b-mc2/sql-create-context", None, "train",
           _join_fields("question", "context", "answer"), max_records=10_000),
    Source("systems", 0.015, "cognitivecomputations/dolphin-coder", None, "train",
           _conv_text, max_records=10_000),

    # === longctx 2% =====================================================
    Source("longctx", 0.01, "emozilla/pg19", None, "train",
           _field("text"), max_records=1_000),
    Source("longctx", 0.01, "ccdv/arxiv-summarization", None, "train",
           _field("article"), max_records=1_500),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-tokens", type=int, default=8_600_000,
                    help="Total tokens to collect (v1 Kimi used 8.6M)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from jang_tools.kimi_prune import build_calib as _v1
    _v1.SOURCES = SOURCES_V2
    print("[build-v2] corpus mix:", flush=True)
    for d in sorted({s.domain for s in SOURCES_V2}):
        w = sum(s.weight for s in SOURCES_V2 if s.domain == d)
        print(f"  {d:<14} {w * 100:.1f}%", flush=True)
    _v1.build(args.out, args.target_tokens, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
