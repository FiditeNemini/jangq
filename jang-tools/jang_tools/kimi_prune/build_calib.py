"""Build mixed-domain calibration corpus for Kimi K2.6 expert pruning.

Streams public HF datasets (no full download), extracts text via per-dataset
field mapping, applies light dedup + length filter, writes a single JSONL
with one {"text": "...", "domain": "..."} record per line.

Target mix (weights by domain, scaled to --target-tokens):

  coding       22%   the-stack-smol (python/js/rust/go), Magicoder OSS,
                     HumanEval+MBPP solutions, CodeFeedback
  cybersec     18%   security_paper_data, CyberNative eval corpus,
                     CTF writeup dump (if available), ShellCommands
  agentic      18%   glaive-function-calling-v2, xlam-function-calling,
                     AgentInstruct, SWE-bench oracle traces
  general      20%   tulu-3 SFT mixture sample, OpenHermes-2.5 sample,
                     ArXiv abstracts
  systems       8%   Linux man pages + kernel docs + networking RFCs
                     (via a dedicated HF dump if one exists, else skipped)
  chinese       8%   COIG-CQIA (Chinese instruction data)
  longctx       6%   PG-19 or arxiv long docs — guards 256K-context ability

The domain tag is preserved so the profiler can later produce
per-domain routing stats (useful for adaptive per-layer prune ratios).

Usage:
  python -m jang_tools.kimi_prune.build_calib \\
      --out <path/to/data-drive>/kimi_calib/corpus.jsonl \\
      --target-tokens 5_000_000 \\
      --seed 42

Tokenization is deferred: the profiler tokenizes on-the-fly using Kimi's
tiktoken tokenizer (trust_remote_code) once the model is downloaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


# Rough chars-per-token estimate for pre-tokenization budget allocation.
# Kimi's tiktoken extends GPT-4 BPE with CJK; ~3.5 chars/tok across a
# mixed English+code+Chinese corpus is the right-sized overestimate
# (budgets will be trimmed to exact count after tokenize pass).
CHARS_PER_TOK = 3.5


@dataclass
class Source:
    domain: str
    weight: float           # 0..1 share of target tokens
    dataset: str            # HF repo id
    config: str | None      # HF config name, or None
    split: str              # e.g. "train"
    text_fn: Callable[[dict], str | None]
    max_records: int = 500_000  # hard stop per source to avoid over-pull


def _conv_text(row: dict) -> str | None:
    """Extract text from a ShareGPT / conversations-format row."""
    convs = row.get("conversations") or row.get("messages") or []
    parts = []
    for m in convs:
        if isinstance(m, dict):
            parts.append(str(m.get("value") or m.get("content") or ""))
        elif isinstance(m, str):
            parts.append(m)
    joined = "\n\n".join(p for p in parts if p)
    return joined if len(joined) > 60 else None


def _field(key: str):
    def _f(row: dict) -> str | None:
        v = row.get(key)
        if isinstance(v, str) and len(v) > 60:
            return v
        return None
    return _f


def _join_fields(*keys: str):
    def _f(row: dict) -> str | None:
        parts = [str(row.get(k)) for k in keys if row.get(k)]
        joined = "\n\n".join(parts)
        return joined if len(joined) > 60 else None
    return _f


def _code_solution(row: dict) -> str | None:
    parts = []
    for k in ("prompt", "instruction", "problem", "task"):
        v = row.get(k)
        if v:
            parts.append(f"# {v}")
    for k in ("canonical_solution", "solution", "code", "completion", "response"):
        v = row.get(k)
        if v:
            parts.append(str(v))
    joined = "\n\n".join(parts)
    return joined if len(joined) > 80 else None


SOURCES: list[Source] = [
    # --- Coding ---------------------------------------------------------
    # the-stack-smol is gated; use public instruction/code datasets instead.
    Source("coding", 0.07, "ise-uiuc/Magicoder-OSS-Instruct-75K", None, "train",
           _join_fields("problem", "solution"), max_records=75_000),
    Source("coding", 0.05, "m-a-p/CodeFeedback-Filtered-Instruction", None, "train",
           _join_fields("query", "answer"), max_records=60_000),
    Source("coding", 0.04, "nickrosh/Evol-Instruct-Code-80k-v1", None, "train",
           _join_fields("instruction", "output"), max_records=60_000),
    Source("coding", 0.03, "HuggingFaceH4/CodeAlpaca_20K", None, "train",
           _join_fields("prompt", "completion"), max_records=20_000),
    Source("coding", 0.03, "iamtarun/python_code_instructions_18k_alpaca", None, "train",
           _join_fields("instruction", "input", "output"), max_records=18_000),

    # --- Cybersec / pentest --------------------------------------------
    # HF cybersec ecosystem is thin; CyberNative DPO is the main load-bearing
    # source. Added broader security-instruction datasets as fallbacks.
    Source("cybersec", 0.08, "CyberNative/Code_Vulnerability_Security_DPO", None, "train",
           _join_fields("system", "question", "chosen"), max_records=25_000),
    Source("cybersec", 0.05, "Trendyol/cybersecurity-instruction-datasets", None, "train",
           _join_fields("instruction", "input", "output"), max_records=20_000),
    Source("cybersec", 0.05, "jackhhao/jailbreak-classification", None, "train",
           _field("prompt"), max_records=15_000),

    # --- Agentic / tool calling / swarm --------------------------------
    # xlam-function-calling-60k went gated. Use Hermes FC and glaiveai
    # ShareGPT variant as primary; SWE-bench oracle for agentic patches.
    Source("agentic", 0.06, "NousResearch/hermes-function-calling-v1", None, "train",
           _conv_text, max_records=50_000),
    Source("agentic", 0.05, "glaiveai/glaive-function-calling-v2", None, "train",
           _field("chat"), max_records=30_000),
    Source("agentic", 0.03, "lilacai/glaive-function-calling-v2-sharegpt", None, "train",
           _conv_text, max_records=20_000),
    # THUDM/AgentInstruct uses split names AS config: os/db/alfworld/webshop/kg/mind2web.
    Source("agentic", 0.02, "THUDM/AgentInstruct", None, "os",
           _conv_text, max_records=3_000),
    Source("agentic", 0.02, "THUDM/AgentInstruct", None, "webshop",
           _conv_text, max_records=3_000),
    Source("agentic", 0.03, "princeton-nlp/SWE-bench_oracle", None, "train",
           _join_fields("problem_statement", "patch"), max_records=10_000),

    # --- General knowledge ---------------------------------------------
    Source("general", 0.08, "allenai/tulu-3-sft-mixture", None, "train",
           _conv_text, max_records=80_000),
    Source("general", 0.05, "teknium/OpenHermes-2.5", None, "train",
           _conv_text, max_records=60_000),
    Source("general", 0.04, "HuggingFaceH4/ultrachat_200k", None, "train_sft",
           _conv_text, max_records=40_000),
    Source("general", 0.03, "ccdv/arxiv-summarization", None, "train",
           _field("abstract"), max_records=30_000),

    # --- Systems knowledge (sysadmin/networking/devops) ----------------
    # Approximate via command datasets + technical instruction subsets.
    Source("systems", 0.04, "b-mc2/sql-create-context", None, "train",
           _join_fields("question", "context", "answer"), max_records=15_000),
    Source("systems", 0.04, "cognitivecomputations/dolphin-coder", None, "train",
           _conv_text, max_records=20_000),

    # --- Chinese --------------------------------------------------------
    # BAAI/COIG-CQIA resolves as gated; use public alternatives.
    Source("chinese", 0.04, "silk-road/alpaca-data-gpt4-chinese", None, "train",
           _join_fields("instruction_zh", "input_zh", "output_zh"), max_records=40_000),
    Source("chinese", 0.02, "wangrui6/Zhihu-KOL", None, "train",
           _join_fields("INSTRUCTION", "RESPONSE"), max_records=20_000),
    Source("chinese", 0.02, "YeungNLP/firefly-train-1.1M", None, "train",
           _join_fields("input", "target"), max_records=30_000),

    # --- Long context ---------------------------------------------------
    # deepmind/pg19 loads via script (unsupported). Use emozilla's parquet
    # mirror, and add a second source for technical long docs.
    Source("longctx", 0.04, "emozilla/pg19", None, "train",
           _field("text"), max_records=1_500),
    Source("longctx", 0.02, "ccdv/arxiv-summarization", None, "train",
           _field("article"), max_records=2_000),
]


def _hash(s: str) -> str:
    return hashlib.blake2b(s.encode("utf-8", "ignore"), digest_size=8).hexdigest()


def _stream(src: Source) -> Iterator[dict]:
    from datasets import load_dataset  # heavy import, done inside
    kwargs = dict(split=src.split, streaming=True)
    if src.config:
        kwargs["name"] = src.config
    ds = load_dataset(src.dataset, **kwargs)
    for i, row in enumerate(ds):
        if i >= src.max_records:
            return
        yield row


def build(out_path: Path, target_tokens: int, seed: int = 42) -> dict:
    rng = random.Random(seed)
    seen: set[str] = set()
    counts: dict[str, int] = {}
    token_budget_by_domain: dict[str, int] = {}

    total_weight = sum(s.weight for s in SOURCES)
    for s in SOURCES:
        token_budget_by_domain.setdefault(s.domain, 0)
        token_budget_by_domain[s.domain] += int(target_tokens * s.weight / total_weight)

    print(f"[build_calib] target={target_tokens:,} tokens "
          f"(~{target_tokens * CHARS_PER_TOK / 1e9:.2f} GB raw text)", flush=True)
    for d, b in token_budget_by_domain.items():
        print(f"  {d:10s} budget={b:>10,} tok  (~{b * CHARS_PER_TOK / 1e6:.1f} MB)",
              flush=True)

    written_tokens: dict[str, int] = {d: 0 for d in token_budget_by_domain}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")

    t0 = time.time()
    n_written = 0
    with tmp_path.open("w", encoding="utf-8") as fh:
        for src in SOURCES:
            domain_budget = token_budget_by_domain[src.domain]
            # Each source contributes roughly (src.weight / sum_domain_weight) of domain budget.
            domain_w = sum(s.weight for s in SOURCES if s.domain == src.domain)
            src_budget = int(domain_budget * src.weight / domain_w)
            src_tokens = 0

            print(f"[{src.domain}] {src.dataset}"
                  f"{'/' + src.config if src.config else ''} "
                  f"budget={src_budget:,}", flush=True)
            try:
                for row in _stream(src):
                    if written_tokens[src.domain] >= domain_budget:
                        break
                    if src_tokens >= src_budget:
                        break
                    txt = src.text_fn(row)
                    if not txt:
                        continue
                    # Hard-cap single record at 32 KB to keep shape reasonable.
                    if len(txt) > 32_000:
                        txt = txt[:32_000]
                    h = _hash(txt[:512])
                    if h in seen:
                        continue
                    seen.add(h)
                    approx_tok = int(len(txt) / CHARS_PER_TOK)
                    rec = {"text": txt, "domain": src.domain, "src": src.dataset}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    src_tokens += approx_tok
                    written_tokens[src.domain] += approx_tok
                    n_written += 1
                    counts[src.domain] = counts.get(src.domain, 0) + 1
                    if n_written % 5000 == 0:
                        dt = time.time() - t0
                        print(f"  ... {n_written:,} records, "
                              f"{sum(written_tokens.values()):,} approx-tokens, "
                              f"{dt:.0f}s elapsed", flush=True)
            except Exception as e:
                print(f"  !! skipping {src.dataset}: {type(e).__name__}: {e}",
                      flush=True)
                continue

    tmp_path.rename(out_path)
    summary = {
        "out": str(out_path),
        "records": n_written,
        "approx_tokens_by_domain": written_tokens,
        "records_by_domain": counts,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(f"[build_calib] done: {json.dumps(summary, indent=2)}", flush=True)
    with out_path.with_suffix(".summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-tokens", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(args.out, args.target_tokens, args.seed)


if __name__ == "__main__":
    main()
