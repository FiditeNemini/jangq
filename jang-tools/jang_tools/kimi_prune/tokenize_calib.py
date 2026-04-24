"""Tokenize corpus.jsonl into packed fixed-length batches using Kimi tokenizer.

Output: a single safetensors file with
  tokens: (N, T) int32 — packed token ids, causal
  meta:   sidecar .json with {'n_sequences': N, 'seq_len': T, 'domains': [...]}

Packing strategy: concatenate tokenized texts with EOS between records,
then split into fixed T-length chunks. Domain tag is assigned to a chunk
based on the majority source record (for per-domain breakdowns in score.py).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file


def build_chunks(
    corpus_path: Path,
    tokenizer,
    seq_len: int,
    max_sequences: int,
    eos_token_id: int,
) -> tuple[np.ndarray, list[str]]:
    """Stream-tokenize corpus records, pack into (N, seq_len) int32."""
    buf: list[int] = []
    buf_domains: list[str] = []  # one domain tag per token (for majority vote)
    chunks: list[np.ndarray] = []
    chunk_domains: list[str] = []

    t0 = time.time()
    total_records = 0
    total_tokens = 0
    with corpus_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if len(chunks) >= max_sequences:
                break
            rec = json.loads(line)
            text = rec["text"]
            domain = rec.get("domain", "?")
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids.append(eos_token_id)
            buf.extend(ids)
            buf_domains.extend([domain] * len(ids))
            total_records += 1
            total_tokens += len(ids)
            # Emit as many full chunks as we have
            while len(buf) >= seq_len and len(chunks) < max_sequences:
                chunk = np.asarray(buf[:seq_len], dtype=np.int32)
                dom_counts = Counter(buf_domains[:seq_len])
                dom, _ = dom_counts.most_common(1)[0]
                chunks.append(chunk)
                chunk_domains.append(dom)
                buf = buf[seq_len:]
                buf_domains = buf_domains[seq_len:]
            if total_records % 1000 == 0:
                dt = time.time() - t0
                print(f"  records={total_records:,}  tokens={total_tokens:,}  "
                      f"chunks={len(chunks)}/{max_sequences}  {dt:.0f}s", flush=True)

    if len(chunks) < max_sequences and len(buf) > 0 and len(buf) < seq_len:
        # Pad tail to make a final chunk if we have room.
        pad_id = eos_token_id
        buf_full = buf + [pad_id] * (seq_len - len(buf))
        chunks.append(np.asarray(buf_full, dtype=np.int32))
        chunk_domains.append(Counter(buf_domains).most_common(1)[0][0] if buf_domains else "?")

    arr = np.stack(chunks[:max_sequences], axis=0)  # (N, seq_len)
    print(f"[tokenize] packed {arr.shape[0]} chunks × {seq_len} tokens "
          f"= {arr.size:,} total from {total_records:,} records "
          f"({time.time() - t0:.0f}s)", flush=True)
    return arr, chunk_domains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--tokenizer-path", required=True, type=Path,
                    help="Path to model dir containing tokenizer_config + tiktoken.model")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--n-sequences", type=int, default=256,
                    help="Target number of packed sequences (batches).")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    print(f"[tokenize] loading Kimi tokenizer from {args.tokenizer_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    print(f"  vocab_size={tok.vocab_size}  eos_id={tok.eos_token_id}", flush=True)

    arr, domains = build_chunks(
        args.corpus, tok, args.seq_len, args.n_sequences, tok.eos_token_id
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"tokens": torch.from_numpy(arr)}, str(args.out))

    meta = {
        "n_sequences": int(arr.shape[0]),
        "seq_len": int(arr.shape[1]),
        "total_tokens": int(arr.size),
        "eos_token_id": tok.eos_token_id,
        "domains_per_chunk": domains,
    }
    with args.out.with_suffix(".json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[tokenize] wrote {args.out}  (+ sidecar .json)", flush=True)


if __name__ == "__main__":
    main()
