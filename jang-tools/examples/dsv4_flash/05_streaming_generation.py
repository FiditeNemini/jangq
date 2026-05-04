"""05 — Token-by-token streaming with per-frame reasoning vs content split.

Demonstrates the same reasoning-leak guarantees as 02_thinking but at the
streaming boundary: as each token arrives, we maintain the parser state
and emit two SSE-style streams — `reasoning_delta` and `content_delta`
— with no overlap, no leak, and no out-of-order emission.

This is the pattern an OpenAI-compatible server must implement.

Run: python3 05_streaming_generation.py [bundle_path]
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

DEFAULT_BUNDLE = Path.home() / ".mlxstudio/models/_bundles/DeepSeek-V4-Flash-JANGTQ"


class StreamingThinkSplitter:
    """Stream parser that splits `<think>...</think>` from final content.

    Tracks state across token boundaries — works even if `<think>` or
    `</think>` arrives split across multiple tokens. Emits `(kind, text)`
    where kind ∈ {"reasoning", "content"} and text is the new fragment
    since the last call.
    """
    OPEN  = "<think>"
    CLOSE = "</think>"

    def __init__(self, enable_thinking: bool):
        self.in_thought = False
        self.expecting_open = enable_thinking
        self.pending = ""    # buffered chars not yet committed (for tag boundary)
        self.reasoning = ""
        self.content = ""

    def feed(self, new_text: str):
        """Returns list of (kind, fragment) for this delta."""
        out = []
        self.pending += new_text
        # Resolve as much of pending as possible
        while self.pending:
            if self.expecting_open:
                idx = self.pending.find(self.OPEN)
                if idx == -1:
                    # No <think> yet — could be partial. Hold last len(OPEN)-1 chars.
                    safe = self.pending[:max(0, len(self.pending) - (len(self.OPEN) - 1))]
                    if safe:
                        self.content += safe
                        out.append(("content", safe))
                        self.pending = self.pending[len(safe):]
                    return out
                # Pre-tag content goes to content
                pre = self.pending[:idx]
                if pre:
                    self.content += pre
                    out.append(("content", pre))
                self.pending = self.pending[idx + len(self.OPEN):]
                self.in_thought = True
                self.expecting_open = False
                continue
            if self.in_thought:
                idx = self.pending.find(self.CLOSE)
                if idx == -1:
                    # Hold last len(CLOSE)-1 chars in case of partial
                    safe = self.pending[:max(0, len(self.pending) - (len(self.CLOSE) - 1))]
                    if safe:
                        self.reasoning += safe
                        out.append(("reasoning", safe))
                        self.pending = self.pending[len(safe):]
                    return out
                # In-thought content
                pre = self.pending[:idx]
                if pre:
                    self.reasoning += pre
                    out.append(("reasoning", pre))
                self.pending = self.pending[idx + len(self.CLOSE):]
                self.in_thought = False
                continue
            # After </think> → all content
            self.content += self.pending
            out.append(("content", self.pending))
            self.pending = ""
            return out
        return out


def main():
    bundle = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BUNDLE
    if not bundle.exists():
        print(f"bundle not found: {bundle}"); sys.exit(2)

    os.environ.setdefault("DSV4_LONG_CTX", "1")
    import mlx.core as mx
    mx.set_memory_limit(int(os.environ.get("JANG_MEMORY_LIMIT_GB", "200")) * 1024**3)

    from jang_tools.load_jangtq import load_jangtq_model
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from jang_tools.dsv4.runtime import _inject_chat_template

    print(f"=== Loading DSV4-Flash from {bundle.name} ===", flush=True)
    t0 = time.time()
    model, tok = load_jangtq_model(str(bundle))
    _inject_chat_template(tok, str(bundle))
    print(f"  loaded in {time.time()-t0:.1f}s\n", flush=True)

    prompt = "What is 17 + 28? Think briefly, then answer."
    text = tok.apply_chat_template(
        [{"role":"user","content":prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ids = mx.array(tok.encode(text))

    splitter = StreamingThinkSplitter(enable_thinking=True)
    sampler = make_sampler(temp=0.6, top_p=0.95)
    print("=== Streaming ===")
    print(f"USER: {prompt}\n")
    last_kind = None
    out_ids = []
    for tid, _ in generate_step(prompt=ids, model=model, max_tokens=512, sampler=sampler):
        out_ids.append(int(tid))
        if int(tid) == tok.eos_token_id: break
        # Decode only the new piece. tok.decode([tid]) drops leading space sometimes;
        # decode the full ids and diff to be safe (slower but correct).
        full = tok.decode(out_ids)
        # Naive: emit one char at a time would be too chatty — emit per-token here.
        # Compute the new fragment since the last full decode.
        # (For real servers, use a streaming detokenizer with byte-level.)
        if not hasattr(main, "_last_full"): main._last_full = ""
        delta = full[len(main._last_full):]
        main._last_full = full
        for kind, frag in splitter.feed(delta):
            if kind != last_kind:
                print(f"\n[{kind.upper()}] ", end="", flush=True)
                last_kind = kind
            print(frag, end="", flush=True)
    print("\n\n=== Final ===")
    print(f"reasoning: {len(splitter.reasoning)} chars")
    print(f"content:   {splitter.content!r}")
    # Leak audit
    for tag in ("<think>", "</think>"):
        if tag in splitter.content:
            print(f"FAIL — {tag!r} leaked into content"); sys.exit(3)
    print("PASS — streaming split clean")


if __name__ == "__main__":
    main()
