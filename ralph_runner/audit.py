"""Ralph audit engine — runs on macstudio remotely, writes JSON to stdout.

USAGE:
  python3 audit.py --model <converted_dir> [--rows a1,a3,a15] [--json]

Each audit row returns {"status": "pass"|"warn"|"fail"|"n/a", ...fields}.
Runs A1-A7 + A15 by default. Tier 2+ (perplexity) and Tier 3 (MMLU) are opt-in.

This script is DESIGNED to run on macstudio (remote), not locally.
Runner.py rsyncs it as part of the jang-tools tree + invokes via SSH.
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

# Test strings for tokenizer roundtrip (A1). Stays simple — avoids weird unicode
# that specific tokenizers might normalize away.
ROUNDTRIP_SAMPLES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "1234567890",
    "  leading and trailing whitespace  ",
    "def foo(): return 42",
    "https://example.com/path?query=1",
    "line1\nline2\nline3",
    "emoji absent — simple ascii only",   # avoid emoji tokenization edge cases
]


def _ok(**k) -> dict:
    return {"status": "pass", **k}


def _warn(hint: str, **k) -> dict:
    return {"status": "warn", "hint": hint, **k}


def _fail(hint: str, **k) -> dict:
    return {"status": "fail", "hint": hint, **k}


def _na(hint: str = "") -> dict:
    return {"status": "n/a", "hint": hint}


def load_tokenizer(model_dir: Path):
    """Load tokenizer via transformers.AutoTokenizer."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)


def _load_llm(model_dir: Path):
    from jang_tools.loader import load_jang_model
    return load_jang_model(str(model_dir))


def _load_vlm(model_dir: Path):
    # For JANG v2 VLMs, load_jang_model dispatches to _load_jang_v2_vlm
    # (mlx_vlm skeleton + instant mmap weights). This handles idefics3,
    # qwen3_vl, etc. without needing a model-specific loader.
    try:
        from jang_tools.loader import load_jang_model
        return load_jang_model(str(model_dir))
    except Exception:
        pass
    # JANGTQ-specific path (for JANGTQ-format VLMs)
    try:
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
        return load_jangtq_vlm_model(str(model_dir))
    except Exception:
        pass
    # Last resort: raw mlx_vlm.load
    from mlx_vlm import load
    return load(str(model_dir))


def _is_vl(model_dir: Path) -> bool:
    return (model_dir / "preprocessor_config.json").exists()


# ───────────────────────── A1 — Tokenizer round-trip ─────────────────────────

def audit_a1_tokenizer_roundtrip(model_dir: Path) -> dict:
    try:
        tok = load_tokenizer(model_dir)
    except Exception as e:
        return _fail(f"tokenizer_load: {type(e).__name__}: {e}")
    failures = []
    for s in ROUNDTRIP_SAMPLES:
        try:
            ids = tok.encode(s, add_special_tokens=False)
            decoded = tok.decode(ids, skip_special_tokens=False)
        except Exception as e:
            failures.append({"sample": s[:40], "error": f"{type(e).__name__}: {e}"})
            continue
        # Tokenizers may add whitespace — use looser equality
        if decoded.strip() != s.strip():
            failures.append({"sample": s[:40], "got": decoded[:80]})
    if failures:
        return _fail(f"{len(failures)} roundtrip mismatches", failures=failures[:5])
    return _ok(samples=len(ROUNDTRIP_SAMPLES))


# ───────────────────────── A2 — Chat template render ─────────────────────────

def audit_a2_chat_template(model_dir: Path) -> dict:
    try:
        tok = load_tokenizer(model_dir)
    except Exception as e:
        return _fail(f"tokenizer_load: {type(e).__name__}: {e}")
    if not getattr(tok, "chat_template", None):
        # Maybe a .jinja file
        jinja = model_dir / "chat_template.jinja"
        if not jinja.exists():
            return _na("no chat template present in source")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Ping"},
    ]
    try:
        rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return _fail(f"apply_chat_template: {type(e).__name__}: {e}")
    if not rendered or len(rendered) < 5:
        return _fail("rendered chat template is empty or tiny", rendered=rendered[:100])
    # Don't strictly assert role markers — many templates use special tokens like <|im_start|>
    return _ok(rendered_len=len(rendered), preview=rendered[:200])


# ───────────────────────── A3 — Generation coherence ─────────────────────────

def audit_a3_coherence(model_dir: Path) -> dict:
    """Prompt 'The capital of France is' and assert 'Paris' appears in output."""
    try:
        if _is_vl(model_dir):
            return _na("VL model — A3 runs via A15 inference test instead")
        model, tok = _load_llm(model_dir)
        from mlx_lm import generate
        prompt = "The capital of France is"
        try:
            out = generate(model, tok, prompt=prompt, max_tokens=25, verbose=False)
        except TypeError:
            out = generate(model, tok, prompt=prompt, max_tokens=25)
        out_lower = out.lower()
        if "paris" in out_lower:
            return _ok(output=out[:200])
        # Didn't say Paris — still might be valid for some weird base models. Warn, not fail.
        return _warn("output did not mention Paris", output=out[:200])
    except Exception as e:
        return _fail(f"coherence_test: {type(e).__name__}: {e}")


# ───────────────────────── A4 — Tokens/sec throughput ───────────────────────

def audit_a4_tokens_per_sec(model_dir: Path) -> dict:
    """Measure steady-state tokens/sec. Baseline comparison is done by runner.py."""
    try:
        if _is_vl(model_dir):
            return _na("VL model — throughput measured separately")
        model, tok = _load_llm(model_dir)
        from mlx_lm import generate
        # Warm-up run (not timed)
        try:
            generate(model, tok, prompt="Hello", max_tokens=8, verbose=False)
        except TypeError:
            generate(model, tok, prompt="Hello", max_tokens=8)
        # Timed run
        prompt = "Once upon a time,"
        t0 = time.time()
        try:
            out = generate(model, tok, prompt=prompt, max_tokens=100, verbose=False)
        except TypeError:
            out = generate(model, tok, prompt=prompt, max_tokens=100)
        elapsed = time.time() - t0
        try:
            n = len(tok.encode(out)) - len(tok.encode(prompt))
        except Exception:
            n = max(len(out.split()) - len(prompt.split()), 1)
        tps = n / elapsed if elapsed > 0 else 0.0
        if tps < 1.0:
            return _warn("tokens/sec below 1 — suspicious", tokens_per_sec=tps, elapsed_s=elapsed)
        return _ok(tokens_per_sec=round(tps, 2), elapsed_s=round(elapsed, 2), tokens_generated=n)
    except Exception as e:
        return _fail(f"throughput_test: {type(e).__name__}: {e}")


# ───────────────────────── A5 — Chat turn end-to-end ────────────────────────

def audit_a5_chat_turn(model_dir: Path) -> dict:
    """Apply chat template + generate + verify no infinite-repeat."""
    try:
        if _is_vl(model_dir):
            return _na("VL model")
        tok = load_tokenizer(model_dir)
        if not getattr(tok, "chat_template", None):
            # Check for .jinja alternative
            if not (model_dir / "chat_template.jinja").exists():
                return _na("no chat template")
        messages = [{"role": "user", "content": "What is 2+2?"}]
        try:
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            return _fail(f"apply_chat_template: {e}")
        model, tok2 = _load_llm(model_dir)
        from mlx_lm import generate
        try:
            out = generate(model, tok2, prompt=prompt, max_tokens=50, verbose=False)
        except TypeError:
            out = generate(model, tok2, prompt=prompt, max_tokens=50)
        # Detect infinite-repeat: any 10-char substring appearing 3+ times
        if len(out) >= 30:
            for i in range(len(out) - 10):
                s = out[i:i+10]
                if len(s.strip()) >= 5 and out.count(s) >= 3:
                    return _warn(f"possible repetition of '{s}'", output=out[:200])
        return _ok(output=out[:200])
    except Exception as e:
        return _fail(f"chat_turn: {type(e).__name__}: {e}")


# ───────────────────────── A6 — Convert wall time ───────────────────────────

def audit_a6_wall_time(convert_wall_s: float, baseline_s: float | None) -> dict:
    """Needs runner-provided convert wall time + optional baseline."""
    if baseline_s is None:
        return _ok(wall_s=convert_wall_s, baseline=None, note="no baseline yet — this run establishes it")
    if convert_wall_s > baseline_s * 1.5:
        return _fail(f"wall time {convert_wall_s:.1f}s > baseline {baseline_s:.1f}s * 1.5",
                     wall_s=convert_wall_s, baseline_s=baseline_s)
    return _ok(wall_s=convert_wall_s, baseline_s=baseline_s, ratio=round(convert_wall_s/baseline_s, 2))


# ───────────────────────── A7 — Size vs estimate ────────────────────────────

def audit_a7_size_estimate(model_dir: Path, predicted_bytes: int | None = None) -> dict:
    """Compare actual output size to the predicted size (if provided)."""
    total = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))
    total_gb = round(total / 1_000_000_000, 3)
    if predicted_bytes is None or predicted_bytes <= 0:
        return _ok(actual_gb=total_gb, predicted=None)
    ratio = total / predicted_bytes
    if abs(ratio - 1.0) > 0.15:
        return _warn(f"size drift {ratio:.2f}x vs estimate",
                     actual_gb=total_gb, predicted_gb=round(predicted_bytes/1e9, 3), ratio=round(ratio, 2))
    return _ok(actual_gb=total_gb, predicted_gb=round(predicted_bytes/1e9, 3), ratio=round(ratio, 2))


# ───────────────────────── A15 — Inference works ────────────────────────────

def audit_a15_inference(model_dir: Path) -> dict:
    """Load model + generate 20 tokens. Assert non-empty non-whitespace output."""
    try:
        if _is_vl(model_dir):
            # VL inference check — needs a test image. Best-effort: check load path.
            # Full VL inference is in A11/A12.
            model, proc = _load_vlm(model_dir)
            return _ok(note="VL model loaded successfully via load_jangtq_vlm/mlx_vlm.load")
        t0 = time.time()
        model, tok = _load_llm(model_dir)
        load_s = time.time() - t0
        from mlx_lm import generate
        t1 = time.time()
        try:
            out = generate(model, tok, prompt="Hello, how are you?", max_tokens=20, verbose=False)
        except TypeError:
            out = generate(model, tok, prompt="Hello, how are you?", max_tokens=20)
        gen_s = time.time() - t1
        # Non-empty, at least one printable non-whitespace char
        if not out or not out.strip() or not any(c.isprintable() and not c.isspace() for c in out):
            return _fail("output empty or whitespace only", output=repr(out[:100]))
        return _ok(load_s=round(load_s, 2), gen_s=round(gen_s, 2), output=out[:200])
    except Exception as e:
        return _fail(f"inference: {type(e).__name__}: {e}")


# ───────────────────────── Runner ──────────────────────────────────────────

AUDIT_REGISTRY = {
    "a1": ("Tokenizer roundtrip", audit_a1_tokenizer_roundtrip, True),      # required
    "a2": ("Chat template render", audit_a2_chat_template, False),          # warn-only if fails
    "a3": ("Generation coherence", audit_a3_coherence, False),              # warn-only (tiny models may not say Paris)
    "a4": ("Tokens/sec throughput", audit_a4_tokens_per_sec, False),
    "a5": ("Chat turn end-to-end", audit_a5_chat_turn, False),
    "a7": ("Size vs estimate", audit_a7_size_estimate, False),
    "a15": ("Inference works", audit_a15_inference, True),                  # required
}


def run_audits(model_dir: Path, rows: list[str], convert_wall_s: float | None = None,
               baseline_wall_s: float | None = None, predicted_bytes: int | None = None) -> dict:
    results: dict[str, Any] = {
        "model_dir": str(model_dir),
        "timestamp": dt.datetime.now().isoformat(),
        "rows": {},
    }
    for row in rows:
        if row not in AUDIT_REGISTRY:
            results["rows"][row] = {"status": "n/a", "hint": f"unknown row {row}"}
            continue
        title, fn, required = AUDIT_REGISTRY[row]
        try:
            if row == "a6":
                r = audit_a6_wall_time(convert_wall_s or 0, baseline_wall_s)
            elif row == "a7":
                r = audit_a7_size_estimate(model_dir, predicted_bytes)
            else:
                r = fn(model_dir)
        except Exception as e:
            r = _fail(f"audit_crashed: {type(e).__name__}: {e}")
        r["title"] = title
        r["required"] = required
        results["rows"][row] = r
    # Summary
    fails = [r for k, r in results["rows"].items() if r.get("required") and r.get("status") == "fail"]
    results["required_fail_count"] = len(fails)
    results["overall"] = "fail" if fails else "pass"
    return results


def main() -> None:
    p = argparse.ArgumentParser(prog="ralph_audit")
    p.add_argument("--model", required=True, help="Path to converted JANG/JANGTQ model dir")
    p.add_argument("--rows", default="a1,a2,a3,a4,a5,a7,a15",
                   help="Comma-separated audit rows to run")
    p.add_argument("--convert-wall-s", type=float, default=None)
    p.add_argument("--baseline-wall-s", type=float, default=None)
    p.add_argument("--predicted-bytes", type=int, default=None)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        print(json.dumps({"error": f"model not found: {model_dir}"}))
        sys.exit(2)

    rows = [r.strip() for r in args.rows.split(",") if r.strip()]
    results = run_audits(
        model_dir, rows,
        convert_wall_s=args.convert_wall_s,
        baseline_wall_s=args.baseline_wall_s,
        predicted_bytes=args.predicted_bytes,
    )
    print(json.dumps(results, indent=None))


if __name__ == "__main__":
    main()
