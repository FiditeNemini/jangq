"""Load a converted JANG/JANGTQ model and generate text.

Dispatches between dense/MoE (via jang_tools.loader) and VL/video (via load_jangtq_vlm).
Outputs either plain text or JSON with timing info.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Any


def _is_vl(model_dir: Path) -> bool:
    return (model_dir / "preprocessor_config.json").exists()


def _resolve_resource_mb() -> float:
    """Return current process RSS in MB (best-effort)."""
    try:
        import resource
        # On macOS, ru_maxrss is in bytes; on Linux, KB. Detect by value.
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if r > 1e9:   # bytes (macOS)
            return r / 1_000_000
        return r / 1000   # KB (Linux)
    except Exception:
        return 0.0


def _load_llm(model_dir: Path):
    """Load a dense/MoE LLM via jang_tools.loader. Returns (model, tokenizer)."""
    from jang_tools.loader import load_model
    return load_model(str(model_dir))


def _load_vlm(model_dir: Path):
    """Load a VL/video model via jang_tools.load_jangtq_vlm if JANGTQ, else via mlx_vlm."""
    # Try the JANGTQ-VL loader first
    try:
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm
        return load_jangtq_vlm(str(model_dir))
    except Exception:
        pass
    # Fallback: mlx_vlm direct
    from mlx_vlm import load
    return load(str(model_dir))


def _generate_text(model, tokenizer, prompt: str, max_tokens: int, temperature: float) -> dict:
    """Generate via mlx_lm. Returns dict with text + timing."""
    from mlx_lm import generate
    t0 = time.time()
    # mlx_lm.generate signature varies across versions; try both styles
    try:
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    except TypeError:
        # older mlx-lm may not accept verbose kwarg
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    elapsed = time.time() - t0
    # Count tokens in response for tok/s
    try:
        n_toks = len(tokenizer.encode(text)) if hasattr(tokenizer, "encode") else len(text.split())
    except Exception:
        n_toks = len(text.split())
    tok_s = n_toks / elapsed if elapsed > 0 else 0.0
    return {"text": text, "elapsed_s": elapsed, "tokens": n_toks, "tokens_per_sec": tok_s}


def _generate_vl(model, processor, prompt: str, max_tokens: int, image_path: str | None, video_path: str | None) -> dict:
    """Generate for a VL model using mlx_vlm."""
    from mlx_vlm import generate
    t0 = time.time()
    kwargs: dict[str, Any] = {"model": model, "processor": processor, "prompt": prompt, "max_tokens": max_tokens}
    if image_path:
        from PIL import Image
        kwargs["image"] = Image.open(image_path)
    if video_path:
        # video path support varies by mlx-vlm version — pass through
        kwargs["video"] = video_path
    text = generate(**kwargs)
    if hasattr(text, "text"):  # GenerateResponse object
        text = text.text
    elapsed = time.time() - t0
    n_toks = len(str(text).split())
    return {"text": str(text), "elapsed_s": elapsed, "tokens": n_toks, "tokens_per_sec": n_toks / max(elapsed, 1e-6)}


def cmd_inference(args) -> None:
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)

    t_load = time.time()
    result: dict[str, Any]
    try:
        if _is_vl(model_dir):
            model, processor = _load_vlm(model_dir)
            load_s = time.time() - t_load
            result = _generate_vl(model, processor, args.prompt, args.max_tokens, args.image, args.video)
        else:
            model, tokenizer = _load_llm(model_dir)
            load_s = time.time() - t_load
            result = _generate_text(model, tokenizer, args.prompt, args.max_tokens, args.temperature)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if args.json:
            print(json.dumps({"error": msg, "model": str(model_dir)}, indent=None))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(3)

    result["load_time_s"] = load_s
    result["peak_rss_mb"] = _resolve_resource_mb()
    result["model"] = str(model_dir)

    if args.json:
        print(json.dumps(result, indent=None))
    else:
        print(result["text"])


def register(subparsers) -> None:
    p = subparsers.add_parser("inference", help="Generate from a converted JANG/JANGTQ model")
    p.add_argument("--model", required=True, help="Path to converted model dir")
    p.add_argument("--prompt", required=True, help="Prompt text")
    p.add_argument("--max-tokens", type=int, default=50, help="Max new tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    p.add_argument("--image", help="Image path (VL models)")
    p.add_argument("--video", help="Video path (video VL models)")
    p.add_argument("--json", action="store_true", help="JSON output with timing")
    p.set_defaults(func=cmd_inference)
