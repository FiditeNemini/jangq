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
    from jang_tools.loader import load_jang_model
    return load_jang_model(str(model_dir))


def _load_vlm(model_dir: Path):
    """Load a VL/video model via jang_tools.load_jangtq_vlm if JANGTQ, else via mlx_vlm.

    M112 (iter 37): only fall back to mlx_vlm on ImportError / "not a JANGTQ
    model" errors. Pre-iter-37 the `except Exception: pass` caught EVERY
    error from the JANGTQ path — including genuine load failures — and
    silently fell through to mlx_vlm which produces a confusing error
    instead of the informative JANGTQ one. See iter-20 M45 for the
    original symptom (load_jangtq_vlm → load_jangtq_vlm_model rename was
    masked by exactly this except-all pattern).
    """
    # Try the JANGTQ-VL loader first. Narrow catch: only retry via mlx_vlm if
    # the module itself can't be imported (e.g. mlx_vlm standalone install
    # without jang_tools) OR if load_jangtq_vlm_model raises a specific
    # "not a JANGTQ model" sentinel. Any other error — corrupted shard,
    # missing file, kernel crash — should propagate up so the user sees
    # the real problem.
    try:
        from jang_tools.load_jangtq_vlm import load_jangtq_vlm_model
    except ImportError:
        # jang_tools.load_jangtq_vlm module not importable — only happens
        # if someone ran `pip install mlx_vlm` standalone without the
        # jang_tools JANGTQ extras. Fall back to vanilla mlx_vlm.
        from mlx_vlm import load
        return load(str(model_dir))
    # Module imported — use the JANGTQ path. Any failure here is a real
    # problem with the model dir, not a fallback trigger.
    return load_jangtq_vlm_model(str(model_dir))


def _apply_chat_template_if_any(
    tokenizer,
    prompt: str,
    *,
    enable_thinking: bool = True,
) -> str | list[int]:
    """Apply the tokenizer's chat template if it has one — otherwise return the
    raw prompt string unchanged. Without this, Qwen/Llama/Gemma models see
    "Hello" rather than "<|im_start|>user\\nHello<|im_end|>\\n<|im_start|>assistant\\n"
    and either loop forever or emit garbage.

    ``enable_thinking`` (M121): opt-in smoke-test toggle for reasoning models.
    GLM-5.1 / Qwen3.6 / MiniMax M2.7 chat templates wrap the prompt with a
    <think>…</think> block when enable_thinking=True (the HF default). For
    a 150-token in-wizard smoke test that's a black hole — the 100+ tokens
    of thinking consume the whole budget before an answer emits. Pass False
    from `cmd_inference` when the user sets `--no-thinking`, and the kwarg
    flows through to apply_chat_template. Non-reasoning templates simply
    ignore the unknown kwarg — tokenizers that strictly reject it hit the
    fallback path below and get retried without the kwarg (so we still
    produce a templated prompt, not a raw string).
    """
    # TokenizerWrapper (mlx_lm wraps HF tokenizers — chat_template attr lives on
    # the inner HF tokenizer, not the wrapper).
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    if not getattr(inner, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return inner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Strict tokenizer rejected the enable_thinking kwarg — retry without
        # it. We still get a properly templated prompt (the old code path);
        # only the reasoning-toggle behavior degrades.
        try:
            return inner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt
    except Exception:
        # Some templates require other vars the user didn't supply. Fall back
        # to raw prompt rather than crashing generation.
        return prompt


def _make_sampler(temperature: float):
    """Return an mlx_lm sampler or None. temp<=0 → argmax (greedy), no sampler needed."""
    if temperature <= 0.0:
        return None
    try:
        from mlx_lm.sample_utils import make_sampler
        return make_sampler(temp=temperature)
    except Exception:
        return None


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    *,
    enable_thinking: bool = True,
) -> dict:
    """Generate via mlx_lm. Returns dict with text + timing.

    Applies the tokenizer's chat template when available and wires temperature
    through mlx_lm's sampler API. Prior to M29, both were silently ignored:
    temperature slider was a UI lie, and chat models ran with no chat template
    causing infinite loops / garbage output. M121 adds the enable_thinking
    passthrough for reasoning-model smoke tests.
    """
    from mlx_lm import generate
    templated = _apply_chat_template_if_any(tokenizer, prompt, enable_thinking=enable_thinking)
    sampler = _make_sampler(temperature)

    t0 = time.time()
    kwargs: dict[str, Any] = {"prompt": templated, "max_tokens": max_tokens, "verbose": False}
    if sampler is not None:
        kwargs["sampler"] = sampler
    try:
        text = generate(model, tokenizer, **kwargs)
    except TypeError:
        # Older mlx-lm may not accept verbose OR sampler kwarg; retry without.
        kwargs.pop("verbose", None)
        kwargs.pop("sampler", None)
        text = generate(model, tokenizer, **kwargs)
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
            result = _generate_text(
                model, tokenizer, args.prompt, args.max_tokens, args.temperature,
                enable_thinking=not args.no_thinking,
            )
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
    # M121 (iter 45): short-answer smoke-test toggle for reasoning models.
    # GLM-5.1, Qwen3.6, and MiniMax M2.7 chat templates default to
    # enable_thinking=True, wrapping the prompt with a <think>…</think> block
    # that eats 100+ tokens. For wizard smoke tests with --max-tokens 150,
    # pass --no-thinking to skip the reasoning wrapper and see a direct answer.
    # Non-reasoning templates silently ignore the kwarg.
    p.add_argument(
        "--no-thinking",
        action="store_true",
        help="Skip the chat template's <think> block on reasoning models "
             "(GLM-5.1 / Qwen3.6 / MiniMax M2.7) — use for short-answer smoke tests",
    )
    p.set_defaults(func=cmd_inference)
