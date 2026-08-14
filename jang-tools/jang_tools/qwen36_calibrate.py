"""Calibration capture for Qwen3.6-27B (qwen3_5) — one pass, four uses.

Records the **per-input-channel second moment** `E[x_c^2]` for every Linear in
the model. That single statistic serves simultaneously as:

  * **AWQ** salient-channel scales
  * **imatrix** activation weighting
  * the **Hessian diagonal**, since `tr(H) = sum_c E[x_c^2]`
    (see docs/internal/_method/hessian-trace-allocation.md)
  * a per-module **sensitivity score** `tr(H) * ||W||_F^2` for bit allocation

Implementation note: MLX resolves `module(x)` through `type(module).__call__`,
so an instance attribute cannot shadow it. We patch `nn.Linear.__call__` (and
`QuantizedLinear` if present) at class level and dispatch on `id(module)`, which
keeps the hook exact and removable.

Accumulates in float64 to avoid drift over long corpora. Memory is trivial —
one vector of `in_features` per module (607 modules, max 17408 wide).

    python -m jang_tools.qwen36_calibrate <model_dir> <out.safetensors> \
        [--limit N] [--max-tokens N] [--images dir]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# id(module) -> dotted path, for modules we want to record.
_TARGETS: dict[int, str] = {}
_SUMSQ: dict[str, np.ndarray] = {}
_COUNT: dict[str, int] = {}
_PATCHED: list[tuple[type, object]] = []


def _accumulate(path: str, x: mx.array) -> None:
    xf = x.reshape(-1, x.shape[-1])
    s = (xf.astype(mx.float32) ** 2).sum(axis=0)
    mx.eval(s)
    v = np.array(s, dtype=np.float64)
    if path in _SUMSQ:
        _SUMSQ[path] += v
    else:
        _SUMSQ[path] = v
    _COUNT[path] = _COUNT.get(path, 0) + int(xf.shape[0])


def install_hooks(model, include_vision: bool = True) -> int:
    """Patch Linear.__call__ at class level; register target modules by id."""
    classes = [nn.Linear]
    q = getattr(nn, "QuantizedLinear", None)
    if q is not None:
        classes.append(q)

    for cls in classes:
        orig = cls.__call__

        def make(orig=orig):
            def patched(self, x, *a, **k):
                p = _TARGETS.get(id(self))
                if p is not None:
                    try:
                        _accumulate(p, x)
                    except Exception:  # never let capture break the forward
                        pass
                return orig(self, x, *a, **k)
            return patched

        _PATCHED.append((cls, orig))
        cls.__call__ = make()

    n = 0
    for path, mod in model.named_modules():
        if not isinstance(mod, tuple(classes)):
            continue
        if not include_vision and path.startswith("vision_tower"):
            continue
        _TARGETS[id(mod)] = path
        n += 1
    return n


def remove_hooks() -> None:
    for cls, orig in _PATCHED:
        cls.__call__ = orig
    _PATCHED.clear()
    _TARGETS.clear()


# ── calibration corpus ───────────────────────────────────────────────────────
# Matches the DISTRIBUTION THE MODEL IS FOR: thinking traces, agentic coding,
# tool use, long-context recall. Calibrating on bare completions would optimise
# for the wrong thing (the LFM2.5 lesson: in-domain KL came out 2.5x better
# than vendor precisely because the corpus matched).
PROMPTS_REASONING = [
    "A train leaves at 3:15pm travelling 82 km/h. A second leaves 40 minutes later at 110 km/h. When does it catch up? Show your reasoning.",
    "Prove that the square root of 3 is irrational.",
    "A bag has 4 red, 6 blue and 5 green marbles. Two are drawn without replacement. What is P(same colour)?",
    "Explain why gradient descent with momentum converges faster than plain SGD on ill-conditioned problems.",
    "If every A is B, and some B are C, does it follow that some A are C? Explain carefully.",
]
PROMPTS_CODING = [
    "Write a Python function that merges overlapping intervals, then explain its complexity.",
    "Refactor this to remove the nested loop:\n\nfor i in range(n):\n    for j in range(n):\n        if a[i]==b[j]: out.append((i,j))",
    "Implement a thread-safe LRU cache in Python with O(1) get and put.",
    "Find the bug:\n\ndef binsearch(a,t):\n    lo,hi=0,len(a)\n    while lo<hi:\n        m=(lo+hi)//2\n        if a[m]<t: lo=m\n        else: hi=m\n    return lo",
    "Write a SQL query returning the second-highest salary per department, handling ties.",
]
PROMPTS_TOOLS = [
    "What's the weather in Santa Clara and should I bring an umbrella?",
    "Search the repository for every call site of `parse_config` and summarise them.",
    "Read the file at /etc/hosts and tell me which domains are redirected.",
]
PROMPTS_GENERAL = [
    "Summarise the causes of the 1873 financial panic in three paragraphs.",
    "Explain the difference between a B-tree and an LSM tree for storage engines.",
    "Translate to French and explain any idioms: 'Don't count your chickens before they hatch.'",
]
TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


IMAGE_PROMPTS = [
    "Describe this image in detail, including colours and layout.",
    "What text appears in this image? Transcribe it exactly.",
    "Read the chart and describe the trend.",
    "List every distinct shape and where it sits.",
]


def build_image_corpus(proc, model, image_dir: Path):
    """Vision-tower calibration. Video reuses these same Linears (only the
    temporal patching upstream differs), so images cover the tower."""
    from mlx_vlm.prompt_utils import apply_chat_template
    imgs = sorted(str(p) for p in image_dir.glob("*.png"))
    out = []
    for i, img in enumerate(imgs):
        prompt = IMAGE_PROMPTS[i % len(IMAGE_PROMPTS)]
        p = apply_chat_template(proc, model.config,
                                [{"role": "user", "content": prompt}], num_images=1)
        out.append((p, img))
    return out


def build_corpus(tokenizer, limit: int | None = None):
    """Render prompts through the REAL chat template, thinking ON (the default)."""
    items = []
    for p in PROMPTS_REASONING + PROMPTS_CODING + PROMPTS_GENERAL:
        items.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False, enable_thinking=True))
    for p in PROMPTS_TOOLS:
        items.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tools=TOOLS,
            add_generation_prompt=True, tokenize=False, enable_thinking=True))
    # A non-thinking slice too, so the instruct preset is represented.
    for p in PROMPTS_GENERAL:
        items.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False))
    return items[:limit] if limit else items


def main(argv) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 1
    src, out = Path(argv[1]), Path(argv[2])
    limit = None
    max_tokens = 96
    image_dir = None
    for i, a in enumerate(argv):
        if a == "--limit":
            limit = int(argv[i + 1])
        if a == "--max-tokens":
            max_tokens = int(argv[i + 1])
        if a == "--images":
            image_dir = Path(argv[i + 1])

    from mlx_vlm import load, generate

    print(f"  loading {src.name} ...", flush=True)
    t0 = time.time()
    model, proc = load(str(src))
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    n = install_hooks(model, include_vision=True)
    print(f"  hooked {n} Linear modules", flush=True)

    corpus = build_corpus(proc.tokenizer, limit)
    print(f"  corpus: {len(corpus)} prompts, generating {max_tokens} tok each "
          f"(captures BOTH prefill and decode activations)", flush=True)

    t0 = time.time()
    for i, prompt in enumerate(corpus, 1):
        generate(model, proc, prompt, max_tokens=max_tokens,
                 temperature=1.0, verbose=False)
        if i % 3 == 0 or i == len(corpus):
            print(f"    text {i}/{len(corpus)}  ({time.time()-t0:.0f}s, "
                  f"{len(_SUMSQ)} modules seen)", flush=True)

    if image_dir and image_dir.is_dir():
        img_corpus = build_image_corpus(proc, model, image_dir)
        print(f"  vision: {len(img_corpus)} images", flush=True)
        for i, (prompt, img) in enumerate(img_corpus, 1):
            generate(model, proc, prompt, image=[img], max_tokens=max_tokens,
                     temperature=1.0, verbose=False)
            print(f"    image {i}/{len(img_corpus)} {Path(img).name} "
                  f"({time.time()-t0:.0f}s, {len(_SUMSQ)} modules seen)", flush=True)

    remove_hooks()

    # ── emit: second moment per channel + trace + module metadata ────────
    tensors, meta = {}, {}
    for path, ssq in _SUMSQ.items():
        cnt = max(_COUNT[path], 1)
        second_moment = (ssq / cnt).astype(np.float32)   # E[x_c^2]
        tensors[f"{path}.second_moment"] = second_moment
        meta[path] = {"count": cnt,
                      "trace": float(second_moment.sum()),   # tr(H)
                      "in_features": int(second_moment.shape[0])}

    from safetensors.numpy import save_file
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out))
    (out.with_suffix(".json")).write_text(json.dumps(
        {"source": str(src), "modules": len(meta), "prompts": len(corpus),
         "max_tokens": max_tokens, "stats": meta}, indent=1))

    print(f"\n  captured {len(meta)} modules -> {out}")
    print(f"  sidecar  -> {out.with_suffix('.json')}")
    tot = sum(v["count"] for v in meta.values())
    print(f"  total row-samples accumulated: {tot:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
