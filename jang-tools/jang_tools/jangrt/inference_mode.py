"""Inference-mode helpers shared by JANGTQ loaders.

MLX modules default to ``training=True``. The JANGTQ fused SwitchGLU decode
path deliberately refuses to run while a module is in training mode, because
training uses dynamic routing/sorting semantics. JANGTQ loaders are inference
loaders, so they must flip the model to eval mode before returning.
"""
from __future__ import annotations

from typing import Any


def _iter_named_modules(model: Any):
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        try:
            yield from named_modules()
            return
        except (TypeError, RuntimeError):
            # Some lightweight wrappers expose a named_modules attribute that
            # is not usable outside their owning runtime. Fall back to the root
            # module so callers still get best-effort eval handling.
            yield "", model
            return
    yield "", model


def ensure_inference_mode(model: Any, *, label: str = "model") -> dict[str, Any]:
    """Put an MLX model into eval mode and return a small diagnostic report."""
    eval_called = False
    eval_fn = getattr(model, "eval", None)
    if callable(eval_fn):
        eval_fn()
        eval_called = True
    else:
        for _name, module in _iter_named_modules(model):
            if hasattr(module, "training"):
                try:
                    setattr(module, "training", False)
                except (AttributeError, TypeError, RuntimeError):
                    continue

    remaining = []
    for name, module in _iter_named_modules(model):
        if getattr(module, "training", False):
            remaining.append(name or type(module).__name__)

    return {
        "label": label,
        "eval_called": eval_called,
        "training_modules_remaining": len(remaining),
        "remaining_examples": remaining[:8],
    }


__all__ = ["ensure_inference_mode"]
