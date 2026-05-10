"""Inference-time top-K override for MoE routers.

Every modern MoE we ship — Hy3, dots1, DeepSeek-V3/V4, Qwen3-MoE,
Bailing/Ling, Laguna, MiniMax, etc. — exposes a per-token expert count
through `num_experts_per_tok` in `config.json`. The trained value is the
"correct" K (the route_norm distribution was calibrated for it), but at
inference the K can be lowered to trade quality for decode speed.

This module walks a loaded model and overrides every router's `top_k`
attribute (and any sibling `num_experts_per_tok` attribute on outer MoE
containers). Top-1 MoE families like ZAYA have no `top_k` attribute and
are silently skipped.

Usage::

    model, tok = load_jangtq_model(path)
    from jang_tools.topk_override import apply_topk_override
    n_overridden = apply_topk_override(model, 4)

Env var::

    JANGTQ_TOPK_OVERRIDE=4 python ...

The override is applied automatically by `load_jangtq_model` after
hydration when `JANGTQ_TOPK_OVERRIDE` is set to a positive integer.

The override is **non-destructive** — model weights and module shapes
are untouched. Only the integer counter that controls argpartition
selection is changed. Reverting is one call to `apply_topk_override`
with the original K (or `None`).

Quality caveats
---------------
The model was trained with the original K. Reducing K means each
selected expert's weight in the route_norm renormalization rises
(weights sum to 1 across K → per-expert weight ≈ 1/K). On easy prompts
(short text, common code) the quality drop is small. On hard prompts
(long reasoning chains, rare expert specialties) the drop can be large.
Always A/B on your target benchmark before shipping the override as a
default.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def apply_topk_override(model, k: Optional[int]) -> int:
    """Override every MoE router's `top_k` to ``k``.

    Walks `model.named_modules()` and sets:
      - any module with attribute `top_k` (router classes)
      - any module with attribute `num_experts_per_tok` (outer MoE)

    Returns the number of attributes patched. Returns 0 for non-MoE
    models (no router classes found). Top-1 MoE families (ZAYA) have
    no `top_k` and are silently no-op.

    If ``k`` is None, no override is applied (returns 0).
    Raises ``ValueError`` if ``k`` is not a positive integer.
    """
    if k is None:
        return 0
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"top_k override must be positive int, got {k!r}")

    n_patched = 0
    for path, mod in model.named_modules():
        if hasattr(mod, "top_k") and isinstance(getattr(mod, "top_k"), int):
            old = mod.top_k
            mod.top_k = k
            if old != k:
                n_patched += 1
                logger.debug("top_k override: %s.top_k %d -> %d", path, old, k)
        if hasattr(mod, "num_experts_per_tok") and isinstance(
            getattr(mod, "num_experts_per_tok"), int
        ):
            old = mod.num_experts_per_tok
            mod.num_experts_per_tok = k
            if old != k:
                n_patched += 1
                logger.debug(
                    "top_k override: %s.num_experts_per_tok %d -> %d", path, old, k
                )

    if n_patched > 0:
        logger.info(
            "JANGTQ top-K override applied: K=%d (%d router/MoE attributes patched)",
            k,
            n_patched,
        )
    return n_patched


def topk_override_from_env() -> Optional[int]:
    """Return the K value from `JANGTQ_TOPK_OVERRIDE` env var, or None.

    Empty / unset / "0" returns None. Any other value must parse as a
    positive int or `ValueError` is raised at call time.
    """
    val = os.environ.get("JANGTQ_TOPK_OVERRIDE", "").strip()
    if not val or val == "0":
        return None
    try:
        k = int(val)
    except ValueError as exc:
        raise ValueError(
            f"JANGTQ_TOPK_OVERRIDE must be a positive integer, got {val!r}"
        ) from exc
    if k < 1:
        raise ValueError(f"JANGTQ_TOPK_OVERRIDE must be >= 1, got {k}")
    return k
