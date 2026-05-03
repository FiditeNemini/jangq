"""Pure helper: swap nn.Linear / SwitchLinear modules to TurboQuant
variants based on `.tq_packed` / `.tq_norms` / `.tq_bits` keys in a
weight dict, set the packed tensors, and return the leftover regular
(non-TQ) weights for the caller to feed into its existing affine load
path.

Why this exists
---------------
The full ``_hydrate_jangtq_model`` in ``jang_tools.load_jangtq`` is
built around ``mlx_lm.utils.load_model`` and assumes the model class
ships a ``sanitize`` method + lives in ``mlx_lm.models``. Laguna and
Mistral-Medium-3.5 don't satisfy either constraint — they're served
from ``jang_tools.laguna.runtime`` / ``jang_tools.mistral3.runtime``
which build the model directly and do their own ``model.update``.
Without TQ-aware module replacement those runtimes hit
``ValueError: Module does not have parameter named "experts"`` at
``model.update`` because the bundle's ``.tq_packed`` keys have no
matching parameter on the bare ``nn.Linear`` / ``SwitchLinear``.

This helper extracts just the TQ-replacement core so both runtimes
can call it before their existing affine path runs.

Contract
--------
``hydrate_jangtq(model, weights, *, mxtq_seed=42) -> regular_weights``

- ``model``: the instantiated model. The helper walks dotted attribute
  paths derived from the weight key (e.g.
  ``layers.0.mlp.experts.gate_up_proj`` → ``model.layers[0].mlp.experts.gate_up_proj``).
  Modules whose key has a sibling ``.tq_packed`` are SWAPPED in-place to
  ``TurboQuantLinear`` (2D packed) or ``TurboQuantSwitchLinear`` (3D
  packed — already stacked across the experts axis at convert time).
- ``weights``: a flat dict of name → mx.array. Keys ending in
  ``.tq_packed`` / ``.tq_norms`` / ``.tq_bits`` are CONSUMED by this
  helper. Returns a NEW dict containing only the leftover keys.
- ``mxtq_seed``: passed to TurboQuant{Linear,SwitchLinear} for codebook
  / sign generation. Default 42 matches converter convention.

The helper does NOT do per-expert stacking. Laguna and Mistral3
converters already emit pre-stacked 3D tensors. (DSV4 / Kimi /
MiniMax / Qwen3.6 use per-expert tensors and stack at hydrate time —
that path lives in the bigger ``_hydrate_jangtq_model``.)
"""
from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn

from jang_tools.turboquant.tq_kernel import (
    TurboQuantLinear,
    TurboQuantSwitchLinear,
)


def _walk_attr(root: Any, dotted: str) -> Any:
    cur = root
    for p in dotted.split("."):
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur


def _set_attr(root: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = root
    for p in parts[:-1]:
        if p.isdigit():
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = value
    else:
        setattr(cur, last, value)


def hydrate_jangtq(
    model: nn.Module,
    weights: Dict[str, Any],
    *,
    mxtq_seed: int = 42,
) -> Dict[str, Any]:
    """Replace `.tq_packed`-bearing modules in ``model`` with TurboQuant
    variants and return the regular (non-TQ) weight subset.

    The helper:
      1. Finds every key ending in ``.tq_packed`` and its paired
         ``.tq_norms`` / ``.tq_bits``.
      2. Walks ``model`` to the corresponding module via dotted-attr.
      3. Swaps that module to ``TurboQuantLinear`` (2D packed) or
         ``TurboQuantSwitchLinear`` (3D packed — pre-stacked at convert).
      4. Assigns ``.packed`` / ``.norms`` from the weight dict.
      5. Returns a new dict with TQ keys (and ``.tq_bits`` scalars) removed.

    Modules referenced by a TQ key but absent from the model are
    skipped silently — these are typically MTP-only weights that the
    converter writes out but the inference model doesn't instantiate.
    """
    # Group TQ tensors by their base path (the module they belong to).
    tq_groups: Dict[str, Dict[str, Any]] = {}
    regular: Dict[str, Any] = {}
    for k, v in weights.items():
        if k.endswith(".tq_packed"):
            base = k[: -len(".tq_packed")]
            tq_groups.setdefault(base, {})["packed"] = v
        elif k.endswith(".tq_norms"):
            base = k[: -len(".tq_norms")]
            tq_groups.setdefault(base, {})["norms"] = v
        elif k.endswith(".tq_bits"):
            base = k[: -len(".tq_bits")]
            # tq_bits is a 1-element array — coerce to int.
            tq_groups.setdefault(base, {})["bits"] = int(v[0].item())
        else:
            regular[k] = v

    n_swapped = 0
    n_skipped = 0
    for base, parts in tq_groups.items():
        packed = parts.get("packed")
        norms = parts.get("norms")
        bits = parts.get("bits")
        if packed is None or norms is None or bits is None:
            # Triplet incomplete — bundle bug; skip rather than crash so
            # the rest of the load can proceed.
            continue
        try:
            existing = _walk_attr(model, base)
        except (AttributeError, IndexError, KeyError):
            n_skipped += 1
            continue

        vals_per_u32 = 32 // bits
        if packed.ndim == 3:
            # Pre-stacked SwitchLinear: shape (n_experts, out, packed_in)
            n_experts, out_features, packed_cols = packed.shape
            in_features = packed_cols * vals_per_u32
            new_module = TurboQuantSwitchLinear(
                in_features=in_features,
                out_features=out_features,
                num_experts=n_experts,
                bits=bits,
                bias=False,
                seed=mxtq_seed,
            )
        elif packed.ndim == 2:
            # Plain Linear: shape (out, packed_in)
            out_features, packed_cols = packed.shape
            in_features = packed_cols * vals_per_u32
            new_module = TurboQuantLinear(
                in_features=in_features,
                out_features=out_features,
                bits=bits,
                bias=False,
                seed=mxtq_seed,
            )
        else:
            # Unknown shape — skip.
            n_skipped += 1
            continue

        new_module.packed = packed
        new_module.norms = norms
        _set_attr(model, base, new_module)
        # Drop the existing module's reference (nn.Linear weight tensor)
        # to free the placeholder zeros that __init__ allocated.
        del existing
        n_swapped += 1

    # Telemetry — caller can ignore but useful when debugging
    # missing-parameter errors.
    if n_swapped or n_skipped:
        try:
            print(
                f"[jangtq_hydrate] swapped={n_swapped} skipped(no-module)={n_skipped}",
                flush=True,
            )
        except Exception:
            pass

    return regular


__all__ = ["hydrate_jangtq"]
