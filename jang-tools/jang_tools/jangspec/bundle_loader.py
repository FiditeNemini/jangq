"""
Bundle weight loader: read a .jangspec bundle and produce the
{tensor_name: mx.array} dict that mlx-lm models expect.

This is the inverse of `jang_tools.jangspec.builder.JangSpecBuilder`. The
builder splits a source JANG model into hot_core.safetensors + per-expert
blobs; this module recombines them back into the canonical layout that
mlx-lm's `load_weights_from_safetensors` would produce if pointed at the
original directory.

Used by:
- Plan 5's bundle validation script (Python token-equality check).
- Future Plan 6 Swift port (as a Python reference oracle when debugging).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import mlx.core as mx
import numpy as np
from safetensors.numpy import load_file

from . import format as fmt
from .reader import JangSpecReader

_LAYER_RE = re.compile(r"\.?layers\.(\d+)\.")


def _layer_idx(base_name: str) -> int:
    m = _LAYER_RE.search(base_name)
    if not m:
        raise ValueError(f"cannot parse layer index from {base_name!r}")
    return int(m.group(1))


def load_weights_from_bundle(bundle_dir: Path | str) -> Dict[str, "mx.array"]:
    """Load every tensor a model needs from a `.jangspec` bundle.

    The returned dict has the same keys and dtypes you would get from
    `mx.load("model.safetensors")` on the source JANG_xxx directory: the
    hot-core tensors copied through unchanged, plus per-expert blobs
    restacked into 3D `[E, ...]` tensors under their original
    `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` names.

    Memory: hot-core tensors are mmap'd (zero-copy). Expert stacks are
    materialized as new mx.array instances because `mx.stack` over many
    blob slices is required to produce a contiguous tensor.
    """

    bundle_dir = Path(bundle_dir)
    reader = JangSpecReader(bundle_dir)

    out: Dict[str, mx.array] = {}

    # 1. Hot core — mmap'd safetensors, copy keys directly.
    hot_core_path = bundle_dir / fmt.HOT_CORE_FILENAME
    if not hot_core_path.exists():
        raise FileNotFoundError(f"missing {hot_core_path}")
    hot_np = load_file(str(hot_core_path))
    for name, arr in hot_np.items():
        out[name] = mx.array(arr)

    # 2. Per-layer expert restack. Iterate every expert tensor base name
    #    in the manifest, group by layer, then for each (layer, base) pair
    #    walk the experts in order and stack their qweight/scales/biases.
    manifest = reader.manifest
    by_layer: dict[int, list[str]] = {}
    for base in manifest.expert_tensor_names:
        by_layer.setdefault(_layer_idx(base), []).append(base)

    for layer_idx, base_names in by_layer.items():
        # Each layer has 3 base names (gate_proj, up_proj, down_proj). We
        # iterate experts once per layer and dispatch to all 3 bases as
        # we go — avoids re-loading the same blob 3 times.
        buffers: dict[str, dict[str, list[np.ndarray]]] = {
            base: {"weight": [], "scales": [], "biases": []} for base in base_names
        }
        # Map kind name fragment -> the matching base name in this layer.
        kind_to_base: dict[str, str] = {}
        for base in base_names:
            for kind in ("gate_proj", "up_proj", "down_proj"):
                if base.endswith(f".switch_mlp.{kind}"):
                    kind_to_base[kind] = base

        for expert_id in range(reader.n_experts_per_layer):
            blob = reader.load_expert(layer_idx=layer_idx, expert_id=expert_id)
            # `blob.tensors` is an ExpertTensors dataclass with .gate/.up/.down,
            # each a (qweight, scales, biases) numpy triple.
            for kind_name, triple in (
                ("gate_proj", blob.tensors.gate),
                ("up_proj", blob.tensors.up),
                ("down_proj", blob.tensors.down),
            ):
                base = kind_to_base.get(kind_name)
                if base is None:
                    continue
                qw, sc, bi = triple
                buffers[base]["weight"].append(qw)
                buffers[base]["scales"].append(sc)
                buffers[base]["biases"].append(bi)

        # Stack and emit as mx.array.
        for base in base_names:
            stacked_w = np.stack(buffers[base]["weight"], axis=0)
            stacked_s = np.stack(buffers[base]["scales"], axis=0)
            stacked_b = np.stack(buffers[base]["biases"], axis=0)
            out[f"{base}.weight"] = mx.array(stacked_w)
            out[f"{base}.scales"] = mx.array(stacked_s)
            out[f"{base}.biases"] = mx.array(stacked_b)

    reader.close()
    return out


def load_jang_model_from_bundle(bundle_dir: Path | str):
    """Load an mlx-lm model from a `.jangspec` bundle.

    Mirrors `jang_tools.loader.load_jang_model` / `load_jang_vlm_model` but
    sources weights from the bundle's `bundle_loader.load_weights_from_bundle`
    helper instead of the source JANG_xxx directory's shards. Intended for
    the Plan 5 validation script.

    Returns: (model, tokenizer)
    """
    from transformers import AutoTokenizer

    bundle_dir = Path(bundle_dir)
    target_dir = bundle_dir / "target"

    # The model factory needs config.json + jang_config.json. Both are
    # copied verbatim into target/ at bundle build time (see
    # JangSpecBuilder._copy_tokenizer).
    if not (target_dir / "config.json").exists():
        raise FileNotFoundError(
            f"bundle is missing target/config.json — built with an older builder?"
        )

    # Load weights via the bundle reader.
    weights = load_weights_from_bundle(bundle_dir)

    # Build the model skeleton via mlx-lm's factory using config.json.
    import json
    config = json.loads((target_dir / "config.json").read_text())

    # Pick the right factory: VLM if model has vision/audio fields,
    # plain LLM otherwise. Gemma-4 is multimodal, so VLM.
    is_vlm = any(
        k in config for k in ("vision_config", "vision_tower", "audio_config")
    ) or "Conditional" in str(config.get("architectures", []))

    if is_vlm:
        from mlx_vlm.utils import load_model
        result = load_model(target_dir, lazy=True)
    else:
        from mlx_lm.utils import load_model
        result = load_model(target_dir, lazy=True)

    # Some versions return (model, config) tuples; others return just model.
    if isinstance(result, tuple):
        model = result[0]
    else:
        model = result

    # Apply the model's sanitize step (renames switch_mlp keys, etc.).
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Convert the weight dict to (key, value) pairs and load.
    model.load_weights(list(weights.items()), strict=False)

    tokenizer = AutoTokenizer.from_pretrained(str(bundle_dir))
    return model, tokenizer
