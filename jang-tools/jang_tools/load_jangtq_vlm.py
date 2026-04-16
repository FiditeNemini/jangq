"""Load a JANGTQ-quantized VLM (e.g., Qwen 3.5/3.6 Vision MoE).

Sibling of `load_jangtq.load_jangtq_model` but builds the model skeleton via
`mlx_vlm` so the vision_tower + processor are wired up. Returns a tuple
`(model, processor)` ready for `mlx_vlm.generate`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

# We import the heavy lifters from load_jangtq so behavior stays in one place.
# Anything that touches model internals (TQ replacement, SwitchGLU monkey
# patch, router compile, MLA-bits fix, P18 QKV fusion, _fix_quantized_bits)
# is re-used by calling a helper carved out of load_jangtq_model.


def _mlx_vlm_skeleton(model_path: Path):
    """Build the mlx_vlm model + processor without loading weights.

    Replicates the relevant parts of `mlx_vlm.utils.load_model` and
    `mlx_vlm.utils.load` up to the point where weights would be loaded,
    plus the `nn.quantize` step (without it, affine-quantized modules like
    `embed_tokens`, `q_proj`, `shared_expert.*` would stay as plain
    Embedding/Linear and silently consume packed uint32 weights as if they
    were floats — producing 512-d embeddings instead of 2048-d).
    """
    from mlx_vlm.utils import (
        load_config, get_model_and_args, update_module_configs,
        load_processor,
    )
    config = load_config(model_path)

    model_class, _ = get_model_and_args(config=config)

    config.setdefault("text_config", config.pop("llm_config", {}))
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    model_config = model_class.ModelConfig.from_dict(config)
    model_config = update_module_configs(
        model_config, model_class, config,
        ["text", "vision", "perceiver", "projector", "audio"],
    )

    model = model_class.Model(model_config)

    # Apply nn.quantize for affine-quantized modules. Use the same
    # class_predicate pattern as mlx_vlm.utils.load_model: only quantize
    # modules whose `.scales` key actually exists in the on-disk weights.
    # TQ-replaced switch_mlp modules will be skipped (no `.scales`); they
    # get handled later by `_hydrate_jangtq_model`'s TurboQuant pass.
    quantization = config.get("quantization")
    if quantization is not None:
        from safetensors import safe_open
        weight_files = sorted(model_path.glob("model-*.safetensors"))
        # Cheap weight-key index — just enumerate keys, don't materialize tensors.
        weight_keys = set()
        for wf in weight_files:
            with safe_open(str(wf), framework="numpy") as f:
                weight_keys.update(f.keys())

        # mlx_vlm uses a shared sanitize-aware key-rewrite. Our artifact is
        # already in post-sanitize form so the keys we'll see are
        # `language_model.model.embed_tokens.weight` etc. The class_predicate
        # receives the *module path*, which mirrors the post-sanitize key.
        def get_class_predicate(p, m):
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            return f"{p}.scales" in weight_keys

        nn.quantize(
            model,
            group_size=quantization.get("group_size", 64),
            bits=quantization.get("bits", 2),
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )

    processor = load_processor(model_path, add_generation_prompt=True)
    return model, processor, config, model_config


def load_jangtq_vlm_model(model_path) -> Tuple[nn.Module, object]:
    """Load JANGTQ VLM (model + processor)."""
    model_path = Path(model_path)
    config = json.load(open(model_path / "config.json"))
    jang_cfg_path = model_path / "jang_config.json"
    jang_cfg = json.load(open(jang_cfg_path)) if jang_cfg_path.exists() else {}
    mxtq_seed = jang_cfg.get("mxtq_seed", 42)
    mxtq_bits_map = jang_cfg.get("mxtq_bits", {})

    print(f"Loading JANGTQ VLM: {model_path.name}", flush=True)
    print(f"  seed={mxtq_seed}, bits_map={mxtq_bits_map}", flush=True)

    model, processor, _, model_config = _mlx_vlm_skeleton(model_path)
    print(f"  Built mlx_vlm skeleton: {type(model).__name__}", flush=True)
    print(f"  vision_tower depth: {model.vision_tower.config.depth} layers", flush=True)
    print(f"  language_model layers: {model.config.text_config.num_hidden_layers}", flush=True)

    # Now apply the same TQ + quantization treatment as load_jangtq_model.
    # We delegate to a helper inside load_jangtq that does everything *except*
    # building the skeleton.
    from jang_tools.load_jangtq import _hydrate_jangtq_model
    _hydrate_jangtq_model(
        model=model,
        model_path=model_path,
        mxtq_seed=mxtq_seed,
        mxtq_bits_map=mxtq_bits_map,
        model_config=model_config,
    )
    return model, processor
