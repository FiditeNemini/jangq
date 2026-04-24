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

    # Kimi K2.6 (and other recent VLMs) ship a custom processor
    # (`kimi_k25_processor.py` etc.) that AutoProcessor refuses to load
    # without trust_remote_code. Bundles in this loader are opt-in — we're
    # already executing their safetensors — so forward trust_remote_code=True.
    processor = load_processor(
        model_path, add_generation_prompt=True, trust_remote_code=True,
    )
    _install_video_fallback(processor)
    return model, processor, config, model_config


def _install_video_fallback(processor):
    """Enable video inputs without torchvision.

    transformers' AutoVideoProcessor requires torchvision, which is not in
    the bundled Python vMLX ships. Qwen3VLProcessor.__call__ then falls back
    to ``self.image_processor(videos=videos)`` which raises TypeError because
    Qwen2VLImageProcessor's __call__ has no ``videos`` kwarg.

    This adapter routes video frames through image_processor (one image per
    frame) and promotes the resulting grid to video_grid_thw. Temporal merging
    (temporal_patch_size) is preserved by rewriting the grid's time axis so
    the <|video_pad|> expansion in the chat template produces the correct
    token count. Callers still pass ``videos=[[frame1, frame2, ...]]`` — the
    wrapper then converts internally.
    """
    if getattr(processor, "video_processor", None) is not None:
        return
    if not hasattr(processor, "image_processor"):
        return

    ip = processor.image_processor
    proc_cls = processor.__class__
    orig_call = proc_cls.__call__
    if getattr(orig_call, "_jangtq_video_fallback", False):
        return  # already patched on this class

    temporal_patch = int(getattr(ip, "temporal_patch_size", 2) or 2)

    def _patched_call(self, images=None, text=None, videos=None, **kwargs):
        if videos is None or self.video_processor is not None:
            return orig_call(self, images=images, text=text, videos=videos, **kwargs)

        # Lift each video's frames into a flat image batch, one row per frame.
        # image_processor produces image_grid_thw shape (sum_frames, 3) with t=1.
        # We collapse each video's rows into a single row with
        # t = ceil(n_frames / temporal_patch_size), keeping h, w from the
        # first frame. Preserves total patch count when n_frames % tp == 0.
        import mlx.core as _mx
        import numpy as _np

        flat_frames = []
        frames_per_video = []
        for v in videos:
            fs = v if isinstance(v, (list, tuple)) else [v]
            flat_frames.extend(fs)
            frames_per_video.append(len(fs))

        image_out = ip(images=flat_frames)
        pv = image_out["pixel_values"]           # (sum_patches, D)
        grid = image_out["image_grid_thw"]       # (sum_frames, 3)  t=1 each

        # Build video_grid_thw per video by merging frames along t-axis.
        g_np = grid if isinstance(grid, _np.ndarray) else _np.asarray(grid)
        video_rows = []
        cursor = 0
        for n in frames_per_video:
            if n == 0:
                continue
            frame_rows = g_np[cursor:cursor + n]  # (n, 3)
            cursor += n
            h, w = int(frame_rows[0, 1]), int(frame_rows[0, 2])
            t_patches = max(1, (n + temporal_patch - 1) // temporal_patch)
            video_rows.append([t_patches, h, w])
        v_grid = _np.asarray(video_rows, dtype=_np.int64)

        videos_inputs_synth = {
            "pixel_values_videos": _mx.array(pv) if not isinstance(pv, _mx.array) else pv,
            "video_grid_thw": _mx.array(v_grid),
        }

        # Splice synthesized video_inputs into the parent __call__ by keeping
        # videos=None (so original path skips real video_processor) and
        # post-merging. Simplest: call parent with images only, then merge.
        base = orig_call(self, images=images, text=text, videos=None, **kwargs)

        # Token-expansion for <|video_pad|>: same math as the real path.
        # If `base` already tokenized text without video markers, we must
        # re-tokenize with expanded markers. Easiest: do the text-mutation
        # here before calling orig_call, not after. Restart the call.
        #
        # Because tokenization happens inside orig_call, we re-enter with
        # the pre-expanded text.
        if text is not None:
            texts = text if isinstance(text, list) else [text]
            mutated = []
            merge_len = ip.merge_size ** 2
            idx = 0
            for t_str in texts:
                s = t_str
                while self.video_token in s and idx < len(video_rows):
                    num_video_tokens = int(_np.prod(v_grid[idx])) // merge_len
                    s = s.replace(self.video_token,
                                  "<|placeholder|>" * num_video_tokens, 1)
                    idx += 1
                s = s.replace("<|placeholder|>", self.video_token)
                mutated.append(s)
            # Re-tokenize with expanded markers, drop video token-expansion
            # that orig_call would have done (we already did it).
            tok_kwargs = dict(kwargs)
            tok_kwargs.pop("return_tensors", None)
            tokenized = self.tokenizer(mutated, **tok_kwargs)
            # Re-wrap as BatchFeature like orig_call does.
            from transformers.feature_extraction_utils import BatchFeature
            from mlx_vlm.models.qwen3_vl.processing_qwen3_vl import to_mlx
            merged = {**tokenized, **videos_inputs_synth}
            if images is not None:
                # image_inputs already merged into `base`; copy the image keys.
                for k in ("pixel_values", "image_grid_thw"):
                    if k in base:
                        merged[k] = base[k]
            return BatchFeature(data=to_mlx(merged))

        # No text — just merge and return
        for k, v in videos_inputs_synth.items():
            base[k] = v
        return base

    _patched_call._jangtq_video_fallback = True
    proc_cls.__call__ = _patched_call


def load_jangtq_vlm_model(model_path) -> Tuple[nn.Module, object]:
    """Load JANGTQ VLM (model + processor)."""
    model_path = Path(model_path)
    # M125 (iter 48): context-manage reads so fds close deterministically.
    with open(model_path / "config.json") as f:
        config = json.load(f)
    jang_cfg_path = model_path / "jang_config.json"
    if jang_cfg_path.exists():
        with open(jang_cfg_path) as f:
            jang_cfg = json.load(f)
    else:
        jang_cfg = {}
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
