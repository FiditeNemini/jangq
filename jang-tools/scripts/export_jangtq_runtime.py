#!/usr/bin/env python3
"""
Export a JANGTQ runtime sidecar (signs + codebooks) for the Swift engine.

The Swift JANGTQLoader does NOT regenerate the Hadamard signs (PCG64-based)
or the Lloyd-Max codebooks. Both are deterministic functions of
(in_features, seed) and (in_features, bits) respectively, computed at
quantization time in NumPy/MLX. To avoid reimplementing them bit-exact in
Swift, we ship them as a sidecar safetensors file alongside the model.

This script walks a JANGTQ model directory, finds every unique
in_features used by the routed-expert TurboQuant matrices, computes the
corresponding signs/codebook with the same Python helpers the inference
loader uses, and writes them as named tensors to
`jangtq_runtime.safetensors` next to the model files.

Usage:
    python3 export_jangtq_runtime.py /path/to/MiniMax-M2.7-JANGTQ_2L

Tensor naming (matches Swift loader's expectations):
    signs.{in_features}.{seed}     : float32, shape (in_features,)
    codebook.{in_features}.{bits}  : float32, shape (2^bits,)

The Python loader does not read this sidecar — it computes signs/codebooks
on the fly via in-process caches. The sidecar exists purely so the Swift
engine can avoid having to reimplement NumPy's PCG64 + Lloyd-Max iteration.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx

from jang_tools.turboquant.codebook import compute_codebook
from jang_tools.turboquant.rotation import generate_random_signs


def _walk_for_in_features(model_dir: Path) -> set[int]:
    """Find every unique in_features used by routed-expert .tq_packed tensors.

    Per-expert packed shape is (out_features, in_features / 16) for 2-bit.
    """
    in_features_set: set[int] = set()
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        weights = mx.load(str(shard))
        for k, v in weights.items():
            if not k.endswith(".tq_packed"):
                continue
            shape = list(v.shape)
            if len(shape) != 2:
                continue
            in_features_set.add(shape[1] * 16)
    return in_features_set


def export(model_dir: Path) -> Path:
    config_path = model_dir / "jang_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    cfg = json.loads(config_path.read_text())
    if cfg.get("weight_format") != "mxtq":
        raise ValueError(
            f"{model_dir.name}: weight_format is {cfg.get('weight_format')!r}, expected 'mxtq'"
        )
    seed = int(cfg.get("mxtq_seed", 42))
    bits_routed = int(cfg.get("mxtq_bits", {}).get("routed_expert", 2))

    print(f"Scanning {model_dir.name}...", flush=True)
    in_features = sorted(_walk_for_in_features(model_dir))
    print(f"  routed_expert in_features: {in_features}")
    print(f"  seed = {seed}, bits = {bits_routed}")

    out_tensors: dict[str, mx.array] = {}
    for in_f in in_features:
        signs_key = f"signs.{in_f}.{seed}"
        cb_key = f"codebook.{in_f}.{bits_routed}"
        signs = generate_random_signs(in_f, seed=seed).astype(mx.float32)
        codebook = mx.array(compute_codebook(in_f, bits_routed), dtype=mx.float32)
        mx.eval(signs, codebook)
        out_tensors[signs_key] = signs
        out_tensors[cb_key] = codebook
        print(f"  + {signs_key} (shape={list(signs.shape)})")
        print(f"  + {cb_key} (shape={list(codebook.shape)})")

    out_path = model_dir / "jangtq_runtime.safetensors"
    mx.save_safetensors(str(out_path), out_tensors)
    print(f"\nWrote {out_path} ({len(out_tensors)} tensors)")
    return out_path


def main():
    if len(sys.argv) < 2:
        print("Usage: export_jangtq_runtime.py <model_dir>", file=sys.stderr)
        sys.exit(1)
    model_dir = Path(sys.argv[1]).expanduser().resolve()
    if not model_dir.is_dir():
        print(f"not a directory: {model_dir}", file=sys.stderr)
        sys.exit(1)
    export(model_dir)


if __name__ == "__main__":
    main()
