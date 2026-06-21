"""Torch-vs-MLX numeric parity for the MiMo-V2.5 vision tower.

Loads real ``visual.*`` weights from the source checkpoint, runs one synthetic
image through BOTH the source torch ``MiMoVisionTransformer`` and the MLX
``VisionModel``, and reports cosine similarity / relative error of the merger
output. This is the gate for the window-reorder + sink + GQA port.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

import mlx.core as mx

from jang_tools.mimo_v2.weight_loader import MiMoShardIndex
from jang_tools.mimo_v2.vlm.config import VisionConfig
from jang_tools.mimo_v2.vlm.vision import VisionModel


def load_source_vision_class(src: Path):
    pkg = types.ModuleType("mimo_src_pkg")
    pkg.__path__ = [str(src)]
    sys.modules["mimo_src_pkg"] = pkg
    for stem in ("configuration_mimo_v2", "modeling_mimo_v2"):
        spec = importlib.util.spec_from_file_location(f"mimo_src_pkg.{stem}", src / f"{stem}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"mimo_src_pkg.{stem}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["mimo_src_pkg.modeling_mimo_v2"].MiMoVisionTransformer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--grid", default="1,32,32", help="t,h,w patch grid for the synthetic image")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import json

    cfg_raw = json.loads((args.src / "config.json").read_text())["vision_config"]
    vcfg = VisionConfig.from_dict(cfg_raw)

    t, h, w = (int(x) for x in args.grid.split(","))
    n_patches = t * h * w
    patch_dim = vcfg.in_chans * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size
    rng = np.random.default_rng(args.seed)
    pixels = rng.standard_normal((n_patches, patch_dim)).astype(np.float32) * 0.5
    grid_thw = [(t, h, w)]

    # --- load weights once -------------------------------------------------
    idx = MiMoShardIndex(args.src)
    visual_keys = [k for k in idx.weight_keys if k.startswith("visual.")]
    print(f"visual tensors: {len(visual_keys)}", flush=True)
    weights = {k: idx.read_passthrough(k, out_dtype=torch.float32) for k in visual_keys}

    # --- torch reference ----------------------------------------------------
    VisionClass = load_source_vision_class(args.src)
    torch_cfg = types.SimpleNamespace(**cfg_raw)
    ref = VisionClass(torch_cfg).float()
    missing, unexpected = ref.load_state_dict(
        {k[len("visual.") :]: v for k, v in weights.items()}, strict=False
    )
    print(f"torch load: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing (zeroed like HF _init_weights):", missing[:6])
        with torch.no_grad():
            sd = dict(ref.named_parameters())
            for name in missing:
                sd[name].zero_()
    ref.eval()
    with torch.no_grad():
        out_ref = ref(torch.from_numpy(pixels), torch.tensor(grid_thw)).numpy()

    # --- mlx ------------------------------------------------------------------
    model = VisionModel(vcfg)
    mlx_weights = model.sanitize(
        {k: mx.array(v.numpy()) for k, v in weights.items()}
    )
    model.load_weights([(k[len("visual.") :], v) for k, v in mlx_weights.items()], strict=True)
    out_mlx = np.array(model(mx.array(pixels), grid_thw), copy=False)

    # --- compare ----------------------------------------------------------------
    assert out_ref.shape == out_mlx.shape, (out_ref.shape, out_mlx.shape)
    a = out_ref.reshape(-1).astype(np.float64)
    b = out_mlx.reshape(-1).astype(np.float64)
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
    maxdiff = float(np.max(np.abs(a - b)))
    print(f"shape={out_ref.shape} cos={cos:.6f} rel={rel:.6f} max_abs_diff={maxdiff:.6f}")
    verdict = "PARITY_PASS" if cos > 0.999 and rel < 0.02 else "PARITY_FAIL"
    print(verdict)
    return 0 if verdict == "PARITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
