"""Convert a PyTorch ``JangDFlashDrafter`` checkpoint to an MLX-ready
safetensors file.

The Swift ``JangDFlashDrafter`` module uses the same
``@ModuleInfo(key:)`` parameter paths as the PyTorch model, so the
keys require no renaming. This script just loads the PT state dict,
casts to float16 numpy, and writes safetensors with the same layout.

Usage:
    python -m jang_tools.dflash.convert_to_mlx \\
        --ckpt /data/dflash-drafter-v1/drafter.pt \\
        --out  /data/dflash-drafter-v1/drafter.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file


def _pt_to_numpy_fp16(tensor: torch.Tensor) -> np.ndarray:
    """Best-effort conversion of an arbitrary PT tensor to fp16
    numpy. BF16 tensors must be cast to fp32 first because numpy
    lacks a native bf16 dtype."""
    t = tensor.detach().cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t.to(torch.float16).numpy()


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m jang_tools.dflash.convert_to_mlx",
        description="Convert a PyTorch JangDFlashDrafter checkpoint to MLX safetensors.",
    )
    p.add_argument("--ckpt", required=True, help="Path to the PT state dict (drafter.pt).")
    p.add_argument("--out", required=True, help="Output safetensors path.")
    p.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Output dtype for the MLX side. float16 is the widest compatibility; bfloat16 matches the runtime cast Swift does automatically.",
    )
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise SystemExit(f"expected dict state_dict in {ckpt_path}, got {type(sd).__name__}")

    out_dict: dict[str, np.ndarray] = {}
    total_bytes = 0
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            print(f"[convert] skip non-tensor key '{k}' ({type(v).__name__})", file=sys.stderr)
            continue
        # Output dtype choice: both float16 and bfloat16 go through
        # fp16 on-disk since numpy can't represent bf16. Swift's
        # JangDFlashLoader will cast back to bf16 at load time when
        # castToBF16 is true, which matches the main model loader
        # policy for MoE-friendly matmul.
        arr = _pt_to_numpy_fp16(v)
        out_dict[k] = arr
        total_bytes += arr.nbytes

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(out_dict, str(out_path))

    print(
        f"[convert] wrote {out_path} "
        f"({len(out_dict)} tensors, {total_bytes / 1e6:.1f} MB)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
