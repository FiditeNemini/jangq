"""Compare MiMo source, bundle affine sidecars, and candidate qdq codecs.

This diagnostic is for MiMo coherence bring-up. It reads one source tensor, the
matching converted bundle affine triplet, and reports relative error for:

- the bundle sidecars as loaded/dequantized by MLX
- MLX ``mx.quantize`` round-trip on the source tensor
- the CPU min/max round-trip used by ``source_profile_probe.py``

Use it to decide whether a failure is codec/packing-related before building a
new full bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from safetensors import safe_open

from jang_tools.mimo_v2.weight_loader import MiMoShardIndex
from source_profile_probe import quant_dequant_affine


def _rel_stats(ref: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    ref_f = ref.float()
    act_f = actual.float()
    diff = ref_f - act_f
    rmse = torch.sqrt(torch.mean(diff * diff))
    rms = torch.sqrt(torch.mean(ref_f * ref_f)) + 1e-12
    return float(rmse / rms), float(diff.abs().max()), float(torch.mean(diff.abs()))


def _read_bundle_triplet(bundle: Path, tensor: str):
    idx = (bundle / "model.safetensors.index.json").read_text()
    import json

    weight_map = json.loads(idx)["weight_map"]
    base = tensor[: -len(".weight")] if tensor.endswith(".weight") else tensor
    keys = [f"{base}.weight", f"{base}.scales", f"{base}.biases"]
    out = []
    for key in keys:
        shard = bundle / weight_map[key]
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            out.append(f.get_tensor(key))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--bits", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    loader = MiMoShardIndex(args.src)
    source = loader.read_tensor(args.tensor, out_dtype=torch.float32)

    b_weight, b_scales, b_biases = _read_bundle_triplet(args.bundle, args.tensor)
    bundle_dq = mx.dequantize(
        mx.array(np.array(b_weight)),
        mx.array(b_scales.float().numpy()),
        mx.array(b_biases.float().numpy()),
        group_size=args.group_size,
        bits=args.bits,
        mode="affine",
        dtype=mx.float32,
    )
    bundle_t = torch.from_numpy(np.array(bundle_dq))

    source_mx = mx.array(source.numpy())
    mx_qw, mx_qs, mx_qb = mx.quantize(source_mx, group_size=args.group_size, bits=args.bits)
    mx_dq = mx.dequantize(
        mx_qw,
        mx_qs,
        mx_qb,
        group_size=args.group_size,
        bits=args.bits,
        mode="affine",
        dtype=mx.float32,
    )
    mx_t = torch.from_numpy(np.array(mx_dq))

    cpu_t = quant_dequant_affine(source, bits=args.bits, group_size=args.group_size)

    print(f"tensor={args.tensor}")
    print(f"shape={tuple(source.shape)} bits={args.bits} group_size={args.group_size}")
    for label, tensor in (
        ("bundle", bundle_t),
        ("mx_quantize", mx_t),
        ("cpu_minmax", cpu_t),
    ):
        rel, maxerr, mae = _rel_stats(source, tensor)
        print(f"{label}: rel_rmse={rel:.6f} maxerr={maxerr:.6f} mae={mae:.6f}")
    rel, maxerr, mae = _rel_stats(mx_t, bundle_t)
    print(f"bundle_vs_mx_quantize: rel_rmse={rel:.6f} maxerr={maxerr:.6f} mae={mae:.6f}")
    rel, maxerr, mae = _rel_stats(cpu_t, bundle_t)
    print(f"bundle_vs_cpu_minmax: rel_rmse={rel:.6f} maxerr={maxerr:.6f} mae={mae:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
