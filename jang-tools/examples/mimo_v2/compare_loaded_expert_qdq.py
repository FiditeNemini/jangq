"""Compare loaded MLX SwitchGLU expert weights against source-side QDQ.

This probes the converter/runtime boundary for pruned MiMo expert bundles.  The
bundle stores kept experts renumbered to local slots, while the source checkpoint
uses original expert ids.  For one slot, this script resolves the keep map,
dequantizes the loaded MLX tensor, and compares it to the source tensor after
the same affine QDQ used by the source-side profile probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401
from jang_tools.mimo_v2.weight_loader import MiMoShardIndex
from source_profile_probe import quant_dequant_affine


def get_path(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def rel_stats(ref: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    ref = ref.float()
    actual = actual.float()
    diff = ref - actual
    rmse = torch.sqrt(torch.mean(diff * diff))
    rms = torch.sqrt(torch.mean(ref * ref)) + 1e-12
    return float(rmse / rms), float(diff.abs().max()), float(diff.abs().mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--proj", choices=("gate_proj", "up_proj", "down_proj"), default="down_proj")
    args = parser.parse_args()

    cfg = json.loads((args.bundle / "config.json").read_text())
    keep = cfg["runtime"]["expert_keep_map"]["layers"][str(args.layer)]
    source_expert = int(keep[args.slot])
    qspec = cfg["quantization"][f"model.layers.{args.layer}.mlp.switch_mlp.{args.proj}"]
    bits = int(qspec["bits"])
    group_size = int(qspec["group_size"])

    model, _ = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    module = get_path(model, f"model.layers.{args.layer}.mlp.switch_mlp.{args.proj}")
    loaded = mx.dequantize(
        module.weight[args.slot : args.slot + 1],
        module.scales[args.slot : args.slot + 1],
        module.biases[args.slot : args.slot + 1],
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
        dtype=mx.float32,
    )[0]
    mx.eval(loaded)
    loaded_t = torch.from_numpy(np.array(loaded))

    tensor_name = f"model.layers.{args.layer}.mlp.experts.{source_expert}.{args.proj}.weight"
    source = MiMoShardIndex(args.src).read_tensor(tensor_name, out_dtype=torch.float32)
    source_qdq = quant_dequant_affine(source, bits=bits, group_size=group_size)

    rel_src, max_src, mae_src = rel_stats(source, loaded_t)
    rel_qdq, max_qdq, mae_qdq = rel_stats(source_qdq, loaded_t)
    rel_source_qdq, _, _ = rel_stats(source, source_qdq)

    print(f"bundle={args.bundle}")
    print(f"layer={args.layer} slot={args.slot} source_expert={source_expert} proj={args.proj}")
    print(f"bits={bits} group_size={group_size}")
    print(f"source_shape={tuple(source.shape)} loaded_shape={tuple(loaded_t.shape)}")
    print(f"source_vs_source_qdq rel_rmse={rel_source_qdq:.8f}")
    print(f"source_vs_loaded rel_rmse={rel_src:.8f} maxerr={max_src:.8f} mae={mae_src:.8f}")
    print(f"source_qdq_vs_loaded rel_rmse={rel_qdq:.8f} maxerr={max_qdq:.8f} mae={mae_qdq:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
