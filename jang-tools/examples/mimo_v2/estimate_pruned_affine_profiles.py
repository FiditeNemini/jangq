"""Header-only size estimator for pruned MiMo affine profiles."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from safetensors import safe_open

from jang_tools.mimo_v2.convert_jang import QuantProfile, classify


EXPERT_RE = re.compile(
    r"model\.layers\.\d+\.mlp\.experts\.(?P<expert>\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)


def qbytes(shape: tuple[int, ...], bits: int, group_size: int) -> int:
    rows, cols = int(shape[0]), int(shape[1])
    groups = math.ceil(cols / group_size)
    packed = rows * groups * math.ceil(group_size * bits / 32) * 4
    sidecars = rows * groups * 2 * 2
    return packed + sidecars


def tensor_bytes(name: str, shape: tuple[int, ...], profile: QuantProfile) -> tuple[str, int] | None:
    bits, method, group_size = classify(name, profile)
    if method == "skip":
        return None
    if method == "affine":
        return method, qbytes(shape, bits, group_size)
    if method == "passthrough_fp32":
        return method, math.prod(shape) * 4
    return method, math.prod(shape) * 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", required=True)
    parser.add_argument("--keep-experts", nargs="+", type=int, required=True)
    args = parser.parse_args()

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    shapes: dict[str, tuple[int, ...]] = {}
    for file in sorted(set(index["weight_map"].values())):
        with safe_open(str(args.src / file), framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())

    for raw in args.profiles:
        profile = QuantProfile.parse(raw)
        for keep in args.keep_experts:
            total = 0
            expert = 0
            other = 0
            for name, shape in shapes.items():
                m = EXPERT_RE.match(name)
                if m and int(m.group("expert")) >= keep:
                    continue
                item = tensor_bytes(name, shape, profile)
                if item is None:
                    continue
                _, size = item
                total += size
                if m:
                    expert += size
                else:
                    other += size
            print(
                f"{profile.name} keep{keep}: {total / 1e9:.3f} GB / "
                f"{total / (1024 ** 3):.3f} GiB experts={expert / 1e9:.3f} "
                f"other={other / 1e9:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
