"""Estimate MiMo affine bundle size for documented candidate policies.

This reads only safetensor headers from a node-local source checkpoint. It does
not copy model weights. The estimator mirrors MLX affine storage:
packed uint32 weights plus BF16/F16 scales and biases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open


EXPERT_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\.\d+\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)


@dataclass(frozen=True)
class EstimatePolicy:
    name: str
    default_bits: int
    expert_bits: dict[str, int]
    expert_group_size: int
    qkv_bits: int
    layer0_dense_bits: int
    o_proj_bits: int | None
    token_io_bf16: bool
    early_down3: int = 0
    early_333: int = 0


def parse_policy(raw: str) -> EstimatePolicy:
    key = raw.lower().replace("_", "").replace("-", "")
    if key == "current2l222g64":
        return EstimatePolicy("current2l222g64", 8, {"gate_proj": 2, "up_proj": 2, "down_proj": 2}, 64, 8, 8, None, False)
    if key == "current2q222g64":
        return EstimatePolicy("current2q222g64", 4, {"gate_proj": 2, "up_proj": 2, "down_proj": 2}, 64, 6, 6, 4, False)
    if key == "current2f222g64":
        return EstimatePolicy("current2f222g64", 4, {"gate_proj": 2, "up_proj": 2, "down_proj": 2}, 64, 4, 4, 4, False)
    if key == "doc2l423g128":
        return EstimatePolicy("doc2l423g128", 8, {"gate_proj": 4, "up_proj": 2, "down_proj": 3}, 128, 8, 8, None, False)
    if key.startswith("doc322d3e"):
        early = int(key.removeprefix("doc322d3e"))
        return EstimatePolicy("doc322d3e" + str(early), 8, {"gate_proj": 3, "up_proj": 2, "down_proj": 2}, 128, 8, 8, None, False, early_down3=early)
    m = re.fullmatch(r"slim322d3e(\d+)(?:b([568]))?(?:q([4568]))?", key)
    if m:
        early = int(m.group(1))
        default_bits = int(m.group(2) or 4)
        qkv_bits = int(m.group(3) or 6)
        suffix = f"b{default_bits}" if default_bits != 4 else ""
        suffix += f"q{qkv_bits}" if qkv_bits != 6 else ""
        return EstimatePolicy("slim322d3e" + str(early) + suffix, default_bits, {"gate_proj": 3, "up_proj": 2, "down_proj": 2}, 128, qkv_bits, 6, 4, False, early_down3=early)
    m = re.fullmatch(r"slim333e(\d+)(?:b([568]))?(?:q([4568]))?", key)
    if m:
        early = int(m.group(1))
        default_bits = int(m.group(2) or 4)
        qkv_bits = int(m.group(3) or 6)
        suffix = f"b{default_bits}" if default_bits != 4 else ""
        suffix += f"q{qkv_bits}" if qkv_bits != 6 else ""
        return EstimatePolicy("slim333e" + str(early) + suffix, default_bits, {"gate_proj": 3, "up_proj": 2, "down_proj": 2}, 128, qkv_bits, 6, 4, False, early_333=early)
    raise ValueError(f"unknown policy {raw}")


def qbytes(shape: tuple[int, ...], bits: int, group_size: int) -> int:
    rows, cols = int(shape[0]), int(shape[1])
    groups = math.ceil(cols / group_size)
    packed = rows * groups * math.ceil(group_size * bits / 32) * 4
    sidecars = rows * groups * 2 * 2
    return packed + sidecars


def classify(name: str, policy: EstimatePolicy) -> tuple[str, int, int | None]:
    if name.endswith(".weight_scale_inv"):
        return "skip", 0, None
    if name.startswith("model.mtp."):
        return "skip", 0, None
    if name.endswith(".e_score_correction_bias"):
        return "fp32", 32, None
    if name.endswith(".mlp.gate.weight") and ".experts." not in name:
        return "fp32", 32, None
    if (
        name.endswith("norm.weight")
        or name.endswith("layernorm.weight")
        or name.endswith(".bias")
        or name.endswith("attention_sink_bias")
        or name.startswith("visual.")
        or name.startswith("audio_encoder.")
        or name.startswith("speech_embeddings.")
        or name.endswith(".eh_proj.weight")
    ):
        return "bf16", 16, None
    if policy.token_io_bf16 and name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return "bf16", 16, None
    if name.endswith(".o_proj.weight"):
        if policy.o_proj_bits is None:
            return "bf16", 16, None
        return "affine", policy.o_proj_bits, 64
    if name.endswith(".self_attn.qkv_proj.weight"):
        return "affine", policy.qkv_bits, 64
    if name.startswith("model.layers.0.mlp.") and name.endswith("_proj.weight"):
        return "affine", policy.layer0_dense_bits, 64
    m = EXPERT_RE.match(name)
    if m:
        bits = dict(policy.expert_bits)
        layer = int(m.group("layer"))
        if policy.early_down3 and 1 <= layer <= policy.early_down3:
            bits["down_proj"] = 3
        if policy.early_333 and 1 <= layer <= policy.early_333:
            bits.update({"gate_proj": 3, "up_proj": 3, "down_proj": 3})
        return "affine", bits[m.group("proj")], policy.expert_group_size
    if name.endswith(".weight"):
        return "affine", policy.default_bits, 64
    return "bf16", 16, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", required=True)
    args = parser.parse_args()

    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    shapes: dict[str, tuple[int, ...]] = {}
    for file in files:
        with safe_open(str(args.src / file), framework="pt", device="cpu") as f:
            for key in f.keys():
                shapes[key] = tuple(f.get_slice(key).get_shape())

    for raw in args.policies:
        policy = parse_policy(raw)
        total = 0
        role_bytes: dict[str, int] = {}
        for name, shape in shapes.items():
            method, bits, group_size = classify(name, policy)
            if method == "skip":
                continue
            if method == "affine":
                b = qbytes(shape, bits, int(group_size or 64))
            elif method == "fp32":
                b = math.prod(shape) * 4
            else:
                b = math.prod(shape) * 2
            total += b
            role = "experts" if EXPERT_RE.match(name) else "other"
            role_bytes[role] = role_bytes.get(role, 0) + b
        print(
            f"{policy.name}: {total / 1e9:.3f} GB / {total / (1024 ** 3):.3f} GiB "
            f"experts={role_bytes.get('experts', 0) / 1e9:.3f} GB other={role_bytes.get('other', 0) / 1e9:.3f} GB"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
