"""Inspect loaded MiMo MLX quantized module metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_lm.utils import load

from jang_tools.mimo_v2 import mlx_register  # noqa: F401


def get_path(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def describe(name: str, obj) -> None:
    print(f"\n{name}: {obj.__class__.__module__}.{obj.__class__.__name__}")
    for attr in ("bits", "group_size", "mode"):
        if hasattr(obj, attr):
            print(f"  {attr}={getattr(obj, attr)}")
    for attr in ("weight", "scales", "biases"):
        value = getattr(obj, attr, None)
        if value is not None:
            print(f"  {attr}.shape={tuple(value.shape)} dtype={value.dtype}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=[
            "model.embed_tokens",
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.1.mlp.switch_mlp.gate_proj",
            "model.layers.1.mlp.switch_mlp.up_proj",
            "model.layers.1.mlp.switch_mlp.down_proj",
            "model.layers.4.mlp.switch_mlp.gate_proj",
            "model.layers.4.mlp.switch_mlp.up_proj",
            "model.layers.4.mlp.switch_mlp.down_proj",
            "lm_head",
        ],
    )
    args = parser.parse_args()

    model, _ = load(str(args.bundle), lazy=True, tokenizer_config={"trust_remote_code": True})
    for path in args.paths:
        describe(path, get_path(model, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
