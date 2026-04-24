"""Native FP4/FP8 passthrough converter — store source quantized tensors
verbatim in bundle, dequant to bf16 at load time.

Bundle size ~180GB (matches source). Quality: bit-exact (within bf16 cast).
Use when re-quantization (via mx.quantize affine) introduces too much
error — as with DSV4's FP4 routed experts whose 16 log-spaced levels
don't fit a 4-bit linear affine codebook cleanly.

This stores tensors in MLX-native safetensors form with renamed keys
to match our MLX model. Sibling `.scale` tensors stored as fp32.
Load-time dequant happens in `load_dsv4_native.py`.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open, save_file as sf_save


def rename_key(name: str) -> str | None:
    """Map DSV4 source tensor name → MLX model sanitize key.

    Mirrors `jang_tools.dsv4.mlx_model.Model.sanitize` but in
    the convert path so the bundle already has the right keys.
    """
    if name.startswith("mtp."):
        return None  # drop MTP
    if ".compressor." in name or ".indexer." in name:
        return None  # drop CSA/HCA (unused at inference)

    if name == "embed.weight":
        return "model.embed.weight"
    if name in ("head.weight", "head.scale"):
        return name.replace("head", "lm_head")
    if name == "norm.weight":
        return "model.norm.weight"
    if name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        return f"model.{name}"

    m = re.match(r"layers\.(\d+)\.(.+)", name)
    if not m:
        return f"model.{name}"
    L, rest = m.group(1), m.group(2)
    pfx = f"model.layers.{L}"

    if rest == "attn_norm.weight":
        return f"{pfx}.input_layernorm.weight"
    if rest == "ffn_norm.weight":
        return f"{pfx}.post_attention_layernorm.weight"
    if rest.startswith("hc_"):
        return f"{pfx}.{rest}"
    if rest.startswith("attn."):
        inner = rest[len("attn."):]
        return f"{pfx}.self_attn.{inner}"
    if rest.startswith("ffn."):
        inner = rest[len("ffn."):]
        if inner.startswith("gate."):
            return f"{pfx}.mlp.gate.{inner[len('gate.'):]}"
        m2 = re.match(r"shared_experts\.(w[123])\.(.*)", inner)
        if m2:
            proj_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            return f"{pfx}.mlp.shared_experts.{proj_map[m2.group(1)]}.{m2.group(2)}"
        m3 = re.match(r"experts\.(\d+)\.(w[123])\.(.*)", inner)
        if m3:
            proj_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
            return f"{pfx}.mlp.experts.{m3.group(1)}.{proj_map[m3.group(2)]}.{m3.group(3)}"
        return f"{pfx}.mlp.{inner}"
    return f"{pfx}.{rest}"


def convert(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    idx_path = src / "model.safetensors.index.json"
    idx = json.loads(idx_path.read_text())
    wm = idx["weight_map"]

    out_map: dict[str, str] = {}
    out_buf: dict[str, torch.Tensor] = {}
    out_bytes = 0
    shard_idx = 1
    MAX_BYTES = 5_000_000_000
    t_start = time.time()

    def flush():
        nonlocal shard_idx, out_buf, out_bytes
        if not out_buf:
            return
        name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
        sf_save(out_buf, str(dst / name))
        for k in out_buf:
            out_map[k] = name
        print(f"  shard {shard_idx}: {len(out_buf)} tensors, "
              f"{out_bytes / 1e9:.2f} GB ({time.time() - t_start:.0f}s)", flush=True)
        out_buf = {}
        out_bytes = 0
        shard_idx += 1

    stats = {"fp4": 0, "fp8": 0, "passthrough": 0, "dropped": 0}
    keys = sorted(wm.keys())
    for i, src_key in enumerate(keys):
        new_key = rename_key(src_key)
        if new_key is None:
            stats["dropped"] += 1
            continue
        shard = src / wm[src_key]
        with safe_open(str(shard), framework="pt") as f:
            t = f.get_tensor(src_key)
        # Classify + rename
        if t.dtype == torch.int8:
            stats["fp4"] += 1
        elif t.dtype == torch.float8_e4m3fn:
            stats["fp8"] += 1
        elif t.dtype == torch.float8_e8m0fnu:
            # scale tensor — store as fp32 (bf16 loses exponent info)
            # Rename sibling .scale → .scale too, matching weight's new name
            t = t.float()
        else:
            stats["passthrough"] += 1
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float16)

        out_buf[new_key] = t
        out_bytes += t.numel() * t.element_size()
        if out_bytes >= MAX_BYTES:
            flush()

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(keys)}  fp4={stats['fp4']} "
                  f"fp8={stats['fp8']} pt={stats['passthrough']} "
                  f"dropped={stats['dropped']}  ({time.time() - t_start:.0f}s)",
                  flush=True)

    flush()

    # Rename shards
    total_shards = shard_idx - 1
    for k in range(1, shard_idx):
        old = dst / f"model-{k:05d}-of-XXXXX.safetensors"
        new = dst / f"model-{k:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)
    final_map = {k: v.replace("XXXXX", f"{total_shards:05d}")
                 for k, v in out_map.items()}
    total_bytes = sum((dst / fn).stat().st_size for fn in set(final_map.values()))
    (dst / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_bytes},
        "weight_map": final_map,
    }, indent=2))

    # config + jang_config
    src_cfg = json.loads((src / "config.json").read_text())
    src_cfg.pop("quantization_config", None)
    src_cfg["_name_or_path"] = "DSV4-Flash-JANG-Native"
    (dst / "config.json").write_text(json.dumps(src_cfg, indent=2))
    (dst / "jang_config.json").write_text(json.dumps({
        "weight_format": "native_fp4_fp8",
        "profile": "JANG-Native",
        "source_model": str(src),
        "description": "Source FP4/FP8 tensors preserved verbatim. "
                       "Dequant to bf16 at load time via fp4_codec / fp8_ue8m0_codec.",
    }, indent=2))

    # Aux files
    for p in src.iterdir():
        if p.is_file() and not p.name.endswith(".safetensors") \
                and p.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(p, dst / p.name)
    enc = src / "encoding"
    if enc.is_dir():
        shutil.copytree(enc, dst / "encoding", dirs_exist_ok=True)

    print(f"\n[done] {total_shards} shards, {total_bytes / 1e9:.2f} GB")
    print(f"  fp4 tensors: {stats['fp4']}  fp8: {stats['fp8']}  "
          f"passthrough: {stats['passthrough']}  dropped: {stats['dropped']}")
    print(f"  elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    args = p.parse_args()
    convert(Path(args.src), Path(args.dst))
