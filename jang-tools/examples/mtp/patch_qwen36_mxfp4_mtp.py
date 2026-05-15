#!/usr/bin/env python3
"""Build a Qwen3.6 MXFP4 bundle with native MTP tensors preserved.

The local CRACK MXFP4 bundle is a valid MLX VLM artifact, but it was built
without `mtp.*` tensors. This helper copies that known-good base, appends MTP
tensors from the exact BF16 source using native MLX MXFP4 quantization, and
stamps runtime metadata so MTP-aware runtimes can discover the preserved layer.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from jang_tools.calibrate import _load_bf16_tensor
from jang_tools.capabilities import build_capabilities, verify_directory


EOS_FIX = {248044: 248046}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with path.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def fix_eos_ids(data: dict[str, Any]) -> bool:
    changed = False

    def fix_obj(obj: dict[str, Any]) -> None:
        nonlocal changed
        value = obj.get("eos_token_id")
        if isinstance(value, int) and value in EOS_FIX:
            obj["eos_token_id"] = EOS_FIX[value]
            changed = True
        elif isinstance(value, list):
            mapped = [EOS_FIX.get(v, v) if isinstance(v, int) else v for v in value]
            new_value = []
            seen = set()
            for item in mapped:
                marker = (type(item).__name__, item)
                if marker in seen:
                    continue
                seen.add(marker)
                new_value.append(item)
            if new_value != value:
                obj["eos_token_id"] = new_value
                changed = True

    fix_obj(data)
    text_cfg = data.get("text_config")
    if isinstance(text_cfg, dict):
        fix_obj(text_cfg)
    return changed


def load_source_tensor(source_dir: Path, shard_name: str, tensor_name: str) -> np.ndarray:
    shard_path = source_dir / shard_name
    with safe_open(str(shard_path), framework="numpy") as f:
        shape = f.get_slice(tensor_name).get_shape()
        try:
            tensor = f.get_tensor(tensor_name)
            if not isinstance(tensor, np.ndarray):
                tensor = np.array(tensor)
        except (TypeError, AttributeError):
            tensor = _load_bf16_tensor(shard_path, tensor_name, shape)
    return tensor


def quantize_mxfp4(tensor: np.ndarray, *, group_size: int, bits: int) -> tuple[np.ndarray, np.ndarray]:
    if tensor.ndim != 2:
        raise ValueError(f"MXFP4 quantization expects 2D tensor, got {tensor.shape}")
    if tensor.shape[-1] % group_size != 0:
        raise ValueError(f"tensor last dim {tensor.shape[-1]} is not divisible by group_size={group_size}")
    w = mx.array(tensor.astype(np.float16))
    qw, scales = mx.quantize(w, group_size=group_size, bits=bits, mode="mxfp4")
    mx.eval(qw, scales)
    out = np.array(qw)
    sc = np.array(scales)
    del w, qw, scales
    mx.clear_cache()
    return out, sc


def copy_base(base_dir: Path, out_dir: Path, *, replace: bool) -> None:
    if out_dir.exists():
        if not replace:
            raise FileExistsError(f"{out_dir} already exists; pass --replace to rebuild it")
        shutil.rmtree(out_dir)
    shutil.copytree(base_dir, out_dir, copy_function=shutil.copy2)


def renumber_base_shards(out_dir: Path, index: dict[str, Any], final_count: int) -> dict[str, str]:
    old_to_new: dict[str, str] = {}
    for shard in sorted(set(index["weight_map"].values())):
        stem = shard.split("-of-", 1)[0]
        old_to_new[shard] = f"{stem}-of-{final_count:05d}.safetensors"

    for old, new in old_to_new.items():
        if old == new:
            continue
        (out_dir / old).rename(out_dir / new)

    index["weight_map"] = {key: old_to_new[value] for key, value in index["weight_map"].items()}
    return old_to_new


def indexed_shard_bytes(out_dir: Path, weight_map: dict[str, str]) -> int:
    return sum((out_dir / shard).stat().st_size for shard in set(weight_map.values()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/Users/eric/models/Sources/Qwen/Qwen3.6-27B"),
        help="BF16 Qwen3.6-27B source directory with native mtp.* tensors",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("/Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-CRACK"),
        help="Known-good MXFP4 base bundle to clone",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/Users/eric/models/dealign.ai/Qwen3.6-27B-MXFP4-MTP"),
        help="Output MXFP4+MTP bundle",
    )
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    source = args.source.expanduser()
    base = args.base.expanduser()
    out = args.out.expanduser()

    source_index = read_json(source / "model.safetensors.index.json")
    source_weight_map = source_index["weight_map"]
    mtp_names = sorted(name for name in source_weight_map if name.startswith("mtp."))
    if not mtp_names:
        raise SystemExit(f"source has no mtp.* tensors: {source}")

    copy_base(base, out, replace=args.replace)

    index_path = out / "model.safetensors.index.json"
    index = read_json(index_path)
    if any(name.startswith("mtp.") for name in index["weight_map"]):
        raise SystemExit(f"output/base already has mtp.* tensors: {out}")

    old_shards = sorted(set(index["weight_map"].values()))
    final_count = len(old_shards) + 1
    renumber_base_shards(out, index, final_count)

    mtp_shard_name = f"model-{final_count:05d}-of-{final_count:05d}.safetensors"
    mtp_tensors: dict[str, np.ndarray] = {}
    quantized_linears = 0
    passthrough = 0
    for name in mtp_names:
        tensor = load_source_tensor(source, source_weight_map[name], name)
        if name.endswith(".weight") and tensor.ndim == 2:
            qw, scales = quantize_mxfp4(tensor, group_size=args.group_size, bits=args.bits)
            base_name = name[:-len(".weight")]
            mtp_tensors[f"{base_name}.weight"] = qw
            mtp_tensors[f"{base_name}.scales"] = scales
            quantized_linears += 1
        else:
            mtp_tensors[name] = tensor.astype(np.float16)
            passthrough += 1
        del tensor

    save_file(mtp_tensors, str(out / mtp_shard_name), metadata={"format": "mlx"})
    for name in mtp_tensors:
        index["weight_map"][name] = mtp_shard_name

    total_bytes = indexed_shard_bytes(out, index["weight_map"])
    index.setdefault("metadata", {})["format"] = "mxfp4"
    index["metadata"]["total_size"] = total_bytes
    write_json(index_path, index)

    config_path = out / "config.json"
    config = read_json(config_path)
    config.pop("quantization_config", None)
    config["weight_format"] = "mxfp4"
    config["quantization"] = {
        "group_size": args.group_size,
        "bits": args.bits,
        "mode": "mxfp4",
    }
    fix_eos_ids(config)
    write_json(config_path, config)

    for extra in ("generation_config.json", "tokenizer_config.json"):
        path = out / extra
        if path.exists():
            data = read_json(path)
            if fix_eos_ids(data):
                write_json(path, data)

    runtime = {
        "total_weight_bytes": total_bytes,
        "total_weight_gb": round(total_bytes / (1024 ** 3), 2),
        "bundle_has_mtp": True,
        "mtp_layers": int(config.get("text_config", {}).get("mtp_num_hidden_layers", 1)),
        "mtp_mode": "preserved_enabled",
    }
    jang_config = {
        "format": "jang",
        "format_version": "2.0",
        "weight_format": "mxfp4",
        "profile": "MXFP4_MTP",
        "source_model": {
            "name": source.name,
            "architecture": config.get("model_type", "qwen3_5"),
        },
        "architecture": {
            "type": "hybrid_ssm",
            "attention": "hybrid",
            "has_vision": True,
            "has_ssm": True,
            "has_moe": False,
        },
        "quantization": {
            "method": "mxfp4",
            "group_size": args.group_size,
            "bits": args.bits,
            "mode": "mxfp4",
            "mtp_policy": "native_mxfp4_linears_fp16_norms",
        },
        "runtime": runtime,
        "mtp": {
            "kept": True,
            "enabled": True,
            "num_layers": runtime["mtp_layers"],
            "source_tensor_count": len(mtp_names),
            "runtime_tensor_count": len(mtp_tensors),
        },
        "bundle_has_mtp": True,
        "mtp_layers": runtime["mtp_layers"],
    }
    caps = build_capabilities(jang_config, config, out)
    if caps:
        jang_config["capabilities"] = caps
    write_json(out / "jang_config.json", jang_config)
    ok, msg = verify_directory(out)
    if not ok:
        raise SystemExit(f"capabilities verify failed: {msg}")

    print(json.dumps({
        "output": str(out),
        "source_mtp_tensors": len(mtp_names),
        "runtime_mtp_entries": len(mtp_tensors),
        "quantized_linears": quantized_linears,
        "passthrough_tensors": passthrough,
        "shards": final_count,
        "total_weight_bytes": total_bytes,
        "total_weight_gb": round(total_bytes / (1024 ** 3), 2),
        "verify": msg,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
