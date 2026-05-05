"""Pad MiniMax JANGTQ expert counts to a router-friendly boundary.

MiniMax-M2.7-Small was emitted with 154 routed experts. That shape is valid,
but it is a poor decode shape for the per-token router/top-k path. This
module migrates an existing artifact by rewriting the affected safetensors
shards in-place:

* gate.weight rows are zero-padded
* e_score_correction_bias is padded with a large negative value
* inert zeroed TQ tensors are added for dummy experts
* config.json and jang_config.json record the padded expert count

No runtime monkeypatch is required; the resulting model is a normal JANGTQ
artifact whose config and tensor shapes agree.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


EXPERT_PAD_MULTIPLE = 32
DUMMY_BIAS = -10000.0
PROJECTIONS = ("w1", "w2", "w3")
SUFFIXES = ("tq_packed", "tq_norms", "tq_bits")


_DTYPE_TO_NUMPY = {
    "F16": np.float16,
    "BF16": np.uint16,
    "F32": np.float32,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
    "I16": np.int16,
    "U32": np.uint32,
    "I32": np.int32,
    "U64": np.uint64,
    "I64": np.int64,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _pad_count(n_experts: int, multiple: int) -> int:
    return ((n_experts + multiple - 1) // multiple) * multiple


def _dtype_size(dtype: str) -> int:
    if dtype in {"F64", "I64", "U64"}:
        return 8
    if dtype in {"F32", "I32", "U32"}:
        return 4
    if dtype in {"F16", "BF16", "I16", "U16"}:
        return 2
    if dtype in {"U8", "I8", "BOOL"}:
        return 1
    raise ValueError(f"Unsupported safetensors dtype: {dtype}")


def _tensor_nbytes(shape: list[int], dtype: str) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n * _dtype_size(dtype)


def _index_tensors(model_dir: Path, weight_map: dict[str, str]) -> dict[str, tuple[str, list[int], str]]:
    """Return tensor -> (shard, shape, dtype) from safetensors headers only."""
    by_shard: dict[str, list[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        by_shard[shard].append(key)

    out: dict[str, tuple[str, list[int], str]] = {}
    for shard in sorted(by_shard):
        with safe_open(model_dir / shard, framework="np") as f:
            for key in by_shard[shard]:
                if key not in f.keys():
                    raise KeyError(f"{key} mapped to {shard} but missing from shard")
                sl = f.get_slice(key)
                out[key] = (shard, list(sl.get_shape()), sl.get_dtype())
    return out


def _dummy_tensor(shape: list[int], dtype: str, suffix: str, bits: int) -> np.ndarray:
    np_dtype = _DTYPE_TO_NUMPY.get(dtype)
    if np_dtype is None:
        raise ValueError(f"Unsupported dtype for dummy tensor: {dtype}")
    if suffix == "tq_bits":
        return np.array([bits], dtype=np_dtype)
    return np.zeros(tuple(shape), dtype=np_dtype)


def _estimate_added_bytes(
    n_layers: int,
    n_original: int,
    n_padded: int,
    hidden_size: int,
    tensor_info: dict[str, tuple[str, list[int], str]],
) -> int:
    pad = n_padded - n_original
    gate_bytes = n_layers * pad * hidden_size * 2
    bias_bytes = n_layers * pad * 2
    dummy_bytes = 0
    ref_expert = n_original - 1
    for layer in range(n_layers):
        for proj in PROJECTIONS:
            for suffix in SUFFIXES:
                key = (
                    f"model.layers.{layer}.block_sparse_moe.experts."
                    f"{ref_expert}.{proj}.{suffix}"
                )
                _, shape, dtype = tensor_info[key]
                dummy_bytes += pad * _tensor_nbytes(shape, dtype)
    return gate_bytes + bias_bytes + dummy_bytes


def migrate(model_dir: Path, apply: bool = False) -> dict[str, Any]:
    model_dir = model_dir.expanduser().resolve()
    config_path = model_dir / "config.json"
    jang_path = model_dir / "jang_config.json"
    index_path = model_dir / "model.safetensors.index.json"

    config = _read_json(config_path)
    jang_config = _read_json(jang_path)
    index = _read_json(index_path)
    weight_map: dict[str, str] = dict(index["weight_map"])

    if config.get("model_type") != "minimax_m2":
        raise ValueError(f"Expected model_type=minimax_m2, got {config.get('model_type')!r}")
    if jang_config.get("weight_format") != "mxtq":
        raise ValueError("Expected a JANGTQ/MXTQ artifact")

    n_layers = int(config.get("num_hidden_layers", 62))
    n_original = int(config.get("num_local_experts", 0))
    n_padded = _pad_count(n_original, EXPERT_PAD_MULTIPLE)
    if n_original <= 0:
        raise ValueError("config.json is missing num_local_experts")
    if n_original == n_padded:
        return {
            "model_dir": str(model_dir),
            "already_aligned": True,
            "num_local_experts": n_original,
            "padded_num_local_experts": n_padded,
            "touched_shards": 0,
            "added_bytes": 0,
        }

    hidden_size = int(config.get("hidden_size", 3072))
    bits_cfg = jang_config.get("mxtq_bits", {})
    bits = int(bits_cfg.get("routed_expert", config.get("mxtq_bits", 2)))
    pad = n_padded - n_original
    ref_expert = n_original - 1

    tensor_info = _index_tensors(model_dir, weight_map)
    additions_by_shard: dict[str, list[tuple[str, list[int], str, str]]] = defaultdict(list)
    new_weight_map = dict(weight_map)

    touched_shards: set[str] = set()
    for layer in range(n_layers):
        gate = f"model.layers.{layer}.block_sparse_moe.gate.weight"
        bias = f"model.layers.{layer}.block_sparse_moe.e_score_correction_bias"
        for key in (gate, bias):
            if key not in weight_map:
                raise KeyError(f"Missing router tensor: {key}")
            touched_shards.add(weight_map[key])

        for proj in PROJECTIONS:
            for suffix in SUFFIXES:
                ref_key = (
                    f"model.layers.{layer}.block_sparse_moe.experts."
                    f"{ref_expert}.{proj}.{suffix}"
                )
                if ref_key not in tensor_info:
                    raise KeyError(f"Missing reference tensor: {ref_key}")
                shard, shape, dtype = tensor_info[ref_key]
                touched_shards.add(shard)
                for expert in range(n_original, n_padded):
                    new_key = (
                        f"model.layers.{layer}.block_sparse_moe.experts."
                        f"{expert}.{proj}.{suffix}"
                    )
                    if new_key in weight_map:
                        raise ValueError(f"Refusing to overwrite existing tensor {new_key}")
                    additions_by_shard[shard].append((new_key, shape, dtype, suffix))
                    new_weight_map[new_key] = shard

    added_bytes = _estimate_added_bytes(
        n_layers=n_layers,
        n_original=n_original,
        n_padded=n_padded,
        hidden_size=hidden_size,
        tensor_info=tensor_info,
    )

    summary = {
        "model_dir": str(model_dir),
        "already_aligned": False,
        "num_local_experts": n_original,
        "padded_num_local_experts": n_padded,
        "pad_experts": pad,
        "touched_shards": len(touched_shards),
        "added_bytes": added_bytes,
        "added_gb": round(added_bytes / 1_000_000_000, 3),
    }
    if not apply:
        return summary

    temp_paths: dict[str, Path] = {}
    try:
        for shard_i, shard in enumerate(sorted(touched_shards), start=1):
            src = model_dir / shard
            tmp = model_dir / f".{shard}.padtmp"
            tensors: dict[str, np.ndarray] = {}
            with safe_open(src, framework="np") as f:
                metadata = f.metadata()
                for key in f.keys():
                    arr = f.get_tensor(key)
                    if key.endswith(".block_sparse_moe.gate.weight") and key in weight_map:
                        pad_rows = np.zeros((pad, arr.shape[1]), dtype=arr.dtype)
                        arr = np.concatenate([arr, pad_rows], axis=0)
                    elif key.endswith(".block_sparse_moe.e_score_correction_bias") and key in weight_map:
                        pad_bias = np.full((pad,), DUMMY_BIAS, dtype=arr.dtype)
                        arr = np.concatenate([arr, pad_bias], axis=0)
                    tensors[key] = arr

            for key, shape, dtype, suffix in additions_by_shard.get(shard, []):
                tensors[key] = _dummy_tensor(shape, dtype, suffix, bits)

            save_file(tensors, str(tmp), metadata=metadata)
            with safe_open(tmp, framework="np") as f:
                if len(f.keys()) != len(tensors):
                    raise RuntimeError(f"Validation failed for {tmp.name}")
            temp_paths[shard] = tmp
            print(
                f"[pad-minimax] wrote temp shard {shard_i}/{len(touched_shards)}: {shard}",
                flush=True,
            )
            del tensors

        for shard, tmp in temp_paths.items():
            os.replace(tmp, model_dir / shard)

        config["num_local_experts"] = n_padded
        config["weight_format"] = "mxtq"
        config["mxtq_bits"] = bits
        config["quantization"] = {"group_size": 64, "bits": bits}

        jang_config["expert_padding"] = {
            "original_num_local_experts": n_original,
            "padded_num_local_experts": n_padded,
            "multiple": EXPERT_PAD_MULTIPLE,
            "dummy_bias": DUMMY_BIAS,
            "reason": "router decode speed alignment",
        }
        source_cfg = jang_config.setdefault("source_config", {})
        if isinstance(source_cfg, dict):
            source_cfg.setdefault("n_routed_experts", n_original)
            source_cfg["padded_num_local_experts"] = n_padded

        index["weight_map"] = new_weight_map
        index["metadata"] = {
            **(index.get("metadata") or {}),
            "total_size": sum(p.stat().st_size for p in model_dir.glob("model-*-of-*.safetensors")),
        }

        _write_json_atomic(config_path, config)
        _write_json_atomic(jang_path, jang_config)
        _write_json_atomic(index_path, index)
    finally:
        for tmp in temp_paths.values():
            if tmp.exists():
                tmp.unlink()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pad a MiniMax JANGTQ artifact's routed expert count to a 32-wide boundary."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite safetensors/config/index in-place. Without this flag, only prints a dry run.",
    )
    args = parser.parse_args()
    summary = migrate(args.model_dir, apply=args.apply)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
