"""Inspect tencent/Hy3-preview without loading full weights.

This is safe during the long download. It reads JSON files and safetensor
headers only. If the index is not present yet, it reports partial shard state.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/Users/eric/models/Tencent/Hy3-preview")
LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return data


def read_safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def header_summary(model_dir: Path) -> dict[str, Any]:
    dtype_counts: collections.Counter[str] = collections.Counter()
    layer_counts: collections.Counter[int] = collections.Counter()
    samples: dict[str, list[int]] = {}
    wanted = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.e_score_correction_bias",
        "model.layers.1.mlp.expert_bias",
        "model.layers.1.mlp.router.gate.weight",
        "model.layers.1.mlp.shared_mlp.gate_proj.weight",
        "model.layers.1.mlp.shared_mlp.up_proj.weight",
        "model.layers.1.mlp.shared_mlp.down_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.80.self_attn.q_proj.weight",
        "lm_head.weight",
    }

    shards = sorted(model_dir.glob("model-*.safetensors"))
    for shard in shards:
        try:
            header = read_safetensor_header(shard)
        except Exception as exc:
            print(f"warning: could not read header for {shard.name}: {exc}")
            continue
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_counts[str(meta.get("dtype", "?"))] += 1
            m = LAYER_RE.match(name)
            if m:
                layer_counts[int(m.group(1))] += 1
            if name in wanted:
                samples[name] = list(meta.get("shape", []))

    return {
        "local_shards": len(shards),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "layers_seen": sorted(layer_counts),
        "sample_shapes": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser()
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json under {model_dir}")

    config = read_json(config_path)
    index_path = model_dir / "model.safetensors.index.json"
    index = read_json(index_path) if index_path.exists() else None
    local = header_summary(model_dir)

    summary = {
        "path": str(model_dir),
        "download_state": {
            "index_present": index is not None,
            "local_shards": local["local_shards"],
            "expected_shards": len(set(index.get("weight_map", {}).values())) if index else None,
            "tensor_count": len(index.get("weight_map", {})) if index else None,
            "total_size": index.get("metadata", {}).get("total_size") if index else None,
        },
        "config": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "moe_intermediate_size": config.get("moe_intermediate_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": config.get("head_dim"),
            "num_experts": config.get("num_experts"),
            "num_experts_per_tok": config.get("num_experts_per_tok"),
            "num_shared_experts": config.get("num_shared_experts"),
            "first_k_dense_replace": config.get("first_k_dense_replace"),
            "moe_router_use_sigmoid": config.get("moe_router_use_sigmoid"),
            "moe_router_enable_expert_bias": config.get("moe_router_enable_expert_bias"),
            "route_norm": config.get("route_norm"),
            "router_scaling_factor": config.get("router_scaling_factor"),
            "qk_norm": config.get("qk_norm"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rope_parameters": config.get("rope_parameters"),
            "enable_lm_head_fp32": config.get("enable_lm_head_fp32"),
            "vocab_size": config.get("vocab_size"),
        },
        "headers": local,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
