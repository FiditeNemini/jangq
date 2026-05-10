"""Inspect Zyphra/ZAYA1-VL-8B without loading full weights.

This is a header-only check: reads JSON + index + shard headers only.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/Users/eric/models/Zyphra/ZAYA1-VL-8B")
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.zaya_block\.experts\.local_experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight$"
)
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")


def read_safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def layer_sets(keys: list[str]) -> tuple[list[int], list[int]]:
    attn = set()
    moe = set()
    for key in keys:
        m = LAYER_RE.search(key)
        if not m:
            continue
        layer = int(m.group(1))
        if ".self_attn." in key:
            attn.add(layer)
        if ".zaya_block." in key:
            moe.add(layer)
    return sorted(attn), sorted(moe)


def print_header_shapes(model_dir: Path, shards: list[str]) -> None:
    existing = [model_dir / shard for shard in shards if (model_dir / shard).exists()]
    missing = [shard for shard in shards if not (model_dir / shard).exists()]
    print("\nSafetensor headers")
    print(f"  shards present: {len(existing)}/{len(shards)}")
    if missing:
        print(f"  missing shards: {', '.join(missing)}")
    if not existing:
        return

    dtype_counts: collections.Counter[str] = collections.Counter()
    shape_samples: dict[str, list[int]] = {}
    wanted = {
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.qkv.linear_q.weight",
        "model.layers.0.self_attn.qkv.linear_k.weight",
        "model.layers.0.zaya_block.experts.local_experts.0.linear_fc1.weight",
        "model.layers.0.zaya_block.experts.local_experts.0.linear_fc2.weight",
        "model.vision_projection.linear.weight",
        "model.mm_projector.linear1.weight",
        "model.final_norm.weight",
    }

    for shard in existing:
        header = read_safetensor_header(shard)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_counts[str(meta.get("dtype", "?"))] += 1
            if name in wanted:
                shape_samples[name] = list(meta.get("shape", []))

    print("  dtype counts:")
    for dtype, count in sorted(dtype_counts.items()):
        print(f"    {dtype}: {count}")
    print("  selected shapes:")
    for name in sorted(wanted):
        print(f"    {name}: {shape_samples.get(name)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser()
    require(model_dir / "config.json")
    require(model_dir / "model.safetensors.index.json")
    require(model_dir / "tokenizer_config.json")

    config = read_json(model_dir / "config.json")
    index = read_json(model_dir / "model.safetensors.index.json")
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    preproc = read_json(model_dir / "preprocessor_config.json")
    keys = sorted(index["weight_map"])
    shards = sorted(set(index["weight_map"].values()))

    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    q_heads = int(config.get("cca_num_q_heads", config["num_attention_heads"]))
    kv_heads = int(config["num_query_groups"])
    head_dim = hidden // heads
    attn_layers, moe_layers = layer_sets(keys)
    local_expert_keys = [m.group(1) for m in (EXPERT_RE.match(k) for k in keys) if m is not None]

    print(f"ZAYA1-VL source: {model_dir}")
    print(f"model_type: {config.get('model_type')}")
    print(f"architectures: {config.get('architectures')}")
    print(f"tensors: {len(keys)} across {len(shards)} shards")
    print(f"total size bytes: {index.get('metadata', {}).get('total_size')}")
    print(f"num_hidden_layers: {config.get('num_hidden_layers')}")
    print(f"num_attention_heads: {heads}")
    print(f"num_query_groups: {kv_heads}")
    print(f"num_experts: {config.get('num_experts')}")

    print("\nLayer schedule")
    print(f"  attention layers ({len(attn_layers)}): {attn_layers[:6]} ... {attn_layers[-6:]}")
    print(f"  moe layers ({len(moe_layers)}): {moe_layers[:6]} ... {moe_layers[-6:]}")
    print(f"  local expert coverage keys: {len(set(local_expert_keys))}")

    print("\nVision stack")
    vision_cfg = config.get("vision_config") or {}
    print(f"  vision_config.model_type: {vision_cfg.get('model_type')}")
    print(f"  preprocessor: {preproc.get('processor_class')} / {preproc.get('image_processor_type')}")
    print(f"  image_token_id: {config.get('image_token_id')}")
    print(f"  vision_start_token_id: {config.get('vision_start_token_id')}")
    print(f"  vision_end_token_id: {config.get('vision_end_token_id')}")

    chat_template = tokenizer_config.get("chat_template")
    print("\nTokenizer/template")
    print(f"  tokenizer_class: {tokenizer_config.get('tokenizer_class')}")
    print(f"  chat_template (from config): {type(chat_template).__name__}")
    print(f"  chat_template.jinja present: {(model_dir / 'chat_template.jinja').exists()}")
    print(f"  chat_template.json present: {(model_dir / 'chat_template.json').exists()}")

    print("\nGeometry")
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    conv_dim = q_dim + kv_dim
    print(f"  hidden={hidden}, heads={heads}, head_dim={head_dim}")
    print(f"  q_heads={q_heads} (defaults to num_attention_heads if cca_num_q_heads absent)")
    print(f"  kv_heads={kv_heads}, conv_qk_channels={conv_dim}")
    print(f"  attention KV/state shape pattern: [B, {kv_heads}, T, {head_dim}] + conv state [B, {conv_dim}, 2]")

    print_header_shapes(model_dir, shards)


if __name__ == "__main__":
    main()
