"""Inspect a local Zyphra/ZAYA1-8B source snapshot without loading weights.

This script is intentionally header-only: it reads JSON files, the safetensors
index, and shard headers. It never materializes tensor data.

Run:
    python3 00_inspect_source.py [/path/to/ZAYA1-8B]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import struct
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B")
EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.zaya_block\.experts\.local_experts\.(\d+)\.(linear_fc1|linear_fc2)\.weight$"
)
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_size))


def require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required file: {path}")


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


def summarize_experts(keys: list[str]) -> dict[int, set[int]]:
    by_layer: dict[int, set[int]] = collections.defaultdict(set)
    for key in keys:
        m = EXPERT_RE.match(key)
        if not m:
            continue
        by_layer[int(m.group(1))].add(int(m.group(2)))
    return by_layer


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
        "model.layers.0.self_attn.qkv.val_proj1.weight",
        "model.layers.0.self_attn.qkv.val_proj2.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.1.zaya_block.experts.local_experts.0.linear_fc1.weight",
        "model.layers.1.zaya_block.experts.local_experts.0.linear_fc2.weight",
        "model.layers.1.zaya_block.router.down_proj.weight",
        "model.layers.1.zaya_block.router.router_mlp.4.weight",
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
        shape = shape_samples.get(name)
        print(f"    {name}: {shape if shape is not None else 'pending'}")


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
    weight_map = index["weight_map"]
    keys = sorted(weight_map)
    shards = sorted(set(weight_map.values()))

    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    q_heads = int(config["cca_num_q_heads"])
    kv_heads = int(config["num_query_groups"])
    head_dim = hidden // heads
    q_dim = q_heads * head_dim
    kv_dim = kv_heads * head_dim
    conv_dim = q_dim + kv_dim
    attn_layers, moe_layers = layer_sets(keys)
    experts = summarize_experts(keys)

    print(f"ZAYA source: {model_dir}")
    print(f"model_type: {config.get('model_type')}")
    print(f"architectures: {config.get('architectures')}")
    print(f"tensors: {len(keys)} across {len(shards)} shards")
    print(f"total parameters: {index.get('metadata', {}).get('total_parameters')}")
    print(f"total size bytes: {index.get('metadata', {}).get('total_size')}")

    print("\nLayer schedule")
    print(f"  attention layers ({len(attn_layers)}): {attn_layers[:8]} ... {attn_layers[-8:]}")
    print(f"  moe layers ({len(moe_layers)}): {moe_layers[:8]} ... {moe_layers[-8:]}")
    bad_expert_layers = {
        layer: sorted(vals)
        for layer, vals in experts.items()
        if len(vals) != int(config["num_experts"])
    }
    print(f"  expert layers with all {config['num_experts']} experts: {len(experts) - len(bad_expert_layers)}/{len(moe_layers)}")
    if bad_expert_layers:
        print(f"  bad expert coverage: {bad_expert_layers}")

    print("\nAttention geometry")
    print(f"  hidden={hidden}, heads={heads}, head_dim={head_dim}")
    print(f"  CCA q_heads={q_heads}, kv_heads={kv_heads}")
    print(f"  q_dim={q_dim}, k_dim={kv_dim}, v_dim={kv_dim}, conv_qk_channels={conv_dim}")
    print(f"  standard KV cache per attention layer: [B, {kv_heads}, T, {head_dim}]")
    print(f"  CCA conv state per attention layer: [B, {conv_dim}, 2]")
    print(f"  CCA prev_hs per attention layer: [B, {hidden}]")

    print("\nMoE geometry")
    print(f"  actual experts per MoE layer: {config['num_experts']}")
    print(f"  router logits: {int(config['num_experts']) + (1 if config.get('zaya_use_mod') else 0)} including MOD skip")
    print(f"  topk: {config['moe_router_topk']}")
    print(f"  expert linear_fc1: [4096, {hidden}] -> split gate/up halves for JANGTQ")
    print(f"  expert linear_fc2: [{hidden}, {hidden}]")

    chat_template = tokenizer_config.get("chat_template")
    chat_template_path = model_dir / "chat_template.jinja"
    print("\nTokenizer/template")
    print(f"  tokenizer_class: {tokenizer_config.get('tokenizer_class')}")
    print(f"  tokenizer_config.chat_template: {type(chat_template).__name__}")
    print(f"  chat_template.jinja present: {chat_template_path.exists()}")
    print(f"  config eos_token_id: {config.get('eos_token_id')}")
    gen_path = model_dir / "generation_config.json"
    if gen_path.exists():
        generation_config = read_json(gen_path)
        print(f"  generation_config eos_token_id: {generation_config.get('eos_token_id')}")
    print("  runtime stop set should include both <|im_end|> and <eos> until source behavior is verified")

    print("\nRuntime policy")
    print("  continuous batching: compatible with per-slot KV + CCA state")
    print("  paged KV: compatible for standard attention KV only")
    print("  prefix cache: disabled for first port; official vLLM asserts it off")
    print("  TurboQuant KV: KV-only experimental; keep CCA conv/prev_hs float32")
    print("  chunked prefill: do not enable until CCA state copy tests pass")

    print_header_shapes(model_dir, shards)


if __name__ == "__main__":
    main()
