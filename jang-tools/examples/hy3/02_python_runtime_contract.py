"""Emit the Hy3-preview runtime/cache contract as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/Users/eric/models/Tencent/Hy3-preview")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser()
    config = read_json(model_dir / "config.json")
    generation_config = (
        read_json(model_dir / "generation_config.json")
        if (model_dir / "generation_config.json").exists()
        else {}
    )

    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    n_mtp = int(config.get("num_nextn_predict_layers", 0))

    contract = {
        "model": {
            "path": str(model_dir),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "text_only": "vision_config" not in config,
        },
        "attention": {
            "type": "dense_gqa",
            "cache_topology": "standard_kv",
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "num_key_value_groups": heads // kv_heads,
            "head_dim": head_dim,
            "q_shape": ["batch", heads, "tokens", head_dim],
            "kv_shape": ["batch", kv_heads, "tokens", head_dim],
            "qk_norm": bool(config.get("qk_norm")),
            "rope": config.get("rope_parameters"),
            "max_position_embeddings": config.get("max_position_embeddings"),
        },
        "moe": {
            "type": "sigmoid_bias_topk_with_shared_expert",
            "num_experts": int(config["num_experts"]),
            "num_experts_per_tok": int(config["num_experts_per_tok"]),
            "num_shared_experts": int(config.get("num_shared_experts", 0)),
            "moe_intermediate_size": int(config["moe_intermediate_size"]),
            "first_k_dense_replace": int(config.get("first_k_dense_replace", 0)),
            "route_norm": bool(config.get("route_norm")),
            "router_scaling_factor": config.get("router_scaling_factor"),
            "router_uses_sigmoid": bool(config.get("moe_router_use_sigmoid")),
            "router_uses_expert_bias": bool(config.get("moe_router_enable_expert_bias")),
            "runtime_rule": (
                "route with sigmoid logits, add expert correction bias for top-k "
                "choice, gather original sigmoid weights, normalize selected weights, "
                "multiply by router_scaling_factor, then add shared expert output"
            ),
        },
        "mtp": {
            "num_nextn_predict_layers": n_mtp,
            "runtime_mode": "preserved_disabled until accept/reject speculative decode is implemented",
            "policy": (
                "implement speculative decode explicitly or drop/ignore MTP tensors "
                "with a documented runtime boundary"
            ),
        },
        "tokenizer": {
            "bos_token_id": config.get("bos_token_id"),
            "eos_token_id": config.get("eos_token_id"),
            "pad_token_id": config.get("pad_token_id"),
            "generation_eos_token_id": generation_config.get("eos_token_id"),
            "chat_template_present": (model_dir / "chat_template.jinja").exists(),
            "reasoning_effort_levels": ["no_think", "low", "high"],
        },
        "quantization_policy": {
            "jangtq": {
                "first_release_candidate": "JANGTQ2",
                "memory_note": (
                    "JANGTQ2 is the 128 GB release candidate; JANGTQ_K is "
                    "quality-first and likely tight on 128 GB unless measured "
                    "runtime load proof says otherwise"
                ),
                "profiles": ["JANGTQ_K", "JANGTQ2", "JANGTQ4"],
                "routed_experts": {
                    "JANGTQ_K": {"gate_proj": 2, "up_proj": 2, "down_proj": 4},
                    "JANGTQ2": 2,
                    "JANGTQ4": 4,
                },
                "attention": "8-bit affine first pass",
                "shared_expert": "8-bit affine first pass",
                "dense_ffn": "8-bit affine first pass",
                "mtp": "8-bit affine where present; runtime must document speculative vs normal decode",
                "precision_floors": [
                    "router gate",
                    "expert correction bias",
                    "q_norm/k_norm",
                    "all RMSNorms",
                    "lm_head until coherence proves safe",
                ],
            },
        },
    }
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
