"""Emit the ZAYA runtime/cache contract as JSON.

The script is dependency-light and header-only. It reads config/index/tokenizer
metadata from a local Zyphra/ZAYA1-8B snapshot and prints the geometry that a
Python, Swift, or service runtime needs before implementing batching, paged KV,
prefix restore, or TurboQuant KV.

Run:
    python3 02_python_runtime_contract.py [/path/to/ZAYA1-8B]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("/Users/eric/jang/models/Zyphra/ZAYA1-8B")


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
    index = read_json(model_dir / "model.safetensors.index.json")
    tokenizer_config = read_json(model_dir / "tokenizer_config.json")
    generation_config = (
        read_json(model_dir / "generation_config.json")
        if (model_dir / "generation_config.json").exists()
        else {}
    )

    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    q_heads = int(config["cca_num_q_heads"])
    kv_heads = int(config["num_query_groups"])
    head_dim = hidden // heads
    conv_channels = (q_heads + kv_heads) * head_dim

    contract = {
        "model": {
            "path": str(model_dir),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "total_tensors": len(index.get("weight_map", {})),
            "total_parameters": index.get("metadata", {}).get("total_parameters"),
            "total_size": index.get("metadata", {}).get("total_size"),
        },
        "layers": {
            "num_hidden_layers": layers,
            "attention_layers": list(range(0, layers, 2)),
            "moe_layers": list(range(1, layers, 2)),
        },
        "attention": {
            "type": "cca",
            "hidden_size": hidden,
            "num_attention_heads": heads,
            "cca_num_q_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "q_dim": q_heads * head_dim,
            "kv_dim": kv_heads * head_dim,
            "conv_qk_channels": conv_channels,
            "standard_kv_shape": ["batch", kv_heads, "tokens", head_dim],
            "conv_state_shape": ["batch", conv_channels, 2],
            "prev_hs_shape": ["batch", hidden],
        },
        "moe": {
            "type": "top1_zaya",
            "num_experts": int(config["num_experts"]),
            "router_logits": int(config["num_experts"]) + (1 if config.get("zaya_use_mod") else 0),
            "topk": int(config["moe_router_topk"]),
            "source_fc1": [2 * hidden, hidden],
            "source_fc2": [hidden, hidden],
            "jang_switch_mlp": {
                "gate_proj": "linear_fc1[:hidden_size, :]",
                "up_proj": "linear_fc1[hidden_size:, :]",
                "down_proj": "linear_fc2",
            },
        },
        "runtime_policy": {
            "continuous_batching": {
                "status": "compatible",
                "requirement": "each sequence slot owns KV plus CCA conv_state and prev_hs",
            },
            "paged_kv": {
                "status": "standard_kv_only_first",
                "requirement": "page KV blocks separately from CCA inner state; do not mark a prefix restored unless both are restored",
            },
            "prefix_cache": {
                "status": "disabled_first_port",
                "requirement": "official vLLM ZAYA disables prefix caching; enable only after exact CCA state restore tests",
            },
            "turboquant_kv": {
                "status": "kv_only_experimental",
                "requirement": "encode standard attention K/V only; keep conv_state and prev_hs float32",
            },
            "chunked_prefill": {
                "status": "blocked_until_state_tests",
                "requirement": "prefill chunks must carry exact CCA conv_state and prev_hs boundaries",
            },
        },
        "tokenizer": {
            "tokenizer_class": tokenizer_config.get("tokenizer_class"),
            "tokenizer_config_has_chat_template": bool(tokenizer_config.get("chat_template")),
            "chat_template_jinja_present": (model_dir / "chat_template.jinja").exists(),
            "config_eos_token_id": config.get("eos_token_id"),
            "generation_eos_token_id": generation_config.get("eos_token_id"),
            "stop_ids": sorted(
                {
                    int(x)
                    for x in [config.get("eos_token_id"), generation_config.get("eos_token_id")]
                    if isinstance(x, int)
                }
            ),
        },
        "quantization_policy": {
            "jangtq": {
                "expert_layout": "prestacked switch_mlp tq_packed/tq_norms/tq_bits",
                "router": "passthrough first pass",
                "attention": "8-bit affine first pass",
                "cca_state": "not weight-quantized; runtime state stays float32",
                "rowwise_packing": "required for 3-bit widths that are not divisible by 32 // bits",
            },
            "mxfp4": {
                "linears": "4-bit affine group_size=32 first pass",
                "passthrough": ["router", "conv_qk", "temp", "norms", "residual_scales", "balancing_biases"],
            },
        },
    }

    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
