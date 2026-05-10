#!/usr/bin/env python3
"""Hy3 JANGTQ runtime skeleton for future vmlx Python work.

This file is intentionally lightweight and does not load model weights. It
captures the architecture/runtime contract future agents should implement in
`../vmlx`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Literal


MTPMode = Literal["none", "preserved_disabled", "enabled"]


@dataclass(frozen=True)
class Hy3AttentionSpec:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    qk_norm: bool
    rope_theta: float
    max_position_embeddings: int

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_bytes_per_token_fp16(self) -> int:
        return self.num_layers * 2 * self.num_key_value_heads * self.head_dim * 2


@dataclass(frozen=True)
class Hy3MoESpec:
    num_experts: int
    top_k: int
    num_shared_experts: int
    expert_hidden_dim: int
    first_k_dense_replace: int
    route_norm: bool
    router_scaling_factor: float
    use_sigmoid: bool
    use_expert_bias: bool


@dataclass(frozen=True)
class Hy3RuntimeSpec:
    attention: Hy3AttentionSpec
    moe: Hy3MoESpec
    mtp_layers: int
    mtp_mode: MTPMode


def load_hy3_runtime_spec(model_dir: str | Path) -> Hy3RuntimeSpec:
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "hy_v3":
        raise ValueError(f"expected model_type='hy_v3', got {config.get('model_type')!r}")

    rope = config.get("rope_parameters") or {}
    head_dim = int(config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"])
    attention = Hy3AttentionSpec(
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_hidden_layers"]),
        num_attention_heads=int(config["num_attention_heads"]),
        num_key_value_heads=int(config["num_key_value_heads"]),
        head_dim=head_dim,
        qk_norm=bool(config.get("qk_norm")),
        rope_theta=float(rope.get("rope_theta", 1_000_000.0)),
        max_position_embeddings=int(config["max_position_embeddings"]),
    )
    moe = Hy3MoESpec(
        num_experts=int(config["num_experts"]),
        top_k=int(config["num_experts_per_tok"]),
        num_shared_experts=int(config.get("num_shared_experts", 1)),
        expert_hidden_dim=int(config.get("moe_intermediate_size", config.get("expert_hidden_dim", 1536))),
        first_k_dense_replace=int(config.get("first_k_dense_replace", 0)),
        route_norm=bool(config.get("route_norm")),
        router_scaling_factor=float(config.get("router_scaling_factor", 1.0)),
        use_sigmoid=bool(config.get("moe_router_use_sigmoid")),
        use_expert_bias=bool(config.get("moe_router_enable_expert_bias")),
    )
    runtime = {}
    jang_path = model_dir / "jang_config.json"
    if jang_path.exists():
        runtime = (json.loads(jang_path.read_text()).get("runtime") or {})
    mtp_layers = int(runtime.get("mtp_layers", config.get("num_nextn_predict_layers", 0)))
    mtp_mode = runtime.get("mtp_mode") or ("preserved_disabled" if mtp_layers else "none")
    return Hy3RuntimeSpec(attention=attention, moe=moe, mtp_layers=mtp_layers, mtp_mode=mtp_mode)


def describe_cache(spec: Hy3RuntimeSpec, context_tokens: int = 4096) -> dict:
    kv_bytes = spec.attention.kv_bytes_per_token_fp16 * context_tokens
    return {
        "cache_type": "standard_kv",
        "kv_bytes_per_token_fp16": spec.attention.kv_bytes_per_token_fp16,
        "kv_gb_at_context": kv_bytes / 1_000_000_000,
        "mtp_cache_rule": "draft state must be separate from accepted base KV",
    }


def router_contract() -> list[str]:
    return [
        "router_logits = router_gate(hidden)",
        "router_probs = sigmoid(router_logits)",
        "topk_scores = router_probs + expert_bias for expert choice",
        "selected = topk(topk_scores, k=8)",
        "weights = gather(router_probs, selected)",
        "if route_norm: weights = weights / sum(weights)",
        "weights = weights * router_scaling_factor",
        "output = weighted_sum(selected_expert_mlp(hidden)) + shared_expert(hidden)",
    ]


def implementation_todos() -> list[str]:
    return [
        "Implement Hy3Attention with q/k RMSNorm before RoPE.",
        "Use standard KV cache; do not route through MLA/SSM/CCA helpers.",
        "Map routed experts to JANGTQ/TurboQuant kernel path.",
        "Map attention/shared/dense/embed/lm_head/MTP matmuls to affine quantized linear.",
        "Keep router gate, expert_bias, q/k norms, and RMSNorms unquantized.",
        "Keep mtp_mode='preserved_disabled' until accept/reject speculative decode is tested.",
        "Add Hunyuan/Tencent tool parser and reasoning_effort template plumbing.",
    ]


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, default=4096)
    args = ap.parse_args()
    spec = load_hy3_runtime_spec(args.model_dir)
    print(json.dumps({
        "attention": spec.attention.__dict__ | {"num_key_value_groups": spec.attention.num_key_value_groups},
        "moe": spec.moe.__dict__,
        "mtp_layers": spec.mtp_layers,
        "mtp_mode": spec.mtp_mode,
        "cache": describe_cache(spec, args.context),
        "router_contract": router_contract(),
        "implementation_todos": implementation_todos(),
    }, indent=2))


if __name__ == "__main__":
    main()

