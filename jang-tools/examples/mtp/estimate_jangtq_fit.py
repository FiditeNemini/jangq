#!/usr/bin/env python3
"""Conservative Hy3 JANGTQ memory estimator.

This is a planning tool. Final device claims require measured bundle bytes and
runtime load proof.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GB = 1_000_000_000


def load_config(model_dir: Path) -> dict:
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def profile_bits(profile: str) -> float:
    p = profile.upper()
    if p == "JANGTQ2":
        return 2.0
    if p in {"JANGTQ_K", "JANGTQK"}:
        return (2.0 + 2.0 + 4.0) / 3.0
    if p == "JANGTQ4":
        return 4.0
    raise SystemExit(
        f"unknown profile {profile!r}; expected JANGTQ2, JANGTQ_K, or JANGTQ4"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--profile", default="JANGTQ_K")
    ap.add_argument("--device-gb", type=float, default=128.0)
    ap.add_argument("--bf16-size-gb", type=float, default=598.0)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--runtime-headroom-gb", type=float, default=12.0)
    args = ap.parse_args()

    cfg = load_config(args.model_dir)
    if cfg.get("model_type") != "hy_v3":
        raise SystemExit(
            f"this estimator is Hy3-specific; got model_type={cfg.get('model_type')!r}"
        )

    layers = int(cfg["num_hidden_layers"])
    sparse_layers = layers - int(cfg.get("first_k_dense_replace", 0))
    experts = int(cfg["num_experts"])
    hidden = int(cfg["hidden_size"])
    expert_hidden = int(
        cfg.get("moe_intermediate_size", cfg.get("expert_hidden_dim", 1536))
    )
    routed_params = sparse_layers * experts * 3 * hidden * expert_hidden

    total_params = args.bf16_size_gb * GB / 2.0
    non_routed_params = max(total_params - routed_params, 0)

    routed_bytes = routed_params * profile_bits(args.profile) / 8.0
    # Affine 8-bit group 64: uint8 weights + fp16 scale + fp16 bias per group.
    affine_bytes = non_routed_params * (1.0 + 4.0 / 64.0)
    sidecar_bytes = (routed_bytes + affine_bytes) * 0.04
    estimated_bundle = routed_bytes + affine_bytes + sidecar_bytes

    kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim", hidden // int(cfg["num_attention_heads"])))
    kv_bytes_per_token = layers * 2 * kv_heads * head_dim * 2
    kv_bytes = kv_bytes_per_token * args.context

    estimated_runtime = estimated_bundle + kv_bytes + args.runtime_headroom_gb * GB
    device = args.device_gb * GB
    ratio = estimated_runtime / device
    if ratio < 0.75:
        verdict = "comfortable"
    elif ratio < 0.90:
        verdict = "tight"
    else:
        verdict = "not_comfortable"

    out = {
        "profile": args.profile,
        "device_gb": args.device_gb,
        "context": args.context,
        "routed_params_b": routed_params / 1e9,
        "estimated_routed_gb": routed_bytes / GB,
        "estimated_non_routed_gb": affine_bytes / GB,
        "estimated_sidecar_gb": sidecar_bytes / GB,
        "estimated_bundle_gb": estimated_bundle / GB,
        "kv_cache_gb_at_context": kv_bytes / GB,
        "runtime_headroom_gb": args.runtime_headroom_gb,
        "estimated_runtime_total_gb": estimated_runtime / GB,
        "device_ratio": ratio,
        "verdict": verdict,
        "warning": (
            "Planning estimate only; measured bundle bytes and real runtime "
            "load proof are required."
        ),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
