"""Predict converted size for a source model + profile combo.

Swift's PreflightRunner should call this instead of hardcoding bits-per-weight
estimates — keeps predictions accurate as new profiles ship.

JSON output:
  {
    "source_bytes": 12345678,
    "source_gb": 12.3,
    "predicted_output_bytes": 3456789,
    "predicted_output_gb": 3.5,
    "predicted_avg_bits": 4.23,
    "profile": "JANG_4K",
    "method": "bits_per_weight_linear"
  }
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

from .allocate import JANG_PROFILES, JANG_K_TARGETS
from .profiles_cli import _JANGTQ_PROFILES


def _source_bytes(model_dir: Path) -> int:
    """Sum of all .safetensors files. Approximates unquantized source size."""
    return sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))


def _predict_avg_bits(profile: str) -> float:
    """Best-effort avg-bits/weight for a profile. Used as the multiplier vs source bf16."""
    if profile in JANG_K_TARGETS:
        return JANG_K_TARGETS[profile]
    if profile in JANG_PROFILES:
        crit, imp, comp = JANG_PROFILES[profile]
        # Approximation: compress is majority of weights; critical + important are minorities
        # Typical distribution: 80% compress, 15% important, 5% critical
        return round(0.80 * comp + 0.15 * imp + 0.05 * crit, 2)
    # JANGTQ
    for p in _JANGTQ_PROFILES:
        if p["name"] == profile:
            return float(p["bits"])
    raise ValueError(f"unknown profile: {profile}")


def predict(model_dir: Path, profile: str) -> dict[str, Any]:
    src_bytes = _source_bytes(model_dir)
    if src_bytes == 0:
        # Best-effort fallback: read config and estimate from num params
        import json as _json
        cfg_path = model_dir / "config.json"
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
            hidden = int(cfg.get("hidden_size", 0) or (cfg.get("text_config", {}) or {}).get("hidden_size", 0))
            layers = int(cfg.get("num_hidden_layers", 0) or (cfg.get("text_config", {}) or {}).get("num_hidden_layers", 0))
            vocab = int(cfg.get("vocab_size", 0) or (cfg.get("text_config", {}) or {}).get("vocab_size", 0))
            if hidden and layers:
                # Very rough: ~12*h^2 per layer + embed + lm_head
                approx_params = 12 * hidden * hidden * layers + 2 * hidden * vocab
                src_bytes = approx_params * 2   # assume bf16 source
    avg_bits = _predict_avg_bits(profile)
    # Source assumed bf16 (16 bits/weight). Output = source × (avg_bits / 16) × overhead (1.05 for metadata)
    predicted = int(src_bytes * (avg_bits / 16.0) * 1.05)
    return {
        "source_bytes": src_bytes,
        "source_gb": round(src_bytes / 1_000_000_000, 3),
        "predicted_output_bytes": predicted,
        "predicted_output_gb": round(predicted / 1_000_000_000, 3),
        "predicted_avg_bits": avg_bits,
        "profile": profile,
        "method": "bits_per_weight_linear",
    }


def cmd_estimate_model(args) -> None:
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"ERROR: model dir not found: {model_dir}", file=sys.stderr)
        sys.exit(2)
    try:
        result = predict(model_dir, args.profile)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(3)
    if args.json:
        print(json.dumps(result, indent=None))
    else:
        print(f"Source: {result['source_gb']} GB")
        print(f"Profile: {result['profile']} (avg {result['predicted_avg_bits']} bits/weight)")
        print(f"Predicted output: {result['predicted_output_gb']} GB")


def register(subparsers) -> None:
    p = subparsers.add_parser("estimate-model", help="Predict converted size for a source model + profile")
    p.add_argument("--model", required=True, help="Path to source HuggingFace model dir")
    p.add_argument("--profile", required=True, help="Target JANG or JANGTQ profile name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_estimate_model)
