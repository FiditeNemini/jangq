#!/usr/bin/env python3
"""Validate DeepSeek-V4-Flash artifact RoPE metadata before upload.

DSV4 Flash compressed layers use ``compress_rope_theta=160000`` plus YaRN
``rope_scaling``. ``rope_parameters`` is only a compatibility mirror for newer
Transformers configs; it is not sufficient for the local MLX runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_ROPE_SCALING = {
    "type": "yarn",
    "factor": 16,
    "original_max_position_embeddings": 65536,
    "beta_fast": 32,
    "beta_slow": 1,
}

DEFAULT_TARGETS = (
    Path("/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANGTQ-K"),
    Path("/Users/eric/models/JANGQ/DeepSeek-V4-Flash-JANG"),
)


def _config_path(path: Path) -> Path:
    return path if path.name == "config.json" else path / "config.json"


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def validate_config(path: Path) -> list[str]:
    cfg_path = _config_path(path)
    if not cfg_path.exists():
        return [f"{cfg_path}: missing config.json"]

    cfg = json.loads(cfg_path.read_text())
    errors: list[str] = []

    if cfg.get("model_type") != "deepseek_v4":
        errors.append(f"{cfg_path}: expected model_type=deepseek_v4")

    compress_ratios = cfg.get("compress_ratios")
    if not isinstance(compress_ratios, list) or not any(_num(v) and _num(v) > 0 for v in compress_ratios):
        errors.append(f"{cfg_path}: expected DSV4 compressed layers in compress_ratios")

    if _num(cfg.get("compress_rope_theta")) != 160000:
        errors.append(f"{cfg_path}: expected compress_rope_theta=160000")

    rope_scaling = cfg.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
        errors.append(f"{cfg_path}: rope_scaling must be present and non-null")
        return errors

    rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")
    if rope_type != REQUIRED_ROPE_SCALING["type"]:
        errors.append(f"{cfg_path}: expected rope_scaling.type=yarn")

    for key in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow"):
        actual = _num(rope_scaling.get(key))
        expected = float(REQUIRED_ROPE_SCALING[key])
        if actual != expected:
            errors.append(f"{cfg_path}: expected rope_scaling.{key}={REQUIRED_ROPE_SCALING[key]}")

    rope_parameters = cfg.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        param_type = rope_parameters.get("rope_type") or rope_parameters.get("type")
        if param_type != "yarn":
            errors.append(f"{cfg_path}: rope_parameters mirror must keep rope_type=yarn")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        type=Path,
        default=list(DEFAULT_TARGETS),
        help="Artifact directories or config.json paths to validate.",
    )
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    for target in args.targets:
        errors = validate_config(target)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"PASS { _config_path(target) }")

    for error in all_errors:
        print(f"FAIL {error}", file=sys.stderr)
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
