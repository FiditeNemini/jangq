#!/usr/bin/env python3
"""Qwen3.6 MTP/VL metadata probe.

This is a low-RAM preflight for source trees and converted bundles. It reads
JSON metadata and safetensors headers only; it does not load model weights into
MLX and does not run generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_mtp_name(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("mtp.") or ".mtp." in lower or ".mtp_" in lower


def is_visual_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith("model.visual")
        or ".visual." in lower
        or lower.startswith("vision_tower")
        or "vision_model" in lower
        or "vision_encoder" in lower
        or "embed_vision" in lower
    )


def configured_mtp_layers(config: dict[str, Any]) -> int:
    text_cfg = config.get("text_config", {})
    for value in (
        config.get("num_nextn_predict_layers"),
        config.get("mtp_num_hidden_layers"),
        text_cfg.get("num_nextn_predict_layers") if isinstance(text_cfg, dict) else None,
        text_cfg.get("mtp_num_hidden_layers") if isinstance(text_cfg, dict) else None,
    ):
        if value is None:
            continue
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def header_shapes(model_dir: Path, weight_map: dict[str, str], names: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    by_file: dict[str, list[str]] = {}
    for name in names:
        shard = weight_map.get(name)
        if shard:
            by_file.setdefault(shard, []).append(name)

    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(f"safetensors unavailable: {exc}") from exc

    for shard, shard_names in by_file.items():
        with safe_open(str(model_dir / shard), framework="numpy") as f:
            for name in shard_names:
                out[name] = list(f.get_slice(name).get_shape())
    return out


def probe(model_dir: Path, *, check_headers: bool = True, strict: bool = False) -> dict[str, Any]:
    config = load_json(model_dir / "config.json")
    jang_path = model_dir / "jang_config.json"
    jang = load_json(jang_path) if jang_path.exists() else {}
    index = load_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map", {})
    names = sorted(weight_map)

    mtp_names = [name for name in names if is_mtp_name(name)]
    visual_names = [name for name in names if is_visual_name(name)]
    archs = config.get("architectures") or []
    conditional_generation = any("ConditionalGeneration" in str(a) for a in archs)
    text_cfg = config.get("text_config", {})
    hidden_size = (
        config.get("hidden_size")
        or (text_cfg.get("hidden_size") if isinstance(text_cfg, dict) else None)
    )

    errors: list[str] = []
    warnings: list[str] = []
    mtp_layers = configured_mtp_layers(config)
    if mtp_layers > 0 and not mtp_names:
        errors.append("missing mtp tensor weights")
    if conditional_generation and not visual_names:
        errors.append("missing visual tensor weights")
    if visual_names and not (model_dir / "preprocessor_config.json").exists():
        errors.append("missing preprocessor_config.json")
    if visual_names and not (model_dir / "video_preprocessor_config.json").exists():
        warnings.append("missing video_preprocessor_config.json")

    runtime = jang.get("runtime", {}) if isinstance(jang, dict) else {}
    if mtp_names and runtime:
        mode = runtime.get("mtp_mode")
        if mode == "preserved_disabled":
            errors.append("jang_config runtime.mtp_mode is stale preserved_disabled")

    mtp_shapes = {}
    visual_shapes = {}
    if check_headers:
        mtp_shapes = header_shapes(model_dir, weight_map, mtp_names[:20])
        visual_shapes = header_shapes(model_dir, weight_map, visual_names[:20])

    ok = not errors if strict else True
    return {
        "path": str(model_dir),
        "ok": ok,
        "strict": strict,
        "model_type": config.get("model_type"),
        "architectures": archs,
        "hidden_size": hidden_size,
        "configured_mtp_layers": mtp_layers,
        "mtp_tensor_count": len(mtp_names),
        "mtp_tensor_samples": mtp_names[:20],
        "visual_tensor_count": len(visual_names),
        "visual_tensor_samples": visual_names[:20],
        "has_preprocessor_config": (model_dir / "preprocessor_config.json").exists(),
        "has_video_preprocessor_config": (model_dir / "video_preprocessor_config.json").exists(),
        "runtime": runtime,
        "mtp_shapes": mtp_shapes,
        "visual_shapes": visual_shapes,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-headers", action="store_true")
    args = ap.parse_args()

    out = probe(args.model_dir, check_headers=not args.no_headers, strict=args.strict)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
