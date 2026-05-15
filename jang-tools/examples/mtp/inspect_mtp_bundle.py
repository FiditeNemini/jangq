#!/usr/bin/env python3
"""Low-RAM MTP bundle inspector.

Reads config/index metadata only. Does not load tensor payloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MTP_MARKERS = ("mtp", "nextn", "next_n", "speculative", "draft")
VISUAL_MARKERS = (
    "model.visual",
    "vision_tower",
    "vision_model",
    "vision_encoder",
    "embed_vision",
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_mtp_fields(value, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(marker in path.lower() for marker in MTP_MARKERS):
                out[path] = child
            out.update(collect_mtp_fields(child, path))
    elif isinstance(value, list):
        for i, child in enumerate(value[:50]):
            out.update(collect_mtp_fields(child, f"{prefix}[{i}]"))
    return out


def is_visual_tensor_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith("model.visual")
        or ".visual." in lower
        or any(marker in lower for marker in VISUAL_MARKERS[1:])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    args = ap.parse_args()

    model_dir = args.model_dir
    config = load_json(model_dir / "config.json")
    jang_path = model_dir / "jang_config.json"
    jang = load_json(jang_path) if jang_path.exists() else {}
    index_path = model_dir / "model.safetensors.index.json"
    index = load_json(index_path) if index_path.exists() else {}
    names = sorted(index.get("weight_map", {}).keys())

    mtp_keys = collect_mtp_fields(config)
    runtime = jang.get("runtime") or config.get("runtime") or {}
    jang_mtp_keys = collect_mtp_fields(jang)
    mtp_names = [
        n
        for n in names
        if n.lower().startswith("mtp.")
        or ".mtp" in n.lower()
        or "nextn" in n.lower()
        or "next_n" in n.lower()
        or "mtp_layer" in n.lower()
        or n.startswith("model.layers.80.")
    ]
    visual_names = [n for n in names if is_visual_tensor_name(n)]

    out = {
        "path": str(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "config_mtp_keys": mtp_keys,
        "jang_mtp_keys": jang_mtp_keys,
        "runtime": runtime,
        "index_present": bool(index),
        "tensor_count": len(names),
        "mtp_tensor_count": len(mtp_names),
        "mtp_tensor_samples": mtp_names[:40],
        "artifact_has_mtp_weights": bool(mtp_names),
        "visual_tensor_count": len(visual_names),
        "visual_tensor_samples": visual_names[:40],
        "artifact_has_vision_weights": bool(visual_names),
        "configured_mtp_layers": (
            mtp_keys.get("num_nextn_predict_layers")
            or mtp_keys.get("text_config.mtp_num_hidden_layers")
            or mtp_keys.get("mtp_num_hidden_layers")
        ),
        "mxtq_bits": jang.get("mxtq_bits") or config.get("mxtq_bits"),
        "capabilities": jang.get("capabilities") or config.get("capabilities"),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
