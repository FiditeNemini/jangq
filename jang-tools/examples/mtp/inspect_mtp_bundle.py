#!/usr/bin/env python3
"""Low-RAM MTP bundle inspector.

Reads config/index metadata only. Does not load tensor payloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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

    mtp_keys = {
        k: config.get(k)
        for k in sorted(config)
        if "mtp" in k.lower() or "nextn" in k.lower() or "speculative" in k.lower()
    }
    runtime = jang.get("runtime") or config.get("runtime") or {}
    mtp_names = [
        n
        for n in names
        if ".mtp" in n.lower()
        or "nextn" in n.lower()
        or "mtp_layer" in n.lower()
        or n.startswith("model.layers.80.")
    ]

    out = {
        "path": str(model_dir),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "config_mtp_keys": mtp_keys,
        "runtime": runtime,
        "index_present": bool(index),
        "tensor_count": len(names),
        "mtp_tensor_count": len(mtp_names),
        "mtp_tensor_samples": mtp_names[:40],
        "mxtq_bits": jang.get("mxtq_bits") or config.get("mxtq_bits"),
        "capabilities": jang.get("capabilities") or config.get("capabilities"),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
