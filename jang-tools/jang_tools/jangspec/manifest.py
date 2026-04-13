"""
jangspec.json — human-readable bundle manifest.

The manifest is small (a few KB). It exists so a human or the bundle builder
can quickly inspect a bundle without parsing the binary index. The Swift
runtime parses it at load time to determine which tensors are hot-core vs
streamed, and to verify target-draft compatibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from . import format as fmt


@dataclass
class Manifest:
    bundle_version: int
    source_jang: str
    source_jang_dir: str
    target_arch: str
    n_layers: int
    n_experts_per_layer: int
    target_top_k: int
    tokenizer_hash: str
    hot_core_tensors: List[str]
    expert_tensor_names: List[str]
    n_experts_total: int
    hot_core_bytes: int
    expert_bytes: int
    has_draft: bool
    has_router_prior: bool
    draft_jang: str = ""
    tool_version: str = "jang-spec-0.1.0"
    schema: str = "jangspec/v1"


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)
        f.write("\n")


def load_manifest(path: Path) -> Manifest:
    data = json.loads(Path(path).read_text())
    bv = data.get("bundle_version")
    if bv != fmt.BUNDLE_VERSION:
        raise ValueError(
            f"unsupported bundle_version {bv}, this build supports {fmt.BUNDLE_VERSION}"
        )
    return Manifest(**data)
