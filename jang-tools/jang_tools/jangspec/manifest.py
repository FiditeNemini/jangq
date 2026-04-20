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
from .._json_utils import read_json_object


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
    """Load and validate a jangspec bundle manifest.

    M148 (iter 70) added diagnostic error paths.
    M152 (iter 75): migrated from an inline re-implementation to the
    shared `.._json_utils.read_json_object` after 5 local copies
    crystallized the template. Behavior identical; errors still include
    the bundle path, decode location, and schema-migration hints.
    """
    p = Path(path)
    data = read_json_object(p, purpose="manifest")
    bv = data.get("bundle_version")
    if bv != fmt.BUNDLE_VERSION:
        raise ValueError(
            f"unsupported bundle_version {bv} at {p}, "
            f"this build supports {fmt.BUNDLE_VERSION}"
        )
    try:
        return Manifest(**data)
    except TypeError as exc:
        # Schema-migration mismatch: missing or extra field. Hint the user
        # about version drift so they know where to look.
        raise ValueError(
            f"manifest at {p} failed schema validation "
            f"(likely a bundle written by a different jang-tools version): {exc}"
        ) from exc
