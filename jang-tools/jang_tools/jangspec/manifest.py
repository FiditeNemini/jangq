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
    """Load and validate a jangspec bundle manifest.

    M148 (iter 70): harden error reporting symmetrically with
    ``write_manifest`` and with the iter-43 M120 pattern on
    inspect_source/recommend. Pre-iter-70 a corrupted or
    schema-migrated bundle produced raw ``JSONDecodeError`` /
    ``TypeError: Manifest.__init__() missing 1 required positional
    argument`` tracebacks — opaque to the end user. The iter-70
    version:
      * Catches OSError / UnicodeDecodeError on the read so disk
        faults produce actionable stderr.
      * Catches JSONDecodeError and surfaces the bad file path
        and decode location.
      * Catches Manifest(**data) TypeError (missing/extra keys
        from a schema migration) and hints that the bundle was
        written by an older or newer tool version.
    Every error path includes the ``path`` so diagnostics are
    unambiguous when users have multiple bundles on disk.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read manifest at {p}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manifest at {p} is not valid JSON "
            f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"manifest at {p} has a top-level {type(data).__name__}, "
            f"expected a JSON object"
        )
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
