"""Verifier for JANGTQ-PRESTACK bundles (spec: research/JANGTQ-PRESTACK-SPEC.md).

This module hard-fails on any of:
  - per-expert TQ keys (forbidden post-2026-05-04)
  - missing tq_packed/tq_norms/tq_bits triplets
  - wrong ndim (packed must be 3D, norms must be 2D, bits must be 1D)
  - wrong leading dim (must equal n_routed_experts from config.json)
  - sidecar pollution (jangtq_stacked.safetensors/.json in bundle dir)
  - module replacement count mismatch (when --check-load is set)

Usage (CLI):
    python -m jang_tools.verify_jangtq_prestacked /path/to/bundle [--check-load]

Usage (programmatic):
    from jang_tools.verify_jangtq_prestacked import verify, VerificationError
    report = verify(bundle_path, check_load=False)  # raises VerificationError on hard-fail
    print(report.summary())
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Per-expert key patterns that are FORBIDDEN under the prestack spec.
# Any tensor matching one of these is a spec violation.
_FORBIDDEN_PER_EXPERT = re.compile(
    r"\.experts\.\d+\.(?:w[123]|gate_proj|up_proj|down_proj)\.tq_(?:packed|norms|bits)$"
)

# Allowed prestacked patterns (3 keys per MoE layer × projection).
_PRESTACK_KEY = re.compile(
    r"^("
    r"(?:[^.]+\.)*"  # any prefix
    r")"
    r"switch_mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)"
    r"\.(tq_packed|tq_norms|tq_bits)$"
)

# Sidecar files that must NOT appear in the bundle dir.
_SIDECAR_FILES = ("jangtq_stacked.safetensors", "jangtq_stacked.json")


class VerificationError(Exception):
    """Raised when a bundle fails JANGTQ-PRESTACK spec verification."""

    def __init__(self, message: str, errors: List[str]):
        super().__init__(message)
        self.errors = errors


@dataclass
class VerificationReport:
    bundle_path: Path
    n_routed_experts: Optional[int] = None
    moe_prefix: Optional[str] = None
    prestack_triplets: int = 0  # count of (prefix, proj) groups with full triplet
    incomplete_triplets: List[str] = field(default_factory=list)
    forbidden_keys: List[str] = field(default_factory=list)
    bad_shape_keys: List[str] = field(default_factory=list)
    sidecar_files: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"JANGTQ-PRESTACK verification: {self.bundle_path}",
            f"  n_routed_experts:    {self.n_routed_experts}",
            f"  prestack triplets:   {self.prestack_triplets}",
            f"  forbidden keys:      {len(self.forbidden_keys)}",
            f"  incomplete triplets: {len(self.incomplete_triplets)}",
            f"  bad-shape keys:      {len(self.bad_shape_keys)}",
            f"  sidecar files:       {len(self.sidecar_files)}",
            f"  warnings:            {len(self.warnings)}",
            f"  errors:              {len(self.errors)}",
            f"  PASSED" if self.passed else f"  FAILED",
        ]
        return "\n".join(lines)


def _read_index_or_listing(bundle_path: Path) -> dict:
    """Return {tensor_key: shard_filename}. Prefer model.safetensors.index.json
    when present (shard map). Otherwise scan single-file safetensors.
    """
    idx_path = bundle_path / "model.safetensors.index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text())
        return idx.get("weight_map", {})
    # Fallback — single-shard or non-MoE bundle. Still verifiable but rare.
    out = {}
    for sf in sorted(bundle_path.glob("*.safetensors")):
        try:
            from safetensors import safe_open
        except ImportError as e:
            raise VerificationError(
                "safetensors required for verification", [str(e)]
            )
        with safe_open(str(sf), framework="numpy") as f:
            for k in f.keys():
                out[k] = sf.name
    return out


def _read_n_routed_experts(bundle_path: Path) -> Optional[int]:
    cfg_path = bundle_path / "config.json"
    if not cfg_path.is_file():
        return None
    cfg = json.loads(cfg_path.read_text())
    # Common names across architectures
    for key in (
        "n_routed_experts",
        "num_experts",
        "num_local_experts",
        "n_routed",
    ):
        v = cfg.get(key)
        if isinstance(v, int) and v > 0:
            return v
    # Nested in text_config for VLM wrappers
    tc = cfg.get("text_config")
    if isinstance(tc, dict):
        for key in ("n_routed_experts", "num_experts", "num_local_experts"):
            v = tc.get(key)
            if isinstance(v, int) and v > 0:
                return v
    return None


def _shape_of(bundle_path: Path, shard_name: str, key: str):
    """Return (shape, dtype_str) of a tensor without loading data."""
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise VerificationError("safetensors required", [str(e)])
    with safe_open(str(bundle_path / shard_name), framework="numpy") as f:
        slc = f.get_slice(key)
        return tuple(slc.get_shape()), slc.get_dtype()


def verify(
    bundle_path,
    *,
    check_load: bool = False,
    expected_replacement_count: Optional[int] = None,
) -> VerificationReport:
    """Verify that `bundle_path` is a spec-compliant JANGTQ-PRESTACK bundle.

    Args:
        bundle_path: directory containing model.safetensors[.index.json] +
            config.json + jang_config.json.
        check_load: when True, attempt to load the bundle through
            `jang_tools.load_jangtq` and assert that the count of replaced
            TurboQuantSwitchLinear modules matches `expected_replacement_count`
            (or is consistent with the prestack triplet count when None).
        expected_replacement_count: optional override for the load-time
            module count check. When None, expects 1 module per (prefix, proj)
            triplet (i.e., one TurboQuantSwitchLinear per MoE layer × projection).

    Returns: VerificationReport. Raises VerificationError on hard-fail.
    """
    bundle_path = Path(bundle_path)
    report = VerificationReport(bundle_path=bundle_path)

    if not bundle_path.is_dir():
        report.errors.append(f"bundle path is not a directory: {bundle_path}")
        raise VerificationError(report.summary(), report.errors)

    # 1. Sidecar pollution
    for sf_name in _SIDECAR_FILES:
        if (bundle_path / sf_name).exists():
            report.sidecar_files.append(sf_name)
    if report.sidecar_files:
        report.errors.append(
            f"sidecar files present (forbidden by spec §3): "
            f"{report.sidecar_files}"
        )

    # 2. Read shard index
    weight_map = _read_index_or_listing(bundle_path)
    if not weight_map:
        report.errors.append(
            "no tensors found (no model.safetensors.index.json and no "
            "*.safetensors in bundle dir)"
        )
        raise VerificationError(report.summary(), report.errors)

    # 3. n_routed_experts from config
    report.n_routed_experts = _read_n_routed_experts(bundle_path)
    if report.n_routed_experts is None:
        report.warnings.append(
            "could not determine n_routed_experts from config.json — "
            "leading-dim shape check skipped"
        )

    # 4. Forbidden per-expert keys + prestack triplet collection
    triplets: dict[tuple[str, str], dict[str, str]] = {}
    for key in weight_map:
        if _FORBIDDEN_PER_EXPERT.search(key):
            report.forbidden_keys.append(key)
            continue
        m = _PRESTACK_KEY.match(key)
        if m:
            prefix, proj, suffix = m.group(1), m.group(2), m.group(3)
            triplets.setdefault((prefix, proj), {})[suffix] = key
            if report.moe_prefix is None:
                report.moe_prefix = prefix.rstrip(".")

    if report.forbidden_keys:
        report.errors.append(
            f"per-expert TQ keys present ({len(report.forbidden_keys)}); spec "
            f"forbids any `.experts.<E>.<proj>.tq_*`. First 3: "
            f"{report.forbidden_keys[:3]}"
        )

    # 5. Triplet completeness + shape validation
    REQUIRED = {"tq_packed", "tq_norms", "tq_bits"}
    for (prefix, proj), suffix_map in triplets.items():
        present = set(suffix_map.keys())
        missing = REQUIRED - present
        if missing:
            report.incomplete_triplets.append(
                f"{prefix}switch_mlp.{proj}: missing {sorted(missing)}"
            )
            continue
        # Shape validation
        try:
            packed_shape, packed_dt = _shape_of(
                bundle_path, weight_map[suffix_map["tq_packed"]], suffix_map["tq_packed"]
            )
            norms_shape, norms_dt = _shape_of(
                bundle_path, weight_map[suffix_map["tq_norms"]], suffix_map["tq_norms"]
            )
            bits_shape, bits_dt = _shape_of(
                bundle_path, weight_map[suffix_map["tq_bits"]], suffix_map["tq_bits"]
            )
        except Exception as e:
            report.bad_shape_keys.append(
                f"{prefix}switch_mlp.{proj}: shape probe failed: {e}"
            )
            continue
        if len(packed_shape) != 3:
            report.bad_shape_keys.append(
                f"{suffix_map['tq_packed']}: ndim={len(packed_shape)} (expected 3)"
            )
        if len(norms_shape) != 2:
            report.bad_shape_keys.append(
                f"{suffix_map['tq_norms']}: ndim={len(norms_shape)} (expected 2)"
            )
        if len(bits_shape) not in (1,):
            report.bad_shape_keys.append(
                f"{suffix_map['tq_bits']}: ndim={len(bits_shape)} (expected 1)"
            )
        # Leading dim must == n_routed_experts (when known)
        if report.n_routed_experts is not None:
            n_exp = report.n_routed_experts
            if len(packed_shape) >= 1 and packed_shape[0] != n_exp:
                # gate_up_proj fuses gate+up — leading dim still n_exp.
                # If converter wrote concatenated-row layout, leading dim
                # is still n_exp (out dim doubles). Only fail when leading
                # dim differs from expert count.
                report.bad_shape_keys.append(
                    f"{suffix_map['tq_packed']}: leading dim={packed_shape[0]} "
                    f"(expected n_routed_experts={n_exp})"
                )
            if len(norms_shape) >= 1 and norms_shape[0] != n_exp:
                report.bad_shape_keys.append(
                    f"{suffix_map['tq_norms']}: leading dim={norms_shape[0]} "
                    f"(expected n_routed_experts={n_exp})"
                )
        report.prestack_triplets += 1

    if report.incomplete_triplets:
        report.errors.append(
            f"incomplete prestack triplets ({len(report.incomplete_triplets)}); "
            f"first 3: {report.incomplete_triplets[:3]}"
        )
    if report.bad_shape_keys:
        report.errors.append(
            f"bad-shape prestack tensors ({len(report.bad_shape_keys)}); "
            f"first 3: {report.bad_shape_keys[:3]}"
        )

    # 6. Optional load-time module count check
    if check_load and not report.errors:
        try:
            from .load_jangtq import load_jangtq_model
            from .turboquant.linear import TurboQuantSwitchLinear

            model, _tok = load_jangtq_model(str(bundle_path), skip_params_eval=True)
            actual_count = sum(
                1 for _, m in model.named_modules()
                if isinstance(m, TurboQuantSwitchLinear)
            )
            expected = (
                expected_replacement_count
                if expected_replacement_count is not None
                else report.prestack_triplets
            )
            if actual_count != expected:
                report.errors.append(
                    f"module replacement count mismatch: "
                    f"TurboQuantSwitchLinear found={actual_count}, expected={expected}"
                )
        except Exception as e:
            report.errors.append(f"load-time check failed: {e}")

    if report.errors:
        raise VerificationError(report.summary(), report.errors)
    return report


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="Verify a JANGTQ-PRESTACK bundle")
    p.add_argument("bundle", help="path to bundle directory")
    p.add_argument(
        "--check-load",
        action="store_true",
        help="actually load the bundle and verify module replacement count",
    )
    p.add_argument(
        "--expected-replacement-count",
        type=int,
        default=None,
        help="explicit module count for the load-time check",
    )
    args = p.parse_args(argv)
    try:
        report = verify(
            args.bundle,
            check_load=args.check_load,
            expected_replacement_count=args.expected_replacement_count,
        )
        print(report.summary())
        return 0
    except VerificationError as e:
        print(e.args[0], file=sys.stderr)
        for err in e.errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
