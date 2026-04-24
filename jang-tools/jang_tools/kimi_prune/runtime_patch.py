"""Apply JANG's MLA-absorb fp32 SDPA fix to the main python mlx_lm install.

Kimi K2.6 (`kimi_k25` model_type in mlx_lm) wraps `DeepseekV3Model` from
`mlx_lm.models.deepseek_v3`. That file's MLA attention has the same L==1
absorb branch as `deepseek_v32` and the same bf16 SDPA drift bug — without
this patch, decode produces repetition loops after ~14 tokens on quantized
Kimi bundles. See `research/VMLX-RUNTIME-FIXES.md` for the full history
(originally found on GLM-5.1 JANG_1L).

WHAT THIS DOES:
  Copies `research/deepseek_v3_patched.py` over the MAIN pip install
  (`/opt/homebrew/lib/python3.14/site-packages/mlx_lm/models/deepseek_v3.py`).
  Keeps a timestamped backup next to the original the first time it runs.

WHAT THIS DOES NOT DO:
  - DOES NOT touch vMLX's bundled mlx_lm (vMLX patches via its own source
    tree, per the 2026-04-11 rule).
  - DOES NOT touch any conda / other-venv installs.

IDEMPOTENT: if the patch marker is already present, this is a no-op.

Usage:
    python -m jang_tools.kimi_prune.runtime_patch [--dry-run] [--restore]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from importlib.util import find_spec
from pathlib import Path


PATCH_MARKER = "JANG fast fix (2026-04-22): MLA absorb path at bf16 drifts"
PATCH_SRC = (
    Path(__file__).resolve().parents[2].parent / "research" / "deepseek_v3_patched.py"
)


def _locate_target() -> Path:
    spec = find_spec("mlx_lm.models.deepseek_v3")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "mlx_lm.models.deepseek_v3 is not importable — install `mlx-lm` first."
        )
    target = Path(spec.origin)
    if "vmlx" in str(target).lower() or "/vmlx-" in str(target):
        raise RuntimeError(
            f"Refusing to patch vMLX-bundled file: {target}\n"
            "vMLX has its own deepseek_v3.py; patch it via the vMLX source tree "
            "(see research/VMLX-RUNTIME-FIXES.md)."
        )
    return target


def _has_patch(path: Path) -> bool:
    try:
        return PATCH_MARKER in path.read_text(errors="replace")
    except FileNotFoundError:
        return False


def apply(dry_run: bool = False) -> int:
    if not PATCH_SRC.exists():
        print(f"[patch] ERROR: patched source not found: {PATCH_SRC}", file=sys.stderr)
        return 2

    target = _locate_target()
    print(f"[patch] target: {target}")
    print(f"[patch] source: {PATCH_SRC}")

    if _has_patch(target):
        print("[patch] already applied (marker found) — nothing to do.")
        return 0

    if dry_run:
        print("[patch] DRY-RUN: would copy patched file over target.")
        return 0

    # Back up once with a timestamp. Don't overwrite an existing backup.
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = target.with_name(target.name + f".jang-backup-{ts}")
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[patch] backup: {backup}")
    else:
        print(f"[patch] backup exists: {backup}")

    shutil.copy2(PATCH_SRC, target)
    post_md5 = hashlib.md5(target.read_bytes()).hexdigest()
    print(f"[patch] applied. target md5: {post_md5}")
    # Quick sanity: re-import to surface any syntax issue immediately.
    import importlib, mlx_lm.models.deepseek_v3 as _d3

    importlib.reload(_d3)
    print("[patch] import-check OK")
    return 0


def restore() -> int:
    """Restore the most recent .jang-backup-* next to the target."""
    target = _locate_target()
    backups = sorted(target.parent.glob(target.name + ".jang-backup-*"))
    if not backups:
        print("[patch] no backup found — nothing to restore.", file=sys.stderr)
        return 2
    latest = backups[-1]
    print(f"[patch] restoring from: {latest}")
    shutil.copy2(latest, target)
    print("[patch] done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    if args.restore:
        return restore()
    return apply(dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
