"""Audit capabilities stamps across local artifacts and/or HF repos.

Usage:
  # Audit a single dir (or several)
  python3 -m jang_tools.verify_capabilities /path/to/model [/path ...]

  # Auto-discover every JANG model under common roots
  python3 -m jang_tools.verify_capabilities --discover

  # Discover + machine-readable JSON output (one line per model)
  python3 -m jang_tools.verify_capabilities --discover --json

Exit code is 0 only if every checked directory passes verify_directory.
Use this before any HF push, in CI, or after large batch conversions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from jang_tools.capabilities import verify_directory


_DEFAULT_ROOTS = [
    Path.home() / "models",
    Path.home() / ".mlxstudio",
]
# `~/.cache/huggingface/hub` is intentionally NOT in the default roots.
# That cache stores immutable content-addressed snapshots from `hf download`
# pulls — old snapshots fail verification because they predate the
# capabilities stamp, but they aren't writable and don't represent the live
# state of any HF repo. Pass `--include-hf-cache` if you really want to walk
# those (e.g. to find which models you've cached locally).


def discover(roots: list[Path], include_hf_cache: bool = False) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for jcf in root.rglob("jang_config.json"):
            found.append(jcf.parent)
    if include_hf_cache:
        hf_cache = Path.home() / ".cache/huggingface/hub"
        if hf_cache.exists():
            for jcf in hf_cache.rglob("jang_config.json"):
                found.append(jcf.parent)
    # Dedupe preserving order
    seen = set()
    uniq = []
    for p in found:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return uniq


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m jang_tools.verify_capabilities",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="Model directories to verify (omit when using --discover).",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="Walk default roots and audit every jang_config.json found.",
    )
    parser.add_argument(
        "--include-hf-cache", action="store_true",
        help="Also walk ~/.cache/huggingface/hub (off by default — those "
             "snapshots are immutable HF pull cache, not live repos).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="One JSON line per model: {path, ok, message}.",
    )
    args = parser.parse_args(argv)

    targets: list[Path] = list(args.paths)
    if args.discover:
        targets += discover(_DEFAULT_ROOTS, include_hf_cache=args.include_hf_cache)
    if not targets:
        parser.print_help()
        return 2

    failures = 0
    passes = 0
    for d in targets:
        if not d.is_dir():
            failures += 1
            row = {"path": str(d), "ok": False, "message": "not a directory"}
        else:
            ok, msg = verify_directory(d)
            row = {"path": str(d), "ok": ok, "message": msg}
            if ok:
                passes += 1
            else:
                failures += 1
        if args.json:
            print(json.dumps(row))
        else:
            tag = "PASS" if row["ok"] else "FAIL"
            print(f"  [{tag}] {row['path']}: {row['message']}")

    if not args.json:
        print(f"\nTotal: {len(targets)}  PASS={passes}  FAIL={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
