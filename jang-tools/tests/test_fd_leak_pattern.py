"""M125 (iter 48): regression guard against un-context-managed open() in json IO.

Background: `json.load(open(path))` and `json.dump(obj, open(path, "w"))`
rely on CPython's refcount-GC to close the file promptly. Under:
  - PyPy or other GC-delayed implementations: fd leaks accumulate.
  - Crash between dump() and GC: partial JSON on disk → corrupted model.
  - Dense script (hundreds of files in sequence): transient fd-limit breach.

Iter 48 migrated all 24 known call sites in jang_tools/ to
``with open(...) as f:``. This test pins that migration — it fails
loudly if a future edit reintroduces the unsafe pattern.
"""
from __future__ import annotations

import re
from pathlib import Path


PROD_ROOT = Path(__file__).resolve().parents[1] / "jang_tools"

# Unsafe: json.load(open(...)) / json.dump(..., open(...)) with no context
# manager in scope. A simple regex that matches the same line is enough
# because all historical instances were single-line.
UNSAFE_RX = re.compile(r"json\.(load|dump)\([^)]*open\(")


def _py_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_unwrapped_json_load_or_dump_with_open():
    offenders: list[str] = []
    for py in _py_files(PROD_ROOT):
        text = py.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            # Skip comment-only lines (the M125 rationale is referenced in
            # a comment in convert_minimax_jangtq.py; the grep below must
            # not trip on it).
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if UNSAFE_RX.search(line):
                offenders.append(f"{py.relative_to(PROD_ROOT)}:{line_no}: {stripped}")

    assert not offenders, (
        "Un-context-managed json IO detected. Wrap in `with open(...) as f:` "
        "so the file descriptor closes deterministically — see M125 (iter 48) "
        "in ralph_runner/AUDIT_CHECKLIST.md. Offenders:\n  "
        + "\n  ".join(offenders)
    )
