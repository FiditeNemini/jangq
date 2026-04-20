"""M119 (closed iter 106): coarse + precise invariants for `except
Exception` sites across ralph_runner/.

Companion to iter-105 M113's jang-tools test. Same dual-invariant
strategy: coarse count threshold catches bulk regressions; precise
regex catches the specific silent-swallow anti-pattern
(`except Exception[: as x]:\\n    pass` with no other body).

**Current distribution (iter-106 count):** 36 sites — 34 in audit.py
(one `except Exception as e:` per audit row, which wraps subprocess +
analysis errors into structured fail results — intentional + correct
per M119's original observation), 2 in runner.py.

**Taxonomy of acceptable patterns:**

1. **Audit-row error isolation** — each `@register_audit` row catches
   its own Exception so a crash in row N doesn't kill the whole audit
   sweep. This is audit.py's dominant pattern; the exception is logged
   into the row's result with `status="error"`. Similar to category 4
   of M113 (error-wrapping-with-context).

2. **Subprocess probe fallbacks** — runner.py's paths probe git / pip /
   ssh tooling and fall back if any raises. Analogous to M113's
   "optional imports" category.

3. **CLI top-level catch** — a few `__main__`-level blocks that emit
   clean CLI error output instead of a traceback.

**The anti-pattern is still `except Exception: pass` with no log / no
re-raise.** Same iter-35 M107 / iter-80 M157 / iter-90 M167 class as
the Swift and Python-tools side.
"""
from __future__ import annotations

import re
from pathlib import Path


RALPH_DIR = Path(__file__).parent.parent
PY_FILES = [p for p in RALPH_DIR.glob("*.py") if "__pycache__" not in p.parts]


def _count_except_exception_sites() -> int:
    total = 0
    for py in PY_FILES:
        content = py.read_text(encoding="utf-8", errors="replace")
        total += len(re.findall(r"\bexcept\s+Exception\b", content))
    return total


def test_except_exception_site_count_within_threshold():
    """Coarse count invariant. Today: 36. Threshold: 50 (14 headroom)."""
    total = _count_except_exception_sites()
    assert total <= 50, (
        f"ralph_runner except Exception count ({total}) exceeds threshold — "
        f"audit new additions per the 3-category taxonomy in this module's "
        f"docstring (M119 iter 106). If all fit, bump threshold. If any is "
        f"bare `except Exception: pass` with no log/re-raise, fix per iter-35 "
        f"M107 / iter-90 M167 pattern."
    )


def test_no_bare_except_exception_pass_in_ralph_runner():
    """Precise anti-pattern regex. Matches iter-105 M113's jang-tools test.

    Pattern: `except Exception[: as x]:\\n<indent>pass<EOL>` with no other
    body. Catches the silent-swallow anti-pattern iter-35 M107 fixed on
    the Swift side.
    """
    offenders: list[tuple[str, int]] = []
    pattern = re.compile(
        r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*\n(\s+)pass\s*\n(?!\1)",
        re.MULTILINE,
    )
    for py in PY_FILES:
        content = py.read_text(encoding="utf-8", errors="replace")
        for m in pattern.finditer(content):
            line_no = content[: m.start()].count("\n") + 1
            offenders.append((str(py.relative_to(RALPH_DIR)), line_no))
    # Allowlist for legit best-effort operations — populate with rationale
    # if any appear.
    allowed: set[tuple[str, int]] = set()
    remaining = [o for o in offenders if o not in allowed]
    assert not remaining, (
        f"Found {len(remaining)} bare `except Exception: pass` sites in "
        f"ralph_runner — silent swallows with no log / re-raise. Fix per "
        f"iter-35 M107 / iter-90 M167 pattern:\n" +
        "\n".join(f"  {f}:{l}" for f, l in remaining)
    )
