"""M177 (iter 111): jang-server exception-handling invariant.

Companion to iter-105 M113 (jang-tools) and iter-106 M119 (ralph_runner).
Same dual-invariant strategy: coarse count threshold catches bulk
regressions; precise regex catches the specific silent-swallow anti-
pattern (`except Exception[: as x]:\\n    pass` with no other body).

**Server context matters:** a long-running daemon's silent-swallow bugs
are WORSE than one-shot CLI tools because they accumulate over hours/
days of uptime with no user watching. Iter-111 fixed 3 of 4 bare swallow
sites by adding `log.warning(...)` before the implicit fall-through so
operators debugging a production issue have breadcrumbs.

**Remaining allowlisted site (1):** progress-percentage calculation at
line 1113 — bytes_total=0 would raise ZeroDivisionError; micro-defense
for a progress-meter tick-loop is acceptable noise-free. Other pass
sites are now log.warning().
"""
from __future__ import annotations

import re
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def test_except_exception_site_count_within_threshold():
    """Coarse count invariant. Today: 10. Threshold: 20 (10 headroom)."""
    content = SERVER_PY.read_text(encoding="utf-8")
    total = len(re.findall(r"\bexcept\s+Exception\b", content))
    assert total <= 20, (
        f"jang-server except Exception count ({total}) exceeds threshold. "
        f"Audit per iter-105 M113's 5-category taxonomy + iter-106 M119's "
        f"logging guidance. Server-context rule: if you're adding a new "
        f"`except Exception: pass`, prefer `except Exception as e: "
        f"log.warning(...)` so operators debugging production failures have "
        f"breadcrumbs (iter-111 M177 meta-lesson)."
    )


def test_no_bare_except_exception_pass_in_server():
    """Precise anti-pattern regex. Matches iter-105/106 tests."""
    content = SERVER_PY.read_text(encoding="utf-8")
    pattern = re.compile(
        r"except\s+Exception(?:\s+as\s+\w+)?\s*:\s*\n(\s+)pass\s*\n(?!\1)",
        re.MULTILINE,
    )
    offenders: list[int] = []
    for m in pattern.finditer(content):
        line_no = content[: m.start()].count("\n") + 1
        offenders.append(line_no)
    # Allowlist: progress-pct calculation at ~1113 is acceptable — a
    # ZeroDivisionError on bytes_total=0 shouldn't spam logs every tick.
    # Line numbers may shift across edits; match by approximate line.
    # M190 (iter 127): refactored from a line-number range allowlist to
    # a function-context allowlist. Iter-125's 4-bump rule fired at
    # iter-126 — line shifted four times (1121 → 1150 → 1207 → 1300)
    # and was on track for a 5th. Function-body slicing is immune to
    # line-number shifts as long as the function name + the bare-pass
    # line both stay inside the same def block.
    #
    # Allowed function NAMES whose bodies legitimately contain
    # `except Exception: pass` (per iter-106 M119 progress-tick-guard
    # taxonomy bucket). New entries need a rationale comment.
    allowed_function_bodies = {
        "_phase_download": "tick-loop progress-pct guard — bytes_total=0 "
            "ZeroDivisionError must not spam logs every tick"
    }
    # Compute (start_line, end_line) for each allowed function via simple
    # `def name(` scan. End line = next top-level def/class or EOF.
    func_ranges: dict[str, tuple[int, int]] = {}
    src_lines = SERVER_PY.read_text(encoding="utf-8").split("\n")
    current_func: str | None = None
    current_start: int = 0
    for i, line in enumerate(src_lines, start=1):
        # New top-level def closes the previous function range.
        if line.startswith("def ") or line.startswith("async def "):
            # Close previous if it was an allowed func.
            if current_func and current_func in allowed_function_bodies:
                func_ranges.setdefault(current_func, (current_start, i - 1))
            # Extract `def NAME(` or `async def NAME(`.
            after = line.removeprefix("async ").removeprefix("def ")
            current_func = after.split("(")[0].strip()
            current_start = i
    # Close the last function in the file.
    if current_func and current_func in allowed_function_bodies:
        func_ranges.setdefault(current_func, (current_start, len(src_lines)))

    # Build the line-set from the resolved function ranges.
    allowed_lines: set[int] = set()
    for func_name, (start, end) in func_ranges.items():
        allowed_lines.update(range(start, end + 1))

    # Sanity: every allowed function name must have actually resolved.
    # Catches a renamed/removed function before the next test bump
    # quietly hides a real new bare-pass.
    missing = set(allowed_function_bodies) - set(func_ranges)
    assert not missing, (
        f"M190 invariant: allowed_function_bodies references functions not "
        f"found in server.py: {missing}. Either the function was renamed "
        f"(update the allowlist key) or removed (drop the entry + audit "
        f"whether the bare-pass moved)."
    )

    remaining = [ln for ln in offenders if ln not in allowed_lines]
    assert not remaining, (
        f"Found {len(remaining)} new bare `except Exception: pass` sites in "
        f"jang-server/server.py at lines {remaining}. In a server context, "
        f"prefer `log.warning(...)` before the fall-through so operators "
        f"have debugging breadcrumbs (iter-111 M177 pattern)."
    )
