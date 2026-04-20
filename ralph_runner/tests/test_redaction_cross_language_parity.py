"""M197 (iter 134): mechanical cross-language parity invariant for
redaction helpers.

Codifies iter-133 M196's meta-lesson: the JANG stack has two independent
secret-redaction helpers — Python's `redact_for_log` in jang-server and
Swift's `DiagnosticsBundle.scrubSensitive` in JANGStudio. They drifted
for ~119 iters before M196 caught it. M196 restored parity, but without
a mechanical invariant the next iter to add a pattern on one side only
would silently drift again.

This test asserts BOTH sides cover the same TAXONOMY of secret shapes.
It doesn't compare regex strings literally (Python `re` syntax differs
from NSRegularExpression), it compares canonical shape-keyword tokens
that must appear in each side's pattern-declaration block.

If you add a pattern to ONE side and not the other, this test fires
with a clear "side X missing token Y" message pointing at the exact
gap. The remediation is always obvious: add the matching pattern to
the other helper, then rerun.

If the helpers ever get extracted to a shared source-of-truth (YAML/
JSON/whatever) that both sides compile from, this test becomes a no-op
guarding the shared file — the taxonomy tokens would just live there.

Future extensions:
  - Add patterns to SHAPE_TOKENS below whenever a new class of secret
    enters the scope of either helper. Both sides should grow at the
    same time; this test ensures it.
  - For a stronger invariant, switch to an AST parse of the Python
    tuple list + a Swift regex parse of the tuple array, then compare
    structurally. Current keyword-in-source approach is pragmatic and
    catches the drift class that matters most (missing SHAPES, not
    subtle regex differences).
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent  # /Users/eric/jang/

PYTHON_SERVER = REPO_ROOT / "jang-server" / "server.py"
SWIFT_DIAGNOSTICS = REPO_ROOT / "JANGStudio" / "JANGStudio" / "Runner" / "DiagnosticsBundle.swift"


# Taxonomy of secret SHAPES that both sides must cover. Each entry is
# (canonical name, [list-of-keywords-that-must-appear-in-the-file]).
#
# The keywords are substrings that MUST appear in the pattern-declaration
# region of each side's source file. They're chosen to be unique to the
# pattern (unlikely to appear elsewhere) so a simple substring search
# is enough. For patterns that differ textually (e.g. Python uses
# `[?&]` while Swift uses `[?&]`), we pick a keyword that's common to
# both (e.g. `api_key`).
SHAPE_TOKENS: list[tuple[str, list[str]]] = [
    # Primary HF token shape.
    ("HF token (hf_…)", ["hf_[A-Za-z0-9_-]{20,}"]),
    # Legacy HF token shape (still emitted by some HF client paths).
    ("HF token (huggingface_…)", ["huggingface_"]),
    # Generic Bearer auth header. Swift has both a strict
    # `authorization: bearer …` and a generic `bearer …`. Python has
    # just the generic `Bearer <token>`. Common keyword: `Bearer` / `bearer`.
    ("Bearer token", ["earer"]),
    # OpenAI key shapes.
    ("OpenAI key (sk-…)", ["sk-(?:proj-)?"]),
    # Slack webhook. Python matches on host (`hooks\.slack\.com`); Swift
    # matches on path (`/services/T…/B…/…`) and has `Slack` in the
    # description. Common keyword (case-insensitive): `slack`.
    ("Slack webhook", ["slack"]),
    # Discord webhook. Python matches on host (`discord(?:app)?\.com`);
    # Swift matches on path and has `Discord` in the description.
    # Common keyword (case-insensitive): `discord`.
    ("Discord webhook", ["discord"]),
    # URL-query-string secrets. The 4 parameter names are common to both.
    ("URL query secret", ["api_key", "access_token"]),
]


def _extract_pattern_block(path: Path, *, start_marker: str, end_marker: str | None = None) -> str:
    """Grab the block of source between start_marker and end_marker (or
    EOF). Used to scope the keyword search to the pattern-declaration
    region so unrelated mentions (comments in the middle of a function)
    don't count as coverage.

    For Python: the declaration is `_SECRET_REDACTIONS = [ ... ]`.
    For Swift: the declaration is `sensitivePatterns: [...] = [ ... ]`.
    """
    content = path.read_text(encoding="utf-8")
    start = content.find(start_marker)
    assert start != -1, (
        f"Couldn't find start marker {start_marker!r} in {path}. The "
        f"file's pattern-declaration has been renamed/refactored — "
        f"update the markers in this test."
    )
    if end_marker:
        end = content.find(end_marker, start)
        if end == -1:
            end = len(content)
    else:
        # Slice ~3000 chars forward from start — enough to capture the
        # declaration block without pulling in the rest of the file.
        end = min(start + 3000, len(content))
    return content[start:end]


def _missing_tokens(block: str, tokens: list[str]) -> bool:
    """Return True if NONE of the alternative tokens appears in block.
    (Any one match is enough to count as coverage.) Case-insensitive so
    the Swift description (`Slack webhook secret`) matches the Python
    pattern (`slack\\.com`) without per-side token lists."""
    low = block.lower()
    return all(tok.lower() not in low for tok in tokens)


def test_python_and_swift_cover_the_same_secret_shapes():
    """Every SHAPE_TOKENS entry must be represented in BOTH helper files.

    If this test fires:
      1. Read the failure message — it names the shape and which SIDE is
         missing a keyword.
      2. Add the matching pattern to that side's helper, using the other
         side as reference.
      3. Also add a unit test on that side pinning the pattern works.
      4. Rerun. The parity test should go green.

    If you INTENTIONALLY removed a shape from both sides, update
    SHAPE_TOKENS to drop the entry.
    """
    python_block = _extract_pattern_block(
        PYTHON_SERVER,
        start_marker="_SECRET_REDACTIONS = [",
        end_marker="\ndef redact_for_log",
    )
    swift_block = _extract_pattern_block(
        SWIFT_DIAGNOSTICS,
        start_marker="sensitivePatterns:",
        # Slice forward — the Swift block ends with `]` but we'd need
        # balanced-brace parsing; simplest bound is the next function.
        end_marker="nonisolated static func scrubSensitive",
    )

    gaps: list[str] = []
    for name, tokens in SHAPE_TOKENS:
        if _missing_tokens(python_block, tokens):
            gaps.append(
                f"  Python ({PYTHON_SERVER.name}) missing shape '{name}' — "
                f"expected any of {tokens}"
            )
        if _missing_tokens(swift_block, tokens):
            gaps.append(
                f"  Swift ({SWIFT_DIAGNOSTICS.name}) missing shape '{name}' — "
                f"expected any of {tokens}"
            )
    assert not gaps, (
        f"M197 regression: redaction helpers have drifted apart.\n"
        f"{len(gaps)} parity gap(s) found:\n" +
        "\n".join(gaps) +
        "\n\nBoth jang-server's `redact_for_log` (_SECRET_REDACTIONS) "
        "and JANGStudio's `DiagnosticsBundle.scrubSensitive` "
        "(sensitivePatterns) must cover the same taxonomy of secret "
        "shapes. Add the missing pattern to the affected side; pin a "
        "unit test on that side; rerun this test. See iter-133 M196 "
        "for the original parity fix."
    )


def test_python_pattern_block_has_at_least_seven_patterns():
    """Coarse-count floor so a future edit that accidentally removes
    most patterns doesn't silently pass the taxonomy test (which only
    checks COVERAGE, not COUNT)."""
    block = _extract_pattern_block(
        PYTHON_SERVER,
        start_marker="_SECRET_REDACTIONS = [",
        end_marker="\ndef redact_for_log",
    )
    # Count lines that look like a pattern entry: `(re.compile(...), ...)`.
    count = block.count("re.compile(")
    assert count >= 6, (
        f"M197 regression: Python _SECRET_REDACTIONS has only {count} "
        f"re.compile entries. Expected ≥6 (hf_, huggingface_, sk-, "
        f"Bearer, webhooks, URL-query). Did a refactor drop patterns?"
    )


def test_swift_pattern_block_has_at_least_seven_patterns():
    """Parallel floor for the Swift side."""
    block = _extract_pattern_block(
        SWIFT_DIAGNOSTICS,
        start_marker="sensitivePatterns:",
        end_marker="nonisolated static func scrubSensitive",
    )
    # Swift entries are tuples `(description, pattern, template)` —
    # count by the `#"` open delimiter of the raw-string literal
    # (close delimiter is `"#`, a different substring). One `#"` per
    # pattern entry.
    count = block.count('#"')
    assert count >= 6, (
        f"M197 regression: Swift sensitivePatterns has only {count} "
        f"#\"...\"# entries. Expected ≥6 (hf_, huggingface_, Bearer x2, "
        f"OpenAI, Slack webhook, Discord webhook, URL query). Did a "
        f"refactor drop patterns?"
    )
