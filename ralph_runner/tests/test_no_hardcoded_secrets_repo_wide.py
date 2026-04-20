"""M182 (iter 117): repo-wide regression invariant against hardcoded
secrets. Extends iter-116 M181's jang-server-scoped check to every
.py and .swift file across the repo.

Why repo-wide: M181 caught a leaked HF write-token in server.py — but
without a cross-cutting invariant, the same class of bug could appear
in jang-tools, JANGStudio Swift, ralph_runner, or any future module
without anyone noticing until another audit iter.

**Patterns checked** (matches iter-116 M181 + standard secret shapes):
  - HF tokens: `hf_<20+ chars>` (current format)
  - HF legacy: `huggingface_<20+ chars>`
  - OpenAI: `sk-<20+ chars>` (covers both old `sk-<key>` and `sk-proj-<key>`)
  - AWS access keys: `AKIA<16 chars>` (root) and `ASIA<16 chars>` (STS)
  - GitHub tokens: `gh[pousr]_<36 chars>`

**Allowlisted false-positive shapes:**
  - Test fixtures using clearly-fake hf_abcdef..., hf_SECRET..., etc.
  - DiagnosticsBundle.scrubSensitive's regex source string itself.
  - Comments / docstrings explicitly mentioning a redacted token shape.
  - Third-party vendored code (build/, .venv/, node_modules/).

**When this test fires:**
  1. Identify the file:line.
  2. If a real secret: rotate at the source-of-truth (HF tokens page,
     AWS console, etc.) BEFORE removing from source — anyone with
     repo or git-history access already has it.
  3. Replace the source occurrence with `os.environ.get("X", "")`
     (Python) or `ProcessInfo.processInfo.environment["X"]` (Swift)
     and fail-fast at use-site if missing.
  4. If a clearly-fake test fixture, add it to the per-pattern
     allowlist below WITH RATIONALE.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent  # /Users/eric/jang/

# Directories to skip — vendored third-party + build outputs.
SKIP_DIR_NAMES = {
    "__pycache__", ".git", ".venv", "venv", "build", "Build",
    "DerivedData", "node_modules", "site-packages",
    # M184 (iter 119): SwiftPM .build/ output. Dotted-prefix dirs need
    # explicit entry — `.build` doesn't match the lowercase `build`
    # entry above (Path.parts compares whole components). Pre-fix the
    # M182/M183 sweeps scanned 569 generated files inside JANGQuantizer.
    # swiftpm/.build/ on every run, slowing the test 5× and risking
    # false positives from shape-matching identifiers in generated code.
    ".build",
    # Other common dotted build/cache dirs worth pre-emptively skipping.
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    # NOTE: do NOT skip ".swiftpm" — that's a Swift Package CONTAINER
    # directory (like .app), and Sources/ lives INSIDE it. We want to
    # audit those Swift files. Only the `.build/` subdir within is
    # generated output and is already covered above.
}


SECRET_PATTERNS = [
    # name, regex, brief description (used in failure message)
    ("HF token (hf_*)", re.compile(r"\bhf_[A-Za-z0-9_-]{20,}\b"),
        "HuggingFace token — rotate at https://huggingface.co/settings/tokens"),
    ("HF legacy (huggingface_*)", re.compile(r"\bhuggingface_[A-Za-z0-9_-]{20,}\b"),
        "Older HuggingFace token format — rotate as above"),
    ("OpenAI key (sk-*)", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
        "OpenAI key — rotate at https://platform.openai.com/api-keys"),
    ("AWS access key (AKIA / ASIA)", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
        "AWS access key — rotate via IAM console + audit CloudTrail"),
    ("GitHub token (ghp / gho / etc.)", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
        "GitHub token — rotate at https://github.com/settings/tokens"),
]


# Per-pattern allowlist: clearly-fake fixtures that look like real
# secrets but are test data. (relative_path, regex_pattern_name) tuples.
ALLOWED_FIXTURES: set[tuple[str, str]] = {
    # iter-88 M165 + iter-87 M164 test fixtures using fake hf_abcdef...
    # tokens to exercise scrub-sensitive + URL-validator regex paths.
    ("JANGStudio/Tests/JANGStudioTests/DiagnosticsBundleTests.swift", "HF token (hf_*)"),
    # iter-105 M113's fake_test fixture in jang-tools tests.
    ("jang-tools/tests/test_exception_handling_invariant.py", "HF token (hf_*)"),
    # iter-116 M181's regex literal inside the test itself.
    ("jang-server/tests/test_no_hardcoded_secrets.py", "HF token (hf_*)"),
    # This file itself — the regex SOURCE strings are token-shaped.
    ("ralph_runner/tests/test_no_hardcoded_secrets_repo_wide.py", "HF token (hf_*)"),
    # DiagnosticsBundle.scrubSensitive contains the regex source string.
    ("JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift", "HF token (hf_*)"),
    ("JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift", "HF legacy (huggingface_*)"),
    # iter-29 M91 publish CLI test uses literal-looking fake tokens to
    # verify the file-path-vs-token-string disambiguation + token
    # scrubbing in stderr. Clearly fake (`literal_looking`, `dummy_test`).
    ("jang-tools/tests/test_publish.py", "HF token (hf_*)"),
    # DiagnosticsBundle scrub-sensitive regression test for the older
    # huggingface_* token format. Clearly fake (`abcdef_ghij...`).
    ("JANGStudio/Tests/JANGStudioTests/DiagnosticsBundleTests.swift", "HF legacy (huggingface_*)"),
    # iter-118 M183 extension to non-source files surfaces our own audit
    # docs as offenders — the docs reference test-fixture token names
    # by literal value when explaining the M181/M182/M183 fixes.
    # These are audit documentation, not real secret leaks. Allowlisted
    # at the doc-file level for both regex flavors.
    ("ralph_runner/AUDIT_CHECKLIST.md", "HF token (hf_*)"),
    ("ralph_runner/AUDIT_CHECKLIST.md", "HF legacy (huggingface_*)"),
    ("ralph_runner/INVESTIGATION_LOG.md", "HF token (hf_*)"),
    ("ralph_runner/INVESTIGATION_LOG.md", "HF legacy (huggingface_*)"),
    # iter-130 M193 redaction unit tests use literal fake HF/HF-legacy/
    # OpenAI/Bearer/query-string values to exercise redact_for_log.
    # Clearly-fake (all A-Z + 1234567890 pattern).
    ("jang-server/tests/test_runtime_log_redaction.py", "HF token (hf_*)"),
    ("jang-server/tests/test_runtime_log_redaction.py", "HF legacy (huggingface_*)"),
    ("jang-server/tests/test_runtime_log_redaction.py", "OpenAI key (sk-*)"),
    # iter-132 M195 DB backfill tests seed temp-DB rows with fake tokens
    # to verify the backfill redacts them.
    ("jang-server/tests/test_db_backfill_redaction.py", "HF token (hf_*)"),
    # iter-133 M196 Swift DiagnosticsBundleTests added OpenAI parity
    # fixtures — same clearly-fake pattern as the existing HF fixtures
    # already allowlisted above.
    ("JANGStudio/Tests/JANGStudioTests/DiagnosticsBundleTests.swift", "OpenAI key (sk-*)"),
}


def _iter_source_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.suffix not in (".py", ".swift"):
            continue
        yield p


# M183 (iter 118): extend M182 to config + script + doc files. JSON
# configs and shell scripts are common hardcode-leak vectors (CI env
# files, deploy scripts, .env templates). README + docs sometimes
# embed tokens in example output. Same regex set, same allowlist
# mechanism, broader file coverage.
NONSOURCE_EXTENSIONS = {".json", ".yaml", ".yml", ".sh", ".env", ".md", ".toml", ".cfg"}


def _iter_nonsource_files(root: Path):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in p.parts):
            continue
        if p.suffix.lower() not in NONSOURCE_EXTENSIONS:
            continue
        # Skip pyproject.toml's [project.dependencies] entries that
        # might look like hf_<long-package-name> — those are package
        # specs, not tokens.
        if p.name == "pyproject.toml":
            continue
        # Skip .env.example which BY CONVENTION has placeholder values
        # that look like real secrets — that's the file's purpose.
        if p.name == ".env.example":
            continue
        yield p


def test_no_hardcoded_secrets_in_nonsource_files():
    """M183 (iter 118): extends M182 to JSON / YAML / shell / docs.
    Same patterns + allowlist mechanism. Common hardcode-leak vectors:
    CI env files, deploy scripts, README example output, MCP/plugin
    config blobs."""
    offenders: list[str] = []
    for path in _iter_nonsource_files(REPO_ROOT):
        rel = str(path.relative_to(REPO_ROOT))
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, regex, desc in SECRET_PATTERNS:
            if (rel, name) in ALLOWED_FIXTURES:
                continue
            for m in regex.finditer(content):
                line = content[: m.start()].count("\n") + 1
                masked = m.group(0)[:6] + "<...>" + m.group(0)[-2:]
                offenders.append(f"  {rel}:{line} — {name} [{masked}] — {desc}")
    assert not offenders, (
        f"M183 regression: hardcoded secrets in non-source files:\n" +
        "\n".join(offenders) +
        "\n\nROTATE the secret at its source-of-truth FIRST, then remove "
        "from the file. If a clearly-fake doc/example, add to "
        "ALLOWED_FIXTURES with rationale."
    )


def test_no_hardcoded_secrets_repo_wide():
    """Cross-cutting M181 invariant: no real secret literals in any
    .py / .swift file across the repo (excluding allowlisted test
    fixtures and vendored third-party code)."""
    offenders: list[str] = []
    for path in _iter_source_files(REPO_ROOT):
        rel = str(path.relative_to(REPO_ROOT))
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue   # binary/utf-16 file; skip
        for name, regex, desc in SECRET_PATTERNS:
            if (rel, name) in ALLOWED_FIXTURES:
                continue
            for m in regex.finditer(content):
                line = content[: m.start()].count("\n") + 1
                # Mask the match so the error message doesn't itself
                # leak whatever secret triggered the test.
                masked = m.group(0)[:6] + "<...>" + m.group(0)[-2:]
                offenders.append(f"  {rel}:{line} — {name} [{masked}] — {desc}")
    assert not offenders, (
        f"M181/M182 regression: hardcoded secrets detected in source:\n" +
        "\n".join(offenders) +
        "\n\nFor each: ROTATE the secret at its source-of-truth FIRST "
        "(removing from source does not retroactively un-leak it), then "
        "replace with env-var read using empty/None default + fail-fast "
        "at use-site. If a clearly-fake test fixture, add to "
        "ALLOWED_FIXTURES with a rationale comment."
    )
