"""M181 (iter 116): regression invariant against hardcoded secrets.

Pre-M181, server.py shipped with a real HF write-token as the default
value of `HF_UPLOAD_TOKEN`. That token was committed to source and
needs separate rotation at https://huggingface.co/settings/tokens —
anyone with repo or git-history access already has it.

This test catches future regressions: any `hf_<20+ chars>` literal in
source (outside the explicit allowlist for test fixtures) fails the
test. Matches iter-104 M108 / iter-105 M113's invariant philosophy
("turn observation into testable rule") applied to secrets.

If a legit test fixture needs an `hf_*`-shaped placeholder, add it to
the `TEST_FIXTURE_PATHS` allowlist with a brief rationale.
"""
from __future__ import annotations

import re
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"

# HF token shape: `hf_` + 20+ chars from [A-Za-z0-9_-]. Same regex as
# DiagnosticsBundle.scrubSensitive on the Swift side.
HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9_-]{20,}\b")


def test_no_hardcoded_hf_token_in_server_py():
    """M181: server.py must NOT contain any literal `hf_*` token. Use
    `os.environ.get("HF_UPLOAD_TOKEN", "")` (empty string fallback) so
    missing-token failures surface as actionable errors, never silently
    fall back to a default."""
    content = SERVER_PY.read_text(encoding="utf-8")
    matches = HF_TOKEN_RE.findall(content)
    assert not matches, (
        f"M181 regression: server.py contains hardcoded HF token literal(s): "
        f"{matches}. Remove the default value from os.environ.get and "
        f"ROTATE the leaked token at https://huggingface.co/settings/tokens. "
        f"Anyone with repo or git-history access already has the leaked token."
    )


def test_HF_UPLOAD_TOKEN_default_is_empty():
    """Pin the empty-default contract so a future `HF_UPLOAD_TOKEN = ...`
    assignment with a real value gets caught. Complements the regex
    test above with a more semantic check."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Match the env-var read line; capture the second arg (the default).
    pattern = re.compile(
        r'HF_UPLOAD_TOKEN\s*=\s*os\.environ\.get\(\s*"HF_UPLOAD_TOKEN"\s*,\s*([^\)]+)\)',
        re.MULTILINE,
    )
    matches = pattern.findall(content)
    assert matches, "HF_UPLOAD_TOKEN env-var read line not found in expected shape"
    for default_arg in matches:
        # Acceptable defaults: "" or '' or None. Anything else is suspicious.
        cleaned = default_arg.strip()
        assert cleaned in ('""', "''", "None"), (
            f"HF_UPLOAD_TOKEN default must be empty string or None, got: {cleaned}. "
            f"A non-empty default risks shipping a real token to source."
        )
