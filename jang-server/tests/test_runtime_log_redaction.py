"""M193 (iter 130): runtime-log secret redaction tests.

Pre-M193 several log sites fed raw strings into Python logger AND the
job's log_lines buffer (exposed via GET /jobs/{id}/logs):
  1. _ProgressWriter.write() forwarded subprocess stdout/stderr
     verbatim — if convert_model ever printed an HF URL, exception with
     token in query string, or the token itself, it flowed into both.
  2. Webhook "delivered" log lines included the raw URL — Slack/Discord
     webhook URLs have the write secret in the path.
  3. Exception paths stored `traceback.format_exc()` verbatim — HF
     client exceptions include the failing URL with auth params.

M193 adds `redact_for_log(s)` + applies it at every high-risk log site.

These tests cover:
  - Unit tests for each redaction pattern (HF token, OpenAI key,
    Bearer header, slack webhook, query-string secrets).
  - Pins via source inspection that specific high-risk log sites use
    redact_for_log so a future edit doesn't silently drop the wrapper.
  - Idempotency check (redacted string redacts to itself).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


SERVER_PY = Path(__file__).parent.parent / "server.py"


def _load_server():
    """Import server.py as a module so we can unit-test redact_for_log
    without standing up the FastAPI app."""
    spec = importlib.util.spec_from_file_location("jang_server_under_test", SERVER_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_redact_for_log_hf_token():
    server = _load_server()
    out = server.redact_for_log("token is hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890 stop")
    assert "hf_ABCD" not in out
    assert "***REDACTED***" in out


def test_redact_for_log_huggingface_legacy_token():
    """M196 (iter 133): cross-language parity with Swift's
    DiagnosticsBundle.sensitivePatterns. Pre-M196 `redact_for_log`
    only matched the `hf_*` shape; the legacy `huggingface_*` format
    (still emitted by some HF client error paths) would slip through."""
    server = _load_server()
    out = server.redact_for_log("old format: huggingface_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890")
    assert "huggingface_ABCD" not in out
    assert "***REDACTED***" in out


def test_redact_for_log_openai_key():
    server = _load_server()
    out = server.redact_for_log("openai: sk-proj-abcdefghijklmnopqrst12345 end")
    assert "sk-proj-abc" not in out
    assert "***REDACTED***" in out


def test_redact_for_log_bearer_header():
    server = _load_server()
    out = server.redact_for_log("Authorization: Bearer abc.def.ghi1234567890jkl")
    assert "abc.def" not in out
    assert "Bearer ***REDACTED***" in out


def test_redact_for_log_slack_webhook():
    server = _load_server()
    url = "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXX"
    out = server.redact_for_log(f"Webhook delivered to {url}")
    assert "T00000000" not in out
    assert "XXXXXXXXXXXXXXXX" not in out
    assert "hooks.slack.com" in out  # host stays for diagnostics
    assert "***REDACTED***" in out


def test_redact_for_log_discord_webhook():
    server = _load_server()
    url = "https://discord.com/api/webhooks/1234567890/secret-token-goes-here-abc"
    out = server.redact_for_log(url)
    assert "secret-token" not in out
    assert "discord.com" in out
    assert "***REDACTED***" in out


def test_redact_for_log_query_string_secrets():
    server = _load_server()
    out = server.redact_for_log("fetch https://api.example.com/path?api_key=SECRETVALUE123 done")
    assert "SECRETVALUE123" not in out
    assert "api_key=***REDACTED***" in out


def test_redact_for_log_idempotent():
    """Re-applying must not double-redact (the sentinel doesn't match any pattern)."""
    server = _load_server()
    raw = "token hf_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    once = server.redact_for_log(raw)
    twice = server.redact_for_log(once)
    assert once == twice


def test_redact_for_log_preserves_clean_messages():
    """Redaction must not mangle legit log messages. Model IDs,
    file paths, and stack frames without secrets should survive."""
    server = _load_server()
    clean = "Queue: processing job_abc123 (meta-llama/Llama-3-8B → JANG_4K)"
    assert server.redact_for_log(clean) == clean


def test_progress_writer_uses_redact_for_log():
    """Pin: _ProgressWriter.write must pass lines through redact_for_log
    before calling self.job.log. Subprocess stdout can carry tokens."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Slice the _ProgressWriter.write function.
    start = content.find("class _LogCapture")
    assert start != -1
    # Find the write method within _LogCapture.
    write_idx = content.find("def write(self, s: str) -> int:", start)
    assert write_idx != -1
    # Bound the write method by the next top-level or indented-at-4 def.
    rest = content[write_idx:]
    end = rest.find("\n    def ", 1)
    if end == -1:
        end = rest.find("\nclass ", 1)
    body = rest[: end if end != -1 else 1500]
    assert "redact_for_log" in body, (
        "M193 regression: _LogCapture.write() does NOT pass subprocess "
        "lines through redact_for_log before job.log(). Raw subprocess "
        "output (which can contain tokens from exception messages or "
        "env-dump printouts) would flow into the job log and module log."
    )


def test_webhook_delivery_log_uses_redact_for_log():
    """Pin: the 'Webhook delivered to {url}' log line must redact."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Find the ACTUAL log statement — use rfind to get the LAST
    # occurrence (skips the M193 rationale block at the top of file).
    idx = content.rfind("Webhook delivered to")
    assert idx != -1, "M193 regression: 'Webhook delivered to' log line removed"
    # Grab a small slice around it — must contain redact_for_log.
    snippet = content[idx - 80 : idx + 200]
    assert "redact_for_log" in snippet, (
        "M193 regression: 'Webhook delivered to' log does NOT wrap the "
        "URL with redact_for_log. Slack/Discord/custom webhook URLs "
        "carry the write secret in the path — plain-log leaks them."
    )


def test_exception_storage_uses_redact_for_log():
    """Pin: job.error = traceback.format_exc() must be redacted.
    HF client exceptions embed the failing URL with query params."""
    content = SERVER_PY.read_text(encoding="utf-8")
    # Use rfind so the rationale-block mention of traceback.format_exc()
    # in the M193 header comment doesn't mask the actual assignment.
    idx = content.rfind("traceback.format_exc()")
    assert idx != -1
    # Look for redact_for_log near the traceback dump. 300-char window.
    snippet = content[max(0, idx - 200) : idx + 100]
    assert "redact_for_log" in snippet, (
        "M193 regression: traceback.format_exc() is stored in job.error "
        "without redaction. Tracebacks can include URLs with tokens "
        "(huggingface_hub raises with the failing URL in its message)."
    )
