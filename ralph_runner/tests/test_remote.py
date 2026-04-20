"""Unit tests for remote arg builders (no actual SSH)."""
import subprocess
from unittest.mock import patch

from ralph_runner.remote import (
    build_rsync_args,
    build_ssh_args,
    DEFAULT_EXCLUDES,
    sync_tree,
    pull_tree,
)


def test_rsync_includes_archive_delete():
    args = build_rsync_args("src/", "host:dst/")
    assert args[0] == "rsync"
    assert "-az" in args
    assert "--delete" in args


def test_rsync_default_excludes():
    args = build_rsync_args("src/", "host:dst/")
    # At least --exclude .git should be present
    assert any(e == ".git" for e in args)


def test_rsync_custom_excludes():
    args = build_rsync_args("src/", "host:dst/", excludes=["foo"])
    assert any(e == "foo" for e in args)
    # Default excludes NOT applied when caller passes explicit list
    assert not any(e == ".git" for e in args)


def test_ssh_args_has_timeout_and_batchmode():
    args = build_ssh_args("macstudio", "echo hi", timeout=5)
    assert args[0] == "ssh"
    assert "ConnectTimeout=5" in args
    assert "BatchMode=yes" in args
    assert "macstudio" in args
    assert "echo hi" in args


# ────────────────────────────────────────────────────────────────────
# Iter 41: M118 — sync_tree / pull_tree gained timeout parameters
# ────────────────────────────────────────────────────────────────────

def test_sync_tree_passes_timeout_to_subprocess():
    """M118: sync_tree must send timeout= to subprocess.run; pre-iter-41
    a hung rsync stalled Ralph forever."""
    captured = {}
    def fake_run(args, *, capture_output, text, timeout):
        captured["args"] = args
        captured["timeout"] = timeout
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()
    with patch("ralph_runner.remote.subprocess.run", side_effect=fake_run):
        sync_tree("/tmp/src/", "remote/path", timeout=42)
    assert captured["timeout"] == 42
    assert captured["args"][0] == "rsync"


def test_sync_tree_default_timeout_generous():
    """Default must be generous enough for real jang-tools tree transfer
    (~100 MB) but not infinite. 30 minutes matches iter-41's documented
    rationale. Regression pin against a future over-tightening."""
    captured = {}
    def fake_run(args, *, capture_output, text, timeout):
        captured["timeout"] = timeout
        class R:
            returncode = 0; stdout = ""; stderr = ""
        return R()
    with patch("ralph_runner.remote.subprocess.run", side_effect=fake_run):
        sync_tree("/tmp/src/", "remote/path")
    # Pin a range: ≥5 min (real transfers need headroom) and ≤60 min (don't
    # hang the whole day on a truly-broken network).
    assert 300 <= captured["timeout"] <= 3600


def test_sync_tree_timeout_returns_124_not_raise():
    """On TimeoutExpired, sync_tree must return a structured RemoteResult
    with returncode=124 (conventional timeout exit) + informative stderr.
    Must NOT raise to the caller — cmd_next would treat unhandled
    TimeoutExpired as a crash instead of a retryable failure."""
    def fake_run_timeout(args, *, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)
    with patch("ralph_runner.remote.subprocess.run", side_effect=fake_run_timeout):
        r = sync_tree("/tmp/src/", "remote/path", timeout=1)
    assert r.returncode == 124
    assert "timeout" in r.stderr.lower()
    assert "exceeded" in r.stderr.lower()


def test_pull_tree_also_gets_timeout():
    """pull_tree has the same bug class as sync_tree. Pin the fix on
    both sides so a future simplification doesn't accidentally revert one."""
    captured = {}
    def fake_run(args, *, capture_output, text, timeout):
        captured["timeout"] = timeout
        class R:
            returncode = 0; stdout = ""; stderr = ""
        return R()
    with patch("ralph_runner.remote.subprocess.run", side_effect=fake_run):
        pull_tree("remote/path", "/tmp/dst", timeout=99)
    assert captured["timeout"] == 99


def test_pull_tree_timeout_returns_124():
    def fake_run_timeout(args, *, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd=args, timeout=timeout)
    with patch("ralph_runner.remote.subprocess.run", side_effect=fake_run_timeout):
        r = pull_tree("remote/path", "/tmp/dst", timeout=1)
    assert r.returncode == 124
    assert "timeout" in r.stderr.lower()
