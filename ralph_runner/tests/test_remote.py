"""Unit tests for remote arg builders (no actual SSH)."""
from ralph_runner.remote import build_rsync_args, build_ssh_args, DEFAULT_EXCLUDES


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
