"""Tests for ralph_runner.runner state-machine logic.

These tests avoid hitting macstudio — they exercise pure state operations
(load_state / save_state / recover_interrupted) and the repo-id validator.
The subprocess-driven commands (cmd_next) are not tested here since they
require SSH; those are covered by the live audit harness on macstudio.
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import pytest

from ralph_runner import runner


@pytest.fixture
def state_path(tmp_path, monkeypatch):
    """Redirect STATE_PATH to a tmp file for isolation."""
    p = tmp_path / "state.json"
    monkeypatch.setattr(runner, "STATE_PATH", p)
    return p


def test_load_state_returns_empty_when_missing(state_path):
    state = runner.load_state()
    assert state == {"combos": {}, "active_tier": None, "created": state["created"]}


def test_save_then_load_roundtrips(state_path):
    original = {"combos": {"a__b": {"status": "pending"}}, "active_tier": 1, "created": "2026-04-19"}
    runner.save_state(original)
    reloaded = runner.load_state()
    assert reloaded == original


# ────────────────────────────────────────────────────────────────────
# M54: recover_interrupted flips `running` → `pending` on startup
# ────────────────────────────────────────────────────────────────────

def test_recover_interrupted_no_running_combos():
    state = {"combos": {"a__b": {"status": "pending"}, "c__d": {"status": "green"}}}
    count = runner.recover_interrupted(state)
    assert count == 0
    assert state["combos"]["a__b"]["status"] == "pending"
    assert state["combos"]["c__d"]["status"] == "green"


def test_recover_interrupted_flips_running_to_pending():
    state = {
        "combos": {
            "a__b": {"status": "running", "started": "2026-04-19T10:00:00"},
            "c__d": {"status": "pending"},
            "e__f": {"status": "running", "started": "2026-04-19T11:00:00"},
        }
    }
    count = runner.recover_interrupted(state)
    assert count == 2
    assert state["combos"]["a__b"]["status"] == "pending"
    assert state["combos"]["e__f"]["status"] == "pending"
    assert state["combos"]["c__d"]["status"] == "pending"  # untouched


def test_recover_interrupted_preserves_started_as_recovered():
    """The `started` timestamp should move to a `recovered_from_interrupt`
    field so post-mortem debugging can tell which runs died mid-flight."""
    state = {"combos": {"a__b": {"status": "running", "started": "2026-04-19T10:00:00"}}}
    runner.recover_interrupted(state)
    assert state["combos"]["a__b"]["recovered_from_interrupt"] == "2026-04-19T10:00:00"
    # `started` key should be cleaned up — it's misleading now that the combo
    # is pending again.
    assert "started" not in state["combos"]["a__b"]


def test_recover_interrupted_handles_empty_state():
    # No combos at all → should not crash, returns 0.
    assert runner.recover_interrupted({"combos": {}}) == 0
    assert runner.recover_interrupted({}) == 0  # missing key is OK


# ────────────────────────────────────────────────────────────────────
# M52: _assert_safe_repo_id rejects shell-dangerous repo ids
# ────────────────────────────────────────────────────────────────────

def test_assert_safe_repo_id_accepts_canonical():
    # Should not raise for any of these.
    runner._assert_safe_repo_id("dealignai/MyModel-JANG_4K")
    runner._assert_safe_repo_id("org/name")
    runner._assert_safe_repo_id("a/b")
    runner._assert_safe_repo_id("Org_Name/model.v2")


def test_assert_safe_repo_id_rejects_shell_injection():
    """These would lead to RCE on macstudio if spliced into the Python
    one-liner in ensure_source_model. Harness against regressions."""
    dangerous = [
        'foo/bar"); __import__("os").system("pwn"); print("',
        "foo/bar\"; echo pwned; \"",
        "foo/bar\nrm -rf /",
        "foo/bar`id`",
        "foo/bar$(id)",
        "foo/bar;echo pwn",
        "foo/bar|cat /etc/passwd",
        "foo/bar&echo bg",
    ]
    for repo in dangerous:
        with pytest.raises(ValueError, match="Unsafe HF repo id"):
            runner._assert_safe_repo_id(repo)


def test_assert_safe_repo_id_rejects_structurally_bad():
    for repo in ["", "  ", "justname", "org//name", "org/name/extra", "/name", "org/"]:
        with pytest.raises(ValueError):
            runner._assert_safe_repo_id(repo)


# ────────────────────────────────────────────────────────────────────
# slug() stability (unchanged behaviour — pins downstream file paths)
# ────────────────────────────────────────────────────────────────────

def test_slug_is_stable_and_filesystem_safe():
    # Spaces and slashes get replaced; dots survive (valid HF id char).
    assert runner.slug("org/name", "JANG_4K") == "org__name__JANG_4K"
    assert runner.slug("org name", "P") == "org_name__P"
    # Exact same inputs → exact same output.
    assert runner.slug("a/b", "c") == runner.slug("a/b", "c")


# ────────────────────────────────────────────────────────────────────
# M55: multi-instance lock (iter 12)
# ────────────────────────────────────────────────────────────────────

def test_acquire_lock_happy_path(tmp_path):
    lock = tmp_path / "ralph.lock"
    runner.acquire_lock(lock)
    assert lock.exists()
    info = json.loads(lock.read_text())
    assert info["pid"] == os.getpid()
    assert "host" in info
    assert "started_at" in info


def test_release_lock_removes_file(tmp_path):
    lock = tmp_path / "ralph.lock"
    runner.acquire_lock(lock)
    assert lock.exists()
    runner.release_lock(lock)
    assert not lock.exists()


def test_release_lock_missing_is_noop(tmp_path):
    # release on a non-existent lock should NOT raise. Avoids crashing on
    # finally: release_lock() after an early-abort codepath.
    runner.release_lock(tmp_path / "nope.lock")   # no exception = pass


def test_acquire_fails_when_held_by_live_process(tmp_path):
    lock = tmp_path / "ralph.lock"
    # Pretend a live process on this host holds the lock.
    holder = {
        "pid": os.getpid(),   # definitely alive
        "host": os.uname().nodename,
        "started_at": "2026-04-19T10:00:00",
    }
    lock.write_text(json.dumps(holder))

    with pytest.raises(runner.LockAcquireFailed) as exc:
        runner.acquire_lock(lock)
    assert exc.value.holder["pid"] == os.getpid()


def test_acquire_reclaims_stale_lock_with_dead_pid(tmp_path):
    lock = tmp_path / "ralph.lock"
    # Use a PID that's extremely unlikely to be alive. Any unused PID > 1
    # works; we use 2**22 which is above typical PID ranges and we verify
    # it's not alive.
    dead_pid = 2 ** 22
    # If that happens to be alive on this machine, pick another high one.
    while runner._pid_alive(dead_pid):
        dead_pid += 1
    holder = {
        "pid": dead_pid,
        "host": os.uname().nodename,
        "started_at": "2026-04-19T10:00:00",
    }
    lock.write_text(json.dumps(holder))

    # Should succeed — dead PID means stale lock, reclaim it.
    runner.acquire_lock(lock)
    info = json.loads(lock.read_text())
    assert info["pid"] == os.getpid()


def test_acquire_refuses_cross_host_lock(tmp_path):
    """A lock from a different host is not reclaimable — we can't verify
    the remote PID, so refusing defensively is safer than assuming crash."""
    lock = tmp_path / "ralph.lock"
    holder = {
        "pid": 1,
        "host": "some-other-machine",
        "started_at": "2026-04-19T10:00:00",
    }
    lock.write_text(json.dumps(holder))

    with pytest.raises(runner.LockAcquireFailed) as exc:
        runner.acquire_lock(lock)
    assert exc.value.holder["host"] == "some-other-machine"


def test_acquire_reclaims_unparseable_lock(tmp_path):
    """A corrupted lock file (garbage JSON) is treated as stale and reclaimed.
    Otherwise the first hiccup leaves the whole runner wedged forever."""
    lock = tmp_path / "ralph.lock"
    lock.write_text("not json at all {")
    runner.acquire_lock(lock)
    assert json.loads(lock.read_text())["pid"] == os.getpid()


def test_release_only_removes_our_own_lock(tmp_path):
    """If we call release_lock and the file is owned by a different PID,
    we must NOT remove it. Defends against a process that acquired the
    lock after we crashed accidentally getting its lock yanked by our
    stale `finally: release_lock()`.
    """
    lock = tmp_path / "ralph.lock"
    holder = {
        "pid": os.getpid() + 100_000,  # not us
        "host": os.uname().nodename,
        "started_at": "2026-04-19T10:00:00",
    }
    lock.write_text(json.dumps(holder))
    runner.release_lock(lock)
    assert lock.exists(), "release_lock must NOT remove a lock owned by another PID"


def test_pid_alive_self():
    assert runner._pid_alive(os.getpid())


def test_pid_alive_dead():
    # Pick a PID we know isn't ours.
    dead = 2 ** 22
    while runner._pid_alive(dead):
        dead += 1
    assert not runner._pid_alive(dead)
