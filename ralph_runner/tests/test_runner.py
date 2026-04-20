"""Tests for ralph_runner.runner state-machine logic.

These tests avoid hitting macstudio — they exercise pure state operations
(load_state / save_state / recover_interrupted) and the repo-id validator.
The subprocess-driven commands (cmd_next) are not tested here since they
require SSH; those are covered by the live audit harness on macstudio.
"""
from __future__ import annotations
import json
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
