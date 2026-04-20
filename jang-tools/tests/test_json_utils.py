"""M152 (iter 75): shared `_json_utils` module tests.

Crystallized across 5 call sites (capabilities, examples, format/reader,
jangspec/manifest, loader) into two helpers with distinct contracts:
  - read_json_object → raise on any failure.
  - read_json_object_safe → (data, err) tuple, never raises.

These tests pin the contract at the helper level so future migrations
to additional sites can rely on the guarantee.
"""
import json
from pathlib import Path

import pytest

from jang_tools._json_utils import read_json_object, read_json_object_safe


# ──────────── read_json_object (raise) ────────────


def test_read_json_object_happy_path(tmp_path: Path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"a": 1, "nested": {"b": 2}}))
    data = read_json_object(p, purpose="test config")
    assert data == {"a": 1, "nested": {"b": 2}}


def test_read_json_object_missing_file(tmp_path: Path):
    p = tmp_path / "missing.json"
    with pytest.raises(ValueError) as excinfo:
        read_json_object(p, purpose="missing thing")
    msg = str(excinfo.value)
    assert "missing thing" in msg
    assert str(p) in msg


def test_read_json_object_malformed_json(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{ not valid")
    with pytest.raises(ValueError) as excinfo:
        read_json_object(p, purpose="broken")
    msg = str(excinfo.value)
    assert "not valid JSON" in msg
    assert str(p) in msg
    assert "line" in msg and "col" in msg


def test_read_json_object_non_dict_root(tmp_path: Path):
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(ValueError) as excinfo:
        read_json_object(p, purpose="test")
    msg = str(excinfo.value)
    assert "top-level list" in msg
    assert "expected a JSON object" in msg


def test_read_json_object_accepts_string_path(tmp_path: Path):
    """Callers may pass a str path; helper must convert via Path(path)."""
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"ok": True}))
    data = read_json_object(str(p), purpose="test")
    assert data == {"ok": True}


def test_read_json_object_preserves_exception_chain(tmp_path: Path):
    """`from exc` chaining must preserve the original JSONDecodeError so
    debugging tools (traceback __cause__, logging) still see the root."""
    p = tmp_path / "broken.json"
    p.write_text("{")
    try:
        read_json_object(p, purpose="test")
    except ValueError as ve:
        assert ve.__cause__ is not None
        assert isinstance(ve.__cause__, json.JSONDecodeError)
    else:
        pytest.fail("expected ValueError")


# ──────────── read_json_object_safe (tuple) ────────────


def test_read_json_object_safe_happy_path(tmp_path: Path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"a": 1}))
    data, err = read_json_object_safe(p, purpose="test")
    assert data == {"a": 1}
    assert err is None


def test_read_json_object_safe_malformed_returns_tuple(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("garbage")
    data, err = read_json_object_safe(p, purpose="test")
    assert data is None
    assert err is not None
    assert "not valid JSON" in err
    assert str(p) in err


def test_read_json_object_safe_missing_file_returns_tuple(tmp_path: Path):
    p = tmp_path / "missing.json"
    data, err = read_json_object_safe(p, purpose="test")
    assert data is None
    assert err is not None
    assert str(p) in err


def test_read_json_object_safe_non_dict_returns_tuple(tmp_path: Path):
    p = tmp_path / "list.json"
    p.write_text("[]")
    data, err = read_json_object_safe(p, purpose="test")
    assert data is None
    assert err is not None
    assert "top-level list" in err


def test_read_json_object_safe_never_raises(tmp_path: Path):
    """Contract: read_json_object_safe must NEVER raise — otherwise
    tuple-return callers (verify_directory, detection probes) will crash."""
    # Give it something truly bizarre to make sure no edge case leaks a raise.
    # Path to a directory (not a file) — read_text will OSError / IsADirectoryError.
    data, err = read_json_object_safe(tmp_path, purpose="dir-not-file")
    assert data is None
    assert err is not None
