"""Regression tests for M115 (iter 39) — stale JANG artifact cleanup.

Memory ref: feedback_model_checklist.md rule 1. Re-converting into an
existing output dir used to leave v1 .jang.safetensors shards alongside
v2 files, plus any v2 shards from a different shard count lingered as
never-overwritten junk. Historical bloat source ("155 GB junk shipped").
"""
from pathlib import Path

from jang_tools.convert import (
    STALE_JANG_ARTIFACT_PATTERNS,
    _remove_stale_jang_artifacts,
)


def test_patterns_include_v1_and_v2_shards_and_index():
    """Pin the coverage list so a future simplification doesn't accidentally
    drop a pattern and re-introduce the bloat bug."""
    expected = {
        "*.jang.safetensors",
        "model.jang.index.json",
        "model-*-of-*.safetensors",
        "model.safetensors.index.json",
        "jang_imatrix.safetensors",
        "jang_config.json",
    }
    assert set(STALE_JANG_ARTIFACT_PATTERNS) == expected, (
        f"STALE pattern list changed — removed patterns would leak bloat. "
        f"Missing: {expected - set(STALE_JANG_ARTIFACT_PATTERNS)}, "
        f"added: {set(STALE_JANG_ARTIFACT_PATTERNS) - expected}"
    )


def test_cleanup_removes_v1_shards(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # Plant a v1-format layout
    (out / "model-00001-of-00004.jang.safetensors").write_bytes(b"stale v1 shard")
    (out / "model-00002-of-00004.jang.safetensors").write_bytes(b"stale v1 shard")
    (out / "model.jang.index.json").write_text('{"stale": true}')
    removed = _remove_stale_jang_artifacts(out)
    assert "model-00001-of-00004.jang.safetensors" in removed
    assert "model-00002-of-00004.jang.safetensors" in removed
    assert "model.jang.index.json" in removed
    assert not (out / "model-00001-of-00004.jang.safetensors").exists()


def test_cleanup_removes_v2_shards_from_different_count(tmp_path):
    """If a previous convert wrote 4 shards and the new convert will write
    3, the 4th shard should NOT persist as orphan junk."""
    out = tmp_path / "out"
    out.mkdir()
    for i in range(1, 5):
        (out / f"model-{i:05d}-of-00004.safetensors").write_bytes(b"old shard")
    (out / "model.safetensors.index.json").write_text('{"old": true}')
    removed = _remove_stale_jang_artifacts(out)
    assert len(removed) == 5   # 4 shards + 1 index
    for i in range(1, 5):
        assert not (out / f"model-{i:05d}-of-00004.safetensors").exists()


def test_cleanup_removes_imatrix_and_jang_config(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "jang_imatrix.safetensors").write_bytes(b"stale imatrix")
    (out / "jang_config.json").write_text('{"stale": "config"}')
    removed = _remove_stale_jang_artifacts(out)
    assert "jang_imatrix.safetensors" in removed
    assert "jang_config.json" in removed


def test_cleanup_preserves_user_files(tmp_path):
    """Rule 1 is 'remove JUNK', not 'nuke the whole dir'. User-added files
    (README.md, .gitattributes, custom notes, preserved-from-source files)
    must stay."""
    out = tmp_path / "out"
    out.mkdir()
    # Files that should stay:
    user_files = [
        "README.md",
        ".gitattributes",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "chat_template.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "generation_config.json",
        "modeling_minimax.py",
        "configuration_minimax.py",
        "my_custom_notes.txt",
    ]
    for name in user_files:
        (out / name).write_text("preserved content")
    # Plus one stale file to prove cleanup runs
    (out / "jang_config.json").write_text("stale")

    removed = _remove_stale_jang_artifacts(out)
    assert removed == ["jang_config.json"]

    for name in user_files:
        assert (out / name).exists(), f"{name} was removed — user file protection broken"


def test_cleanup_idempotent_on_clean_dir(tmp_path):
    """Running cleanup on a fresh dir returns empty list, no exceptions."""
    out = tmp_path / "out"
    out.mkdir()
    assert _remove_stale_jang_artifacts(out) == []
    # Double-call — still safe
    assert _remove_stale_jang_artifacts(out) == []


def test_cleanup_tolerates_missing_dir(tmp_path):
    """If the output dir doesn't exist, cleanup doesn't raise — Path.glob
    on a missing dir returns empty, not FileNotFoundError."""
    out = tmp_path / "does-not-exist"
    # No mkdir
    assert _remove_stale_jang_artifacts(out) == []


def test_cleanup_does_not_touch_sibling_dirs(tmp_path):
    """Cleanup is NON-RECURSIVE — it targets only files directly in
    output_path, not nested subdirectories. This protects against
    accidentally nuking user-placed subdirs like a bundled `assets/` dir."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "jang_config.json").write_text("stale")   # top-level, should go
    # Nested that happens to match a pattern — should STAY
    nested = out / "assets"
    nested.mkdir()
    (nested / "jang_config.json").write_text("user content in a subdir")
    (nested / "model-00001-of-00001.safetensors").write_bytes(b"nested content")

    removed = _remove_stale_jang_artifacts(out)
    assert removed == ["jang_config.json"]
    assert (nested / "jang_config.json").exists(), \
        "cleanup recursed into subdir — user content in nested dirs unsafe"
    assert (nested / "model-00001-of-00001.safetensors").exists()
