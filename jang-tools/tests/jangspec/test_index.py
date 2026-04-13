"""Unit tests for jang_tools.jangspec.index."""

from pathlib import Path

import pytest

from jang_tools.jangspec import format as fmt
from jang_tools.jangspec.index import ExpertIndexEntry, write_index, read_index


def test_index_roundtrip(tmp_path: Path):
    entries = [
        ExpertIndexEntry(layer_idx=0, expert_id=0, file_id=0, offset=0, nbytes=4096),
        ExpertIndexEntry(layer_idx=0, expert_id=1, file_id=0, offset=4096, nbytes=8192),
        ExpertIndexEntry(layer_idx=1, expert_id=0, file_id=1, offset=0, nbytes=4096),
    ]
    path = tmp_path / "experts.jsidx"
    write_index(path, entries=entries, n_layers=2, n_experts_per_layer=2)

    loaded = read_index(path)
    assert loaded.n_layers == 2
    assert loaded.n_experts_per_layer == 2
    assert len(loaded.entries) == 3

    # Linear lookup
    first = loaded.lookup(layer_idx=0, expert_id=1)
    assert first is not None
    assert first.file_id == 0
    assert first.offset == 4096
    assert first.nbytes == 8192

    # Unknown tuple returns None
    assert loaded.lookup(layer_idx=5, expert_id=5) is None


def test_index_header_magic(tmp_path: Path):
    path = tmp_path / "experts.jsidx"
    write_index(path, entries=[], n_layers=0, n_experts_per_layer=0)
    import struct
    with open(path, "rb") as f:
        raw = f.read(fmt.INDEX_HEADER_SIZE)
    (magic,) = struct.unpack("<I", raw[:4])
    assert magic == fmt.INDEX_MAGIC


def test_index_rejects_wrong_magic(tmp_path: Path):
    path = tmp_path / "bad.jsidx"
    path.write_bytes(b"\x00" * fmt.INDEX_HEADER_SIZE)
    with pytest.raises(ValueError, match="magic"):
        read_index(path)
