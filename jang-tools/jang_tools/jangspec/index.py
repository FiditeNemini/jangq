"""
Flat binary expert index (experts.jsidx).

The index is loaded once per runtime and serves (layer_idx, expert_id) lookups
during speculative decoding. The on-disk layout is deliberately trivial so
the Swift runtime can mmap the file and cast directly to a struct array.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from . import format as fmt


@dataclass(frozen=True)
class ExpertIndexEntry:
    layer_idx: int
    expert_id: int
    file_id: int
    offset: int
    nbytes: int


@dataclass
class LoadedIndex:
    version: int
    n_layers: int
    n_experts_per_layer: int
    entries: List[ExpertIndexEntry]

    def lookup(self, layer_idx: int, expert_id: int) -> Optional[ExpertIndexEntry]:
        for e in self.entries:
            if e.layer_idx == layer_idx and e.expert_id == expert_id:
                return e
        return None


def write_index(
    path: Path,
    *,
    entries: Iterable[ExpertIndexEntry],
    n_layers: int,
    n_experts_per_layer: int,
) -> None:
    entries = list(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(
            struct.pack(
                fmt.INDEX_HEADER_FORMAT,
                fmt.INDEX_MAGIC,
                1,
                0,
                n_layers,
                n_experts_per_layer,
                len(entries),
            )
        )
        for e in entries:
            f.write(
                struct.pack(
                    fmt.INDEX_ENTRY_FORMAT,
                    e.layer_idx,
                    e.expert_id,
                    e.file_id,
                    0,
                    e.offset,
                    e.nbytes,
                )
            )


def read_index(path: Path) -> LoadedIndex:
    raw = Path(path).read_bytes()
    if len(raw) < fmt.INDEX_HEADER_SIZE:
        raise ValueError("index file too short for header")
    magic, version, _pad, n_layers, n_experts_per_layer, n_entries = struct.unpack(
        fmt.INDEX_HEADER_FORMAT, raw[: fmt.INDEX_HEADER_SIZE]
    )
    if magic != fmt.INDEX_MAGIC:
        raise ValueError(f"bad index magic 0x{magic:08x}, expected 0x{fmt.INDEX_MAGIC:08x}")
    if version != 1:
        raise ValueError(f"unsupported index version {version}")

    expected_size = fmt.INDEX_HEADER_SIZE + n_entries * fmt.INDEX_ENTRY_SIZE
    if len(raw) < expected_size:
        raise ValueError(f"index file truncated: expected {expected_size} bytes, got {len(raw)}")

    entries: List[ExpertIndexEntry] = []
    cursor = fmt.INDEX_HEADER_SIZE
    for _ in range(n_entries):
        layer_idx, expert_id, file_id, _pad2, offset, nbytes = struct.unpack(
            fmt.INDEX_ENTRY_FORMAT, raw[cursor : cursor + fmt.INDEX_ENTRY_SIZE]
        )
        cursor += fmt.INDEX_ENTRY_SIZE
        entries.append(
            ExpertIndexEntry(
                layer_idx=layer_idx,
                expert_id=expert_id,
                file_id=file_id,
                offset=offset,
                nbytes=nbytes,
            )
        )

    return LoadedIndex(
        version=version,
        n_layers=n_layers,
        n_experts_per_layer=n_experts_per_layer,
        entries=entries,
    )
