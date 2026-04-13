"""
JangSpecReader — load a .jangspec bundle for testing, debugging, and
Swift-parity sanity checks. This is NOT the production runtime; it just
exists so we can round-trip bundles in pure Python.
"""

from __future__ import annotations

import mmap
from pathlib import Path
from typing import Dict

from . import format as fmt
from .blob import UnpackedBlob, unpack_expert_blob
from .index import LoadedIndex, read_index
from .manifest import Manifest, load_manifest


class JangSpecReader:
    def __init__(self, bundle_dir: Path):
        self.bundle_dir = Path(bundle_dir)
        self.manifest: Manifest = load_manifest(self.bundle_dir / fmt.MANIFEST_FILENAME)
        self.index: LoadedIndex = read_index(self.bundle_dir / fmt.INDEX_FILENAME)
        self._mm: Dict[int, mmap.mmap] = {}
        self._entry_map = {
            (e.layer_idx, e.expert_id): e for e in self.index.entries
        }

    @property
    def n_layers(self) -> int:
        return self.index.n_layers

    @property
    def n_experts_per_layer(self) -> int:
        return self.index.n_experts_per_layer

    def _mmap(self, file_id: int) -> mmap.mmap:
        if file_id not in self._mm:
            path = self.bundle_dir / fmt.EXPERT_FILE_PATTERN.format(idx=file_id)
            f = open(path, "rb")
            self._mm[file_id] = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        return self._mm[file_id]

    def load_expert(self, layer_idx: int, expert_id: int) -> UnpackedBlob:
        entry = self._entry_map.get((layer_idx, expert_id))
        if entry is None:
            raise KeyError(f"no expert entry for (layer={layer_idx}, id={expert_id})")
        mm = self._mmap(entry.file_id)
        blob = bytes(mm[entry.offset : entry.offset + entry.nbytes])
        return unpack_expert_blob(blob)

    def close(self) -> None:
        for mm in self._mm.values():
            mm.close()
        self._mm.clear()
