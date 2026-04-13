# jang-spec Bundle Format + Python Builder + IO Spike — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working `.jangspec` bundle format, a Python builder that converts any JANG MoE model into a bundle, a Python reader that round-trips it, and a Swift `MTLIOCommandQueue` micro-benchmark that validates the streaming premise.

**Architecture:** A new `jang_tools.jangspec` subpackage with small, focused modules (format constants, blob packer, index writer, manifest, tier classifier, builder, reader). CLI exposed as `jang spec build`. A standalone Swift executable `jang-spec-iobench` in `jang-runtime/` measures direct-to-GPU SSD read throughput.

**Tech Stack:** Python 3.11+, `safetensors`, `numpy`. Swift 6.0, Metal 3 (`MTLIOCommandQueue`). Test runner: pytest. Swift benchmark is standalone (no XCTest needed).

**Spec:** `docs/superpowers/specs/2026-04-13-jang-spec-design.md` §5, §9, §14.

**Out of scope for this plan (owned by later plans):**
- Any Swift runtime code beyond the IO benchmark.
- Router prior generation (`router_prior/coact.safetensors`, `transition.safetensors`) — the builder will write placeholder empty files and populate them in Plan 3.
- Drafting or speculative decoding.
- vmlx integration.

**Test fixture:** `/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M` (128-expert MoE). Set env var `JANGSPEC_TEST_MODEL` to override. Tests that need a real model skip if the fixture is absent.

---

## File structure

New files:

```
jang-tools/jang_tools/jangspec/
  __init__.py
  format.py            # on-disk format constants, magic numbers, struct layouts
  blob.py              # pack/unpack ExpertBlob (one expert: gate+up+down)
  index.py             # write/read experts.jsidx (flat binary index)
  manifest.py          # write/read jangspec.json
  tier.py              # classify tensor names into hot-core vs expert
  builder.py           # JangSpecBuilder end-to-end (source JANG dir -> bundle dir)
  reader.py            # JangSpecReader for tests, debugging, and Plan 2 Swift parity checks
  cli.py               # `jang spec build` argparse subcommand

jang-tools/tests/jangspec/
  __init__.py
  conftest.py          # fixture model path, skip markers
  test_blob.py
  test_index.py
  test_manifest.py
  test_tier.py
  test_builder.py      # integration: builds a real bundle from the fixture model
  test_reader.py       # round-trip: read back a built bundle, compare to source bytes

jang-runtime/Sources/jang-spec-iobench/
  main.swift           # standalone benchmark executable

jang-runtime/Package.swift              # ADD executable target jang-spec-iobench
jang-tools/jang_tools/__main__.py       # MODIFY: register `jang spec` subcommand

docs/superpowers/notes/
  2026-04-13-iobench-results.md         # written after running the benchmark
```

Modified files:
- `jang-tools/jang_tools/__main__.py` — add `spec` subparser that dispatches to `jangspec.cli`.
- `jang-runtime/Package.swift` — add `jang-spec-iobench` executable target.

---

## Task 0: Worktree and branch

**Files:** none (git state only)

- [ ] **Step 1: Confirm working directory and branch**

Run:
```bash
cd /Users/eric/jang && git status && git log -1 --oneline
```
Expected: clean working tree on `main` at commit `a1aa178 jang-spec: drop router-aware drafting from v1, prior-only prefetch` (or newer).

- [ ] **Step 2: Create a working branch**

Run:
```bash
git checkout -b jang-spec-plan1-bundle
```
Expected: `Switched to a new branch 'jang-spec-plan1-bundle'`.

- [ ] **Step 3: Create the jangspec source directory**

Run:
```bash
mkdir -p jang-tools/jang_tools/jangspec jang-tools/tests/jangspec jang-runtime/Sources/jang-spec-iobench docs/superpowers/notes
```
Expected: no output.

---

## Task 1: Format constants and struct layouts

**Files:**
- Create: `jang-tools/jang_tools/jangspec/__init__.py`
- Create: `jang-tools/jang_tools/jangspec/format.py`

- [ ] **Step 1: Create the package init**

Write `jang-tools/jang_tools/jangspec/__init__.py`:
```python
"""
jang-spec — SSD-streamed MoE speculative-decoding bundle format.
Created by Jinho Jang (eric@jangq.ai).

See docs/superpowers/specs/2026-04-13-jang-spec-design.md.
"""

from .format import (
    BUNDLE_VERSION,
    MANIFEST_FILENAME,
    INDEX_FILENAME,
    HOT_CORE_FILENAME,
    EXPERT_FILE_PATTERN,
    EXPERT_FILE_BYTES_MAX,
    BLOB_ALIGNMENT,
    BLOB_MAGIC,
    INDEX_MAGIC,
)

__all__ = [
    "BUNDLE_VERSION",
    "MANIFEST_FILENAME",
    "INDEX_FILENAME",
    "HOT_CORE_FILENAME",
    "EXPERT_FILE_PATTERN",
    "EXPERT_FILE_BYTES_MAX",
    "BLOB_ALIGNMENT",
    "BLOB_MAGIC",
    "INDEX_MAGIC",
]
```

- [ ] **Step 2: Create the format module with all on-disk constants**

Write `jang-tools/jang_tools/jangspec/format.py`:
```python
"""
On-disk layout constants and struct definitions for the .jangspec bundle format.

All multi-byte integers are little-endian. All blobs are 4096-byte aligned so
Metal's MTLIOCommandQueue can read them directly into unified-memory buffers.
"""

import struct

# Top-level bundle version. Bump on any breaking format change.
BUNDLE_VERSION = 1

# Canonical file names inside a <name>.jangspec/ directory.
MANIFEST_FILENAME = "jangspec.json"
INDEX_FILENAME = "target/experts.jsidx"
HOT_CORE_FILENAME = "target/hot_core.safetensors"
EXPERT_FILE_PATTERN = "target/experts-{idx:05d}.bin"

# Roll a new experts-NNNNN.bin file every 4 GB, so no single file gets silly.
EXPERT_FILE_BYTES_MAX = 4 * 1024 * 1024 * 1024

# 4 KB page alignment for direct-to-GPU reads. MTLIOCommandQueue requires
# file offsets to be aligned to the storage block size; 4 KB is the safe
# minimum on APFS-on-NVMe for M-series Macs.
BLOB_ALIGNMENT = 4096

# Magic numbers for on-disk structures (ASCII, little-endian uint32).
BLOB_MAGIC = 0x4550534A   # "JSPE"
INDEX_MAGIC = 0x58494A53  # "SJIX"

# ---- Expert blob header ----
# struct ExpertBlobHeader {
#     uint32 magic;           // BLOB_MAGIC
#     uint16 version;         // = 1
#     uint16 n_tensors;       // = 9 (gate/up/down × qweight/scales/biases)
#     uint32 layer_idx;
#     uint32 expert_id;
#     uint64 payload_offset;  // byte offset from start of blob to payload
#     uint64 payload_bytes;   // total payload length
#     // followed by n_tensors * TensorHeader
# }
BLOB_HEADER_FORMAT = "<IHHIIQQ"
BLOB_HEADER_SIZE = struct.calcsize(BLOB_HEADER_FORMAT)
assert BLOB_HEADER_SIZE == 32, f"blob header must be 32 bytes, got {BLOB_HEADER_SIZE}"

# ---- Per-tensor header (within a blob) ----
# struct TensorHeader {
#     uint8  kind;            // 0 = gate_proj, 1 = up_proj, 2 = down_proj
#     uint8  bits;             // bit width (2..8)
#     uint16 _pad;
#     uint32 dtype;            // 0 = uint32 qweight, 1 = float16 scales, 2 = float16 biases
#     uint32 dims[3];          // outer dims; unused dims are 0
#     uint64 offset;           // from start of payload
#     uint64 nbytes;
# }
TENSOR_HEADER_FORMAT = "<BBHIIIIQQ"
TENSOR_HEADER_SIZE = struct.calcsize(TENSOR_HEADER_FORMAT)
assert TENSOR_HEADER_SIZE == 36, f"tensor header must be 36 bytes, got {TENSOR_HEADER_SIZE}"

# Tensor-kind enum (matches TensorHeader.kind).
KIND_GATE = 0
KIND_UP = 1
KIND_DOWN = 2

# Dtype enum (matches TensorHeader.dtype).
DTYPE_QWEIGHT = 0   # uint32 packed
DTYPE_SCALES = 1    # float16
DTYPE_BIASES = 2    # float16

# ---- Index entry ----
# struct ExpertIndexEntry {
#     uint32 layer_idx;
#     uint32 expert_id;
#     uint16 file_id;        // index into experts-NNNNN.bin (0-based)
#     uint16 _pad;
#     uint64 offset;          // absolute byte offset in the file
#     uint64 nbytes;          // blob length including header and padding
# }
INDEX_ENTRY_FORMAT = "<IIHHQQ"
INDEX_ENTRY_SIZE = struct.calcsize(INDEX_ENTRY_FORMAT)
assert INDEX_ENTRY_SIZE == 28, f"index entry must be 28 bytes, got {INDEX_ENTRY_SIZE}"

# ---- Index file header ----
# struct IndexHeader {
#     uint32 magic;            // INDEX_MAGIC
#     uint16 version;          // = 1
#     uint16 _pad;
#     uint32 n_layers;
#     uint32 n_experts_per_layer;
#     uint64 n_entries;
# }
INDEX_HEADER_FORMAT = "<IHHIIQ"
INDEX_HEADER_SIZE = struct.calcsize(INDEX_HEADER_FORMAT)
assert INDEX_HEADER_SIZE == 24, f"index header must be 24 bytes, got {INDEX_HEADER_SIZE}"


def align_up(n: int, align: int = BLOB_ALIGNMENT) -> int:
    """Round n up to the nearest multiple of align."""
    return (n + align - 1) & ~(align - 1)
```

- [ ] **Step 3: Smoke test the module**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -c "from jang_tools.jangspec import format as f; print(f.BLOB_HEADER_SIZE, f.TENSOR_HEADER_SIZE, f.INDEX_ENTRY_SIZE, f.INDEX_HEADER_SIZE); print(f.align_up(100), f.align_up(4096), f.align_up(4097))"
```
Expected output:
```
32 36 28 24
4096 4096 8192
```

- [ ] **Step 4: Commit**

```bash
git add jang-tools/jang_tools/jangspec/__init__.py jang-tools/jang_tools/jangspec/format.py
git commit -m "jang-spec: bundle format constants and struct layouts"
```

---

## Task 2: Expert blob pack/unpack (TDD)

**Files:**
- Create: `jang-tools/jang_tools/jangspec/blob.py`
- Create: `jang-tools/tests/jangspec/__init__.py` (empty)
- Create: `jang-tools/tests/jangspec/test_blob.py`

- [ ] **Step 1: Create empty test package init**

```bash
touch jang-tools/tests/jangspec/__init__.py
```

- [ ] **Step 2: Write the failing test**

Write `jang-tools/tests/jangspec/test_blob.py`:
```python
"""Unit tests for jang_tools.jangspec.blob."""

import numpy as np
import pytest

from jang_tools.jangspec import format as fmt
from jang_tools.jangspec.blob import ExpertTensors, pack_expert_blob, unpack_expert_blob


def make_fake_expert(
    intermediate: int = 64,
    hidden: int = 32,
    bits: int = 4,
    group_size: int = 16,
) -> ExpertTensors:
    """Build a fake expert with plausible JANG shapes."""
    packed_in = (hidden * bits) // 32  # per-row uint32 count for bit-packed hidden
    assert packed_in > 0, "test shapes must produce at least one packed col"
    n_groups = hidden // group_size

    gate_q = np.arange(intermediate * packed_in, dtype=np.uint32).reshape(intermediate, packed_in)
    up_q = (gate_q + 1).astype(np.uint32)
    down_q = np.arange(hidden * ((intermediate * bits) // 32), dtype=np.uint32).reshape(
        hidden, (intermediate * bits) // 32
    )

    gate_s = np.full((intermediate, n_groups), 0.125, dtype=np.float16)
    up_s = np.full((intermediate, n_groups), 0.25, dtype=np.float16)
    down_s = np.full((hidden, intermediate // group_size), 0.0625, dtype=np.float16)

    gate_b = np.full_like(gate_s, -0.5)
    up_b = np.full_like(up_s, -0.25)
    down_b = np.full_like(down_s, -0.125)

    return ExpertTensors(
        bits=bits,
        gate=(gate_q, gate_s, gate_b),
        up=(up_q, up_s, up_b),
        down=(down_q, down_s, down_b),
    )


def test_pack_then_unpack_preserves_bytes():
    tensors = make_fake_expert()
    blob = pack_expert_blob(layer_idx=3, expert_id=17, tensors=tensors)

    assert len(blob) % fmt.BLOB_ALIGNMENT == 0, "blob must be 4 KB aligned"
    assert len(blob) >= fmt.BLOB_HEADER_SIZE + 3 * fmt.TENSOR_HEADER_SIZE

    restored = unpack_expert_blob(blob)
    assert restored.layer_idx == 3
    assert restored.expert_id == 17
    assert restored.bits == tensors.bits
    np.testing.assert_array_equal(restored.tensors.gate[0], tensors.gate[0])
    np.testing.assert_array_equal(restored.tensors.gate[1], tensors.gate[1])
    np.testing.assert_array_equal(restored.tensors.gate[2], tensors.gate[2])
    np.testing.assert_array_equal(restored.tensors.up[0], tensors.up[0])
    np.testing.assert_array_equal(restored.tensors.down[2], tensors.down[2])


def test_blob_header_magic():
    tensors = make_fake_expert()
    blob = pack_expert_blob(layer_idx=0, expert_id=0, tensors=tensors)
    import struct
    (magic,) = struct.unpack("<I", blob[:4])
    assert magic == fmt.BLOB_MAGIC


def test_unpack_rejects_wrong_magic():
    bad = bytearray(fmt.BLOB_HEADER_SIZE + 3 * fmt.TENSOR_HEADER_SIZE)
    with pytest.raises(ValueError, match="magic"):
        unpack_expert_blob(bytes(bad))
```

- [ ] **Step 3: Run the test and confirm it fails**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_blob.py -v
```
Expected: `ModuleNotFoundError: No module named 'jang_tools.jangspec.blob'`.

- [ ] **Step 4: Implement blob.py**

Write `jang-tools/jang_tools/jangspec/blob.py`:
```python
"""
ExpertBlob: a single MoE expert (gate, up, down sub-tensors) serialized into
a 4 KB-aligned byte string suitable for random-access reads via mmap or
MTLIOCommandQueue.

Layout (little-endian):
    [BLOB_HEADER_SIZE bytes]                  BlobHeader
    [3 * TENSOR_HEADER_SIZE bytes]            TensorHeader[3]
    [payload, padded to BLOB_ALIGNMENT]       contiguous tensor payloads
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from . import format as fmt

# Per-tensor triple: (qweight uint32, scales float16, biases float16).
TensorTriple = Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class ExpertTensors:
    """One expert's three sub-tensors, each a (qweight, scales, biases) triple."""

    bits: int
    gate: TensorTriple
    up: TensorTriple
    down: TensorTriple


@dataclass
class UnpackedBlob:
    """Result of unpacking an ExpertBlob from raw bytes."""

    layer_idx: int
    expert_id: int
    bits: int
    tensors: ExpertTensors


# Order in which the three sub-tensors are serialized inside a blob.
_KINDS = (
    ("gate", fmt.KIND_GATE),
    ("up", fmt.KIND_UP),
    ("down", fmt.KIND_DOWN),
)

_DTYPES = (
    (fmt.DTYPE_QWEIGHT, np.uint32),
    (fmt.DTYPE_SCALES, np.float16),
    (fmt.DTYPE_BIASES, np.float16),
)


def _dims_of(arr: np.ndarray) -> Tuple[int, int, int]:
    """Return a fixed 3-tuple of outer dims, padding with 0."""
    shape = list(arr.shape)
    while len(shape) < 3:
        shape.append(0)
    return tuple(shape[:3])  # type: ignore[return-value]


def pack_expert_blob(layer_idx: int, expert_id: int, tensors: ExpertTensors) -> bytes:
    """Serialize one expert into a 4 KB-aligned byte string."""

    triples = {
        "gate": tensors.gate,
        "up": tensors.up,
        "down": tensors.down,
    }

    tensor_entries: list[tuple[int, int, np.ndarray]] = []  # (kind, dtype_enum, array)
    for name, kind in _KINDS:
        qw, scales, biases = triples[name]
        tensor_entries.append((kind, fmt.DTYPE_QWEIGHT, np.ascontiguousarray(qw, dtype=np.uint32)))
        tensor_entries.append((kind, fmt.DTYPE_SCALES, np.ascontiguousarray(scales, dtype=np.float16)))
        tensor_entries.append((kind, fmt.DTYPE_BIASES, np.ascontiguousarray(biases, dtype=np.float16)))

    n_tensors = len(tensor_entries)
    header_area = fmt.BLOB_HEADER_SIZE + n_tensors * fmt.TENSOR_HEADER_SIZE

    # Compute per-tensor offsets relative to the start of the payload region.
    payload_parts: list[bytes] = []
    tensor_headers: list[bytes] = []
    running = 0
    for kind, dtype_enum, arr in tensor_entries:
        raw = arr.tobytes()
        dims = _dims_of(arr)
        tensor_headers.append(
            struct.pack(
                fmt.TENSOR_HEADER_FORMAT,
                kind,
                tensors.bits,
                0,
                dtype_enum,
                dims[0],
                dims[1],
                dims[2],
                running,
                len(raw),
            )
        )
        payload_parts.append(raw)
        running += len(raw)

    payload = b"".join(payload_parts)
    payload_offset = header_area
    payload_bytes = len(payload)

    blob_header = struct.pack(
        fmt.BLOB_HEADER_FORMAT,
        fmt.BLOB_MAGIC,
        1,
        n_tensors,
        layer_idx,
        expert_id,
        payload_offset,
        payload_bytes,
    )

    unpadded = blob_header + b"".join(tensor_headers) + payload
    total = fmt.align_up(len(unpadded))
    return unpadded + b"\x00" * (total - len(unpadded))


def unpack_expert_blob(blob: bytes) -> UnpackedBlob:
    """Inverse of pack_expert_blob. Numpy arrays are views on the blob bytes."""

    if len(blob) < fmt.BLOB_HEADER_SIZE:
        raise ValueError("blob too short for header")

    magic, version, n_tensors, layer_idx, expert_id, payload_offset, payload_bytes = struct.unpack(
        fmt.BLOB_HEADER_FORMAT, blob[: fmt.BLOB_HEADER_SIZE]
    )
    if magic != fmt.BLOB_MAGIC:
        raise ValueError(f"bad blob magic 0x{magic:08x}, expected 0x{fmt.BLOB_MAGIC:08x}")
    if version != 1:
        raise ValueError(f"unsupported blob version {version}")
    if n_tensors != 9:
        raise ValueError(f"expected 9 tensor entries, got {n_tensors}")

    header_cursor = fmt.BLOB_HEADER_SIZE
    bits_seen: int | None = None
    parts: dict[int, dict[int, np.ndarray]] = {
        fmt.KIND_GATE: {},
        fmt.KIND_UP: {},
        fmt.KIND_DOWN: {},
    }

    for _ in range(n_tensors):
        kind, bits, _pad, dtype_enum, d0, d1, d2, offset, nbytes = struct.unpack(
            fmt.TENSOR_HEADER_FORMAT,
            blob[header_cursor : header_cursor + fmt.TENSOR_HEADER_SIZE],
        )
        header_cursor += fmt.TENSOR_HEADER_SIZE

        if bits_seen is None:
            bits_seen = bits
        elif bits_seen != bits:
            raise ValueError(f"mixed bits in one expert blob: {bits_seen} vs {bits}")

        np_dtype = None
        for enum_val, py_dtype in _DTYPES:
            if enum_val == dtype_enum:
                np_dtype = py_dtype
                break
        if np_dtype is None:
            raise ValueError(f"unknown dtype enum {dtype_enum}")

        start = payload_offset + offset
        end = start + nbytes
        raw = blob[start:end]
        flat = np.frombuffer(raw, dtype=np_dtype)
        shape = tuple(d for d in (d0, d1, d2) if d != 0)
        arr = flat.reshape(shape) if shape else flat
        parts[kind][dtype_enum] = arr

    def triple(kind: int) -> TensorTriple:
        d = parts[kind]
        return (d[fmt.DTYPE_QWEIGHT], d[fmt.DTYPE_SCALES], d[fmt.DTYPE_BIASES])

    assert bits_seen is not None
    return UnpackedBlob(
        layer_idx=layer_idx,
        expert_id=expert_id,
        bits=bits_seen,
        tensors=ExpertTensors(
            bits=bits_seen,
            gate=triple(fmt.KIND_GATE),
            up=triple(fmt.KIND_UP),
            down=triple(fmt.KIND_DOWN),
        ),
    )
```

- [ ] **Step 5: Run the test and verify it passes**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_blob.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add jang-tools/jang_tools/jangspec/blob.py jang-tools/tests/jangspec/__init__.py jang-tools/tests/jangspec/test_blob.py
git commit -m "jang-spec: ExpertBlob pack/unpack with round-trip tests"
```

---

## Task 3: Flat expert index (TDD)

**Files:**
- Create: `jang-tools/jang_tools/jangspec/index.py`
- Create: `jang-tools/tests/jangspec/test_index.py`

- [ ] **Step 1: Write the failing test**

Write `jang-tools/tests/jangspec/test_index.py`:
```python
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_index.py -v
```
Expected: ImportError / ModuleNotFoundError for `index`.

- [ ] **Step 3: Implement index.py**

Write `jang-tools/jang_tools/jangspec/index.py`:
```python
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
```

- [ ] **Step 4: Run the test and verify it passes**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_index.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/jangspec/index.py jang-tools/tests/jangspec/test_index.py
git commit -m "jang-spec: flat expert index writer/reader"
```

---

## Task 4: Bundle manifest (TDD)

**Files:**
- Create: `jang-tools/jang_tools/jangspec/manifest.py`
- Create: `jang-tools/tests/jangspec/test_manifest.py`

- [ ] **Step 1: Write the failing test**

Write `jang-tools/tests/jangspec/test_manifest.py`:
```python
"""Unit tests for jang_tools.jangspec.manifest."""

from pathlib import Path

import pytest

from jang_tools.jangspec.manifest import Manifest, load_manifest, write_manifest


def test_manifest_roundtrip(tmp_path: Path):
    m = Manifest(
        bundle_version=1,
        source_jang="Gemma-4-26B-A4B-it-JANG_4M",
        source_jang_dir="/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M",
        target_arch="gemma4_moe",
        n_layers=48,
        n_experts_per_layer=128,
        target_top_k=8,
        tokenizer_hash="sha256:abc123",
        hot_core_tensors=["model.embed_tokens.weight", "layers.0.self_attn.q_proj.weight"],
        expert_tensor_names=["layers.N.switch_mlp.gate_proj", "layers.N.switch_mlp.up_proj", "layers.N.switch_mlp.down_proj"],
        n_experts_total=48 * 128,
        hot_core_bytes=12_000_000_000,
        expert_bytes=40_000_000_000,
        has_draft=False,
        has_router_prior=False,
    )
    path = tmp_path / "jangspec.json"
    write_manifest(path, m)

    loaded = load_manifest(path)
    assert loaded == m


def test_manifest_rejects_wrong_version(tmp_path: Path):
    path = tmp_path / "jangspec.json"
    path.write_text('{"bundle_version": 99}')
    with pytest.raises(ValueError, match="bundle_version"):
        load_manifest(path)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_manifest.py -v
```
Expected: ModuleNotFoundError for `manifest`.

- [ ] **Step 3: Implement manifest.py**

Write `jang-tools/jang_tools/jangspec/manifest.py`:
```python
"""
jangspec.json — human-readable bundle manifest.

The manifest is small (a few KB). It exists so a human or the bundle builder
can quickly inspect a bundle without parsing the binary index. The Swift
runtime parses it at load time to determine which tensors are hot-core vs
streamed, and to verify target-draft compatibility.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List

from . import format as fmt


@dataclass
class Manifest:
    bundle_version: int
    source_jang: str
    source_jang_dir: str
    target_arch: str
    n_layers: int
    n_experts_per_layer: int
    target_top_k: int
    tokenizer_hash: str
    hot_core_tensors: List[str]
    expert_tensor_names: List[str]
    n_experts_total: int
    hot_core_bytes: int
    expert_bytes: int
    has_draft: bool
    has_router_prior: bool
    draft_jang: str = ""
    tool_version: str = "jang-spec-0.1.0"
    schema: str = "jangspec/v1"


def write_manifest(path: Path, manifest: Manifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, sort_keys=True)
        f.write("\n")


def load_manifest(path: Path) -> Manifest:
    data = json.loads(Path(path).read_text())
    bv = data.get("bundle_version")
    if bv != fmt.BUNDLE_VERSION:
        raise ValueError(
            f"unsupported bundle_version {bv}, this build supports {fmt.BUNDLE_VERSION}"
        )
    return Manifest(**data)
```

- [ ] **Step 4: Run the test and verify it passes**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_manifest.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/jangspec/manifest.py jang-tools/tests/jangspec/test_manifest.py
git commit -m "jang-spec: bundle manifest schema and JSON I/O"
```

---

## Task 5: Tensor tier classification (TDD)

**Files:**
- Create: `jang-tools/jang_tools/jangspec/tier.py`
- Create: `jang-tools/tests/jangspec/test_tier.py`

- [ ] **Step 1: Write the failing test**

Write `jang-tools/tests/jangspec/test_tier.py`:
```python
"""Unit tests for jang_tools.jangspec.tier."""

import pytest

from jang_tools.jangspec.tier import TierSplit, classify_tensors, is_dense_model


def test_splits_moe_tensors():
    names = [
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.embed_tokens.biases",
        "model.norm.weight",
        "lm_head.weight",
        "lm_head.scales",
        "lm_head.biases",
        "layers.0.input_layernorm.weight",
        "layers.0.post_attention_layernorm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj.scales",
        "layers.0.self_attn.q_proj.biases",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.o_proj.weight",
        "layers.0.mlp.gate.weight",   # router
        "layers.0.mlp.gate.scales",
        "layers.0.mlp.gate.biases",
        "layers.0.switch_mlp.gate_proj.weight",
        "layers.0.switch_mlp.gate_proj.scales",
        "layers.0.switch_mlp.gate_proj.biases",
        "layers.0.switch_mlp.up_proj.weight",
        "layers.0.switch_mlp.up_proj.scales",
        "layers.0.switch_mlp.up_proj.biases",
        "layers.0.switch_mlp.down_proj.weight",
        "layers.0.switch_mlp.down_proj.scales",
        "layers.0.switch_mlp.down_proj.biases",
        "layers.0.shared_expert.gate_proj.weight",
        "layers.0.shared_expert.gate_proj.scales",
        "layers.0.shared_expert.gate_proj.biases",
    ]

    split = classify_tensors(names)
    assert isinstance(split, TierSplit)

    # Hot core: embeddings, norms, lm_head, attention, router, shared expert
    assert "model.embed_tokens.weight" in split.hot_core
    assert "lm_head.weight" in split.hot_core
    assert "layers.0.self_attn.q_proj.weight" in split.hot_core
    assert "layers.0.mlp.gate.weight" in split.hot_core
    assert "layers.0.shared_expert.gate_proj.weight" in split.hot_core
    assert "model.norm.weight" in split.hot_core

    # Expert streamed: just the switch_mlp base names, once each
    assert split.expert_base_names == [
        "layers.0.switch_mlp.down_proj",
        "layers.0.switch_mlp.gate_proj",
        "layers.0.switch_mlp.up_proj",
    ]


def test_is_dense_model_true_when_no_switch_mlp():
    names = [
        "model.embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "layers.0.mlp.up_proj.weight",
        "layers.0.mlp.down_proj.weight",
    ]
    assert is_dense_model(names) is True


def test_is_dense_model_false_when_switch_mlp_present():
    names = [
        "model.embed_tokens.weight",
        "layers.0.switch_mlp.gate_proj.weight",
    ]
    assert is_dense_model(names) is False
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_tier.py -v
```
Expected: ModuleNotFoundError for `tier`.

- [ ] **Step 3: Implement tier.py**

Write `jang-tools/jang_tools/jangspec/tier.py`:
```python
"""
Tier classification: decide which source-JANG tensors go into the hot core
(pinned in RAM) versus which get sliced per-expert and streamed from SSD.

Rules (v1, MoE-only):
  - Anything matching `*.switch_mlp.{gate,up,down}_proj.*` is an expert tensor,
    serialized per-expert into ExpertBlobs.
  - Everything else goes into the hot core.

This keeps attention, router (mlp.gate), shared experts, norms, embeddings,
and lm_head resident at all times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List

EXPERT_RE = re.compile(r"\.switch_mlp\.(gate_proj|up_proj|down_proj)\b")

# Strip trailing safetensors suffixes to get the base weight name.
_SUFFIXES = (".weight", ".scales", ".biases", ".qweight", ".bits", ".bit_map", ".block_offsets", ".zeros", ".shape")


def _base_name(tensor_name: str) -> str:
    for suf in _SUFFIXES:
        if tensor_name.endswith(suf):
            return tensor_name[: -len(suf)]
    return tensor_name


@dataclass
class TierSplit:
    hot_core: List[str] = field(default_factory=list)       # full tensor names (with suffix)
    expert_base_names: List[str] = field(default_factory=list)  # base names, sorted, unique


def classify_tensors(tensor_names: Iterable[str]) -> TierSplit:
    hot: List[str] = []
    expert_bases: set[str] = set()
    for name in tensor_names:
        if EXPERT_RE.search(name):
            expert_bases.add(_base_name(name))
        else:
            hot.append(name)
    return TierSplit(
        hot_core=sorted(hot),
        expert_base_names=sorted(expert_bases),
    )


def is_dense_model(tensor_names: Iterable[str]) -> bool:
    """Return True if no tensor matches the switch_mlp expert pattern."""
    for name in tensor_names:
        if EXPERT_RE.search(name):
            return False
    return True
```

- [ ] **Step 4: Run the test and verify it passes**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_tier.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add jang-tools/jang_tools/jangspec/tier.py jang-tools/tests/jangspec/test_tier.py
git commit -m "jang-spec: tier classifier — hot-core vs streamed experts"
```

---

## Task 6: Bundle builder — end-to-end (TDD, fixture-backed)

**Files:**
- Create: `jang-tools/jang_tools/jangspec/builder.py`
- Create: `jang-tools/tests/jangspec/conftest.py`
- Create: `jang-tools/tests/jangspec/test_builder.py`

- [ ] **Step 1: Write the fixture loader**

Write `jang-tools/tests/jangspec/conftest.py`:
```python
"""Shared fixtures for jangspec tests."""

import os
from pathlib import Path

import pytest

DEFAULT_FIXTURE = "/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M"


@pytest.fixture(scope="session")
def jangspec_fixture_model() -> Path:
    """Return the path to a small MoE JANG model for integration tests.

    Set JANGSPEC_TEST_MODEL to override. Skips the test if the path is missing.
    """
    raw = os.environ.get("JANGSPEC_TEST_MODEL", DEFAULT_FIXTURE)
    path = Path(raw)
    if not (path / "config.json").exists():
        pytest.skip(f"fixture MoE model not found at {path}; set JANGSPEC_TEST_MODEL to override")
    return path
```

- [ ] **Step 2: Write the failing integration test**

Write `jang-tools/tests/jangspec/test_builder.py`:
```python
"""Integration tests for jang_tools.jangspec.builder against a real JANG model."""

from pathlib import Path

from jang_tools.jangspec import format as fmt
from jang_tools.jangspec.builder import JangSpecBuilder
from jang_tools.jangspec.index import read_index
from jang_tools.jangspec.manifest import load_manifest


def test_build_creates_all_bundle_files(jangspec_fixture_model: Path, tmp_path: Path):
    out = tmp_path / "fixture.jangspec"
    builder = JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out)
    builder.build()

    # Required files exist
    assert (out / fmt.MANIFEST_FILENAME).exists()
    assert (out / fmt.INDEX_FILENAME).exists()
    assert (out / fmt.HOT_CORE_FILENAME).exists()
    assert (out / "tokenizer.json").exists()
    # At least one experts-NNNNN.bin was emitted
    shards = sorted(out.glob("target/experts-*.bin"))
    assert len(shards) >= 1

    # Manifest is loadable and reports an MoE model
    manifest = load_manifest(out / fmt.MANIFEST_FILENAME)
    assert manifest.n_experts_per_layer > 1
    assert manifest.n_layers > 0
    assert manifest.n_experts_total == manifest.n_layers * manifest.n_experts_per_layer
    assert manifest.has_draft is False
    assert manifest.has_router_prior is False

    # Index is loadable and has one entry per (layer, expert)
    idx = read_index(out / fmt.INDEX_FILENAME)
    assert idx.n_layers == manifest.n_layers
    assert idx.n_experts_per_layer == manifest.n_experts_per_layer
    assert len(idx.entries) == manifest.n_experts_total

    # Every blob offset is 4 KB-aligned and fits inside its file
    for e in idx.entries:
        assert e.offset % fmt.BLOB_ALIGNMENT == 0, f"unaligned blob at {e}"
        shard = out / f"target/experts-{e.file_id:05d}.bin"
        assert shard.exists()
        assert e.offset + e.nbytes <= shard.stat().st_size
```

- [ ] **Step 3: Run and confirm it fails**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_builder.py -v
```
Expected: ModuleNotFoundError for `builder`. (If the fixture model is missing, the test will skip — that's fine but we won't get coverage until it's present.)

- [ ] **Step 4: Implement builder.py**

Write `jang-tools/jang_tools/jangspec/builder.py`:
```python
"""
JangSpecBuilder — convert a source JANG (MoE) model into a .jangspec bundle.

Strategy:
    1. Enumerate the source safetensors shards via the index file.
    2. Classify tensor names into hot_core vs streamed experts (tier.py).
    3. Copy hot_core tensors into a single new safetensors file via mmap.
    4. For each (layer, expert_id) tuple, slice the per-expert (qweight, scales,
       biases) triples out of the stacked 3D MoE tensors, pack into an
       ExpertBlob, append to a rolling experts-NNNNN.bin file.
    5. Record every blob's file/offset/nbytes in an ExpertIndexEntry.
    6. Write experts.jsidx and jangspec.json.
    7. Copy tokenizer files.

v1 does NOT populate `draft/` or `router_prior/`. Those are left empty until
later plans. The manifest records `has_draft=False`, `has_router_prior=False`.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from . import format as fmt
from .blob import ExpertTensors, pack_expert_blob
from .index import ExpertIndexEntry, write_index
from .manifest import Manifest, write_manifest
from .tier import classify_tensors, is_dense_model

# Matches the layer index in tensor names like "layers.7.switch_mlp.gate_proj.weight".
_LAYER_RE = re.compile(r"\.?layers\.(\d+)\.")


def _layer_idx(name: str) -> int:
    m = _LAYER_RE.search(name)
    if not m:
        raise ValueError(f"cannot parse layer index from {name!r}")
    return int(m.group(1))


def _infer_bits(qweight: np.ndarray, scales: np.ndarray, group_size: int) -> int:
    """Recover per-tensor bit width from the uint32 packed shape.

    packed_in = ceil(in_features * bits / 32), in_features = n_groups * group_size
    """
    packed_in = qweight.shape[-1]
    n_groups = scales.shape[-1]
    in_features = n_groups * group_size
    return (packed_in * 32) // in_features


@dataclass
class _ExpertShard:
    """State for the currently-open experts-NNNNN.bin shard file."""

    file_id: int
    path: Path
    size: int = 0

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "wb")

    def close(self) -> None:
        self.handle.close()

    def write(self, blob: bytes) -> Tuple[int, int]:
        """Write a blob, return (offset, nbytes)."""
        offset = self.size
        self.handle.write(blob)
        self.size += len(blob)
        return offset, len(blob)


class JangSpecBuilder:
    def __init__(self, source_dir: Path, out_dir: Path):
        self.source_dir = Path(source_dir)
        self.out_dir = Path(out_dir)

    # ------------------------------------------------------------------ public

    def build(self) -> None:
        self._load_metadata()
        self._validate_moe()
        self._split_tiers()
        self._write_hot_core()
        self._write_experts_and_index()
        self._copy_tokenizer()
        self._write_manifest()

    # ---------------------------------------------------------------- metadata

    def _load_metadata(self) -> None:
        self.config = json.loads((self.source_dir / "config.json").read_text())
        try:
            self.jang_config = json.loads((self.source_dir / "jang_config.json").read_text())
        except FileNotFoundError:
            self.jang_config = {}

        index_path = self.source_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing {index_path}")
        self.st_index = json.loads(index_path.read_text())
        self.shard_map: Dict[str, str] = self.st_index["weight_map"]  # tensor_name -> shard file
        self.all_tensor_names: List[str] = sorted(self.shard_map.keys())

        q = self.config.get("quantization", {})
        self.group_size = int(q.get("group_size", 64))
        self.target_top_k = int(
            self.config.get("num_experts_per_tok")
            or self.config.get("moe_top_k")
            or self.config.get("num_local_experts_per_tok")
            or 2
        )

    def _validate_moe(self) -> None:
        if is_dense_model(self.all_tensor_names):
            raise ValueError(
                f"{self.source_dir} appears to be a dense JANG model. "
                "jang-spec only supports MoE targets — dense models already fit in RAM."
            )

    def _split_tiers(self) -> None:
        self.split = classify_tensors(self.all_tensor_names)

    # --------------------------------------------------------------- hot core

    def _open_shard(self, shard_filename: str):
        path = self.source_dir / shard_filename
        return safe_open(path, framework="numpy", device="cpu")

    def _write_hot_core(self) -> None:
        # Gather all hot-core tensors by source shard to minimise safe_open calls.
        by_shard: Dict[str, List[str]] = {}
        for name in self.split.hot_core:
            by_shard.setdefault(self.shard_map[name], []).append(name)

        tensors: Dict[str, np.ndarray] = {}
        hot_bytes = 0
        for shard_filename, names in by_shard.items():
            with self._open_shard(shard_filename) as f:
                for name in names:
                    arr = f.get_tensor(name)
                    tensors[name] = arr
                    hot_bytes += arr.nbytes

        out_path = self.out_dir / fmt.HOT_CORE_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(out_path))
        self.hot_core_bytes = hot_bytes

    # -------------------------------------------------------------- experts

    def _write_experts_and_index(self) -> None:
        # Load the three 3D stacked expert tensors per layer, on demand.
        # Each base name has suffixes .weight (uint32), .scales (f16), .biases (f16).
        expert_bases = self.split.expert_base_names

        # Group the base names by layer.
        layers: Dict[int, Dict[str, str]] = {}  # layer_idx -> {"gate_proj": base, "up_proj": base, "down_proj": base}
        for base in expert_bases:
            lid = _layer_idx(base)
            for kind in ("gate_proj", "up_proj", "down_proj"):
                if base.endswith(f".switch_mlp.{kind}"):
                    layers.setdefault(lid, {})[kind] = base
                    break

        sorted_layers = sorted(layers.keys())
        if not sorted_layers:
            raise ValueError("no expert layers detected after tier split")

        # Peek the first layer to discover E (num experts) and validate shapes.
        first = layers[sorted_layers[0]]
        shard_of = lambda name: self.shard_map[name + ".weight"]
        with self._open_shard(shard_of(first["gate_proj"])) as f:
            gate_q = f.get_tensor(first["gate_proj"] + ".weight")
        if gate_q.ndim != 3:
            raise ValueError(
                f"expected stacked 3D expert tensor, got shape {gate_q.shape} for {first['gate_proj']}"
            )
        self.n_experts_per_layer = int(gate_q.shape[0])
        self.n_layers = len(sorted_layers)
        assert sorted_layers == list(range(self.n_layers)), (
            f"expert layer indices are not contiguous: {sorted_layers}"
        )

        # Open the first shard file.
        shards: List[_ExpertShard] = []
        current = _ExpertShard(file_id=0, path=self.out_dir / fmt.EXPERT_FILE_PATTERN.format(idx=0))
        current.open()
        shards.append(current)

        entries: List[ExpertIndexEntry] = []

        for lid in sorted_layers:
            base = layers[lid]
            arrays: Dict[str, Dict[str, np.ndarray]] = {}
            for kind in ("gate_proj", "up_proj", "down_proj"):
                name = base[kind]
                with self._open_shard(shard_of(name)) as f:
                    arrays[kind] = {
                        "qweight": f.get_tensor(name + ".weight"),
                        "scales": f.get_tensor(name + ".scales"),
                        "biases": f.get_tensor(name + ".biases"),
                    }

            bits = _infer_bits(arrays["gate_proj"]["qweight"], arrays["gate_proj"]["scales"], self.group_size)

            for expert_id in range(self.n_experts_per_layer):
                expert = ExpertTensors(
                    bits=bits,
                    gate=(
                        np.ascontiguousarray(arrays["gate_proj"]["qweight"][expert_id]),
                        np.ascontiguousarray(arrays["gate_proj"]["scales"][expert_id]),
                        np.ascontiguousarray(arrays["gate_proj"]["biases"][expert_id]),
                    ),
                    up=(
                        np.ascontiguousarray(arrays["up_proj"]["qweight"][expert_id]),
                        np.ascontiguousarray(arrays["up_proj"]["scales"][expert_id]),
                        np.ascontiguousarray(arrays["up_proj"]["biases"][expert_id]),
                    ),
                    down=(
                        np.ascontiguousarray(arrays["down_proj"]["qweight"][expert_id]),
                        np.ascontiguousarray(arrays["down_proj"]["scales"][expert_id]),
                        np.ascontiguousarray(arrays["down_proj"]["biases"][expert_id]),
                    ),
                )
                blob = pack_expert_blob(layer_idx=lid, expert_id=expert_id, tensors=expert)

                # Roll shard if adding this blob would exceed the per-file ceiling.
                if current.size + len(blob) > fmt.EXPERT_FILE_BYTES_MAX and current.size > 0:
                    current.close()
                    current = _ExpertShard(
                        file_id=current.file_id + 1,
                        path=self.out_dir / fmt.EXPERT_FILE_PATTERN.format(idx=current.file_id + 1),
                    )
                    current.open()
                    shards.append(current)

                offset, nbytes = current.write(blob)
                entries.append(
                    ExpertIndexEntry(
                        layer_idx=lid,
                        expert_id=expert_id,
                        file_id=current.file_id,
                        offset=offset,
                        nbytes=nbytes,
                    )
                )

            # Release numpy views early to keep RAM use bounded.
            del arrays

        current.close()

        write_index(
            self.out_dir / fmt.INDEX_FILENAME,
            entries=entries,
            n_layers=self.n_layers,
            n_experts_per_layer=self.n_experts_per_layer,
        )

        self.expert_bytes = sum(e.nbytes for e in entries)

    # ------------------------------------------------------------- tokenizer

    def _copy_tokenizer(self) -> None:
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = self.source_dir / name
            if src.exists():
                shutil.copy2(src, self.out_dir / name)
        # Copy source jang_config.json and config.json into target/ for Swift reader.
        (self.out_dir / "target").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "jang_config.json"):
            src = self.source_dir / name
            if src.exists():
                shutil.copy2(src, self.out_dir / "target" / name)

    # -------------------------------------------------------------- manifest

    def _write_manifest(self) -> None:
        tokenizer_path = self.out_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tok_hash = "sha256:" + hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        else:
            tok_hash = ""

        manifest = Manifest(
            bundle_version=fmt.BUNDLE_VERSION,
            source_jang=self.source_dir.name,
            source_jang_dir=str(self.source_dir),
            target_arch=self.config.get("model_type", "unknown"),
            n_layers=self.n_layers,
            n_experts_per_layer=self.n_experts_per_layer,
            target_top_k=self.target_top_k,
            tokenizer_hash=tok_hash,
            hot_core_tensors=self.split.hot_core,
            expert_tensor_names=self.split.expert_base_names,
            n_experts_total=self.n_layers * self.n_experts_per_layer,
            hot_core_bytes=self.hot_core_bytes,
            expert_bytes=self.expert_bytes,
            has_draft=False,
            has_router_prior=False,
        )
        write_manifest(self.out_dir / fmt.MANIFEST_FILENAME, manifest)
```

- [ ] **Step 5: Run the integration test**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_builder.py -v -s
```
Expected: 1 passed (or skipped if fixture model is absent — if skipped, investigate whether a local MoE fixture can be downloaded before proceeding).

- [ ] **Step 6: Commit**

```bash
git add jang-tools/jang_tools/jangspec/builder.py jang-tools/tests/jangspec/conftest.py jang-tools/tests/jangspec/test_builder.py
git commit -m "jang-spec: JangSpecBuilder — source JANG -> .jangspec bundle"
```

---

## Task 7: Bundle reader and round-trip test

**Files:**
- Create: `jang-tools/jang_tools/jangspec/reader.py`
- Create: `jang-tools/tests/jangspec/test_reader.py`

- [ ] **Step 1: Write the failing round-trip test**

Write `jang-tools/tests/jangspec/test_reader.py`:
```python
"""Round-trip tests: build a bundle, reload it, verify bytes match source."""

from pathlib import Path

import numpy as np
from safetensors import safe_open

from jang_tools.jangspec.builder import JangSpecBuilder
from jang_tools.jangspec.reader import JangSpecReader


def test_reader_matches_source_bytes(jangspec_fixture_model: Path, tmp_path: Path):
    out = tmp_path / "fixture.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out).build()

    reader = JangSpecReader(out)
    assert reader.n_layers > 0
    assert reader.n_experts_per_layer > 1

    # Pick layer 0 expert 0, load it through the reader.
    unpacked = reader.load_expert(layer_idx=0, expert_id=0)
    assert unpacked.layer_idx == 0
    assert unpacked.expert_id == 0

    # Find source tensors for layer 0 and slice expert 0 directly.
    import json
    st_index = json.loads((jangspec_fixture_model / "model.safetensors.index.json").read_text())
    shard_map = st_index["weight_map"]

    gate_base = next(
        n for n in reader.manifest.expert_tensor_names if n.endswith("layers.0.switch_mlp.gate_proj")
    )

    def _load(name: str, suffix: str) -> np.ndarray:
        shard = jangspec_fixture_model / shard_map[name + suffix]
        with safe_open(shard, framework="numpy", device="cpu") as f:
            return f.get_tensor(name + suffix)

    gate_q_all = _load(gate_base, ".weight")
    gate_s_all = _load(gate_base, ".scales")
    gate_b_all = _load(gate_base, ".biases")

    np.testing.assert_array_equal(unpacked.tensors.gate[0], gate_q_all[0])
    np.testing.assert_array_equal(unpacked.tensors.gate[1], gate_s_all[0])
    np.testing.assert_array_equal(unpacked.tensors.gate[2], gate_b_all[0])


def test_reader_random_access(jangspec_fixture_model: Path, tmp_path: Path):
    out = tmp_path / "fixture.jangspec"
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out).build()

    reader = JangSpecReader(out)
    last_layer = reader.n_layers - 1
    last_expert = reader.n_experts_per_layer - 1
    unpacked = reader.load_expert(layer_idx=last_layer, expert_id=last_expert)
    assert unpacked.layer_idx == last_layer
    assert unpacked.expert_id == last_expert
    assert unpacked.tensors.gate[0].size > 0
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_reader.py -v
```
Expected: ModuleNotFoundError for `reader`.

- [ ] **Step 3: Implement reader.py**

Write `jang-tools/jang_tools/jangspec/reader.py`:
```python
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
```

- [ ] **Step 4: Run the tests and verify they pass**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/test_reader.py -v -s
```
Expected: 2 passed (or skipped if fixture missing).

- [ ] **Step 5: Run the full jangspec test suite**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/jangspec/ -v
```
Expected: all unit tests pass; integration tests pass or skip consistently.

- [ ] **Step 6: Commit**

```bash
git add jang-tools/jang_tools/jangspec/reader.py jang-tools/tests/jangspec/test_reader.py
git commit -m "jang-spec: JangSpecReader round-trip against source tensors"
```

---

## Task 8: CLI subcommand — `jang spec build`

**Files:**
- Create: `jang-tools/jang_tools/jangspec/cli.py`
- Modify: `jang-tools/jang_tools/__main__.py`

- [ ] **Step 1: Implement the CLI module**

Write `jang-tools/jang_tools/jangspec/cli.py`:
```python
"""`jang spec ...` subcommand implementations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import JangSpecBuilder
from .reader import JangSpecReader


def cmd_build(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    if out.exists() and not args.force:
        print(f"error: output directory {out} already exists (use --force to overwrite)")
        return 1
    builder = JangSpecBuilder(source_dir=source, out_dir=out)
    builder.build()
    print(f"  built bundle: {out}")
    print(f"    layers:       {builder.n_layers}")
    print(f"    experts/layer:{builder.n_experts_per_layer}")
    print(f"    hot_core:     {builder.hot_core_bytes / 1e9:.2f} GB")
    print(f"    expert_bytes: {builder.expert_bytes / 1e9:.2f} GB")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    reader = JangSpecReader(Path(args.bundle))
    m = reader.manifest
    print(f"  bundle:        {args.bundle}")
    print(f"  source jang:   {m.source_jang}")
    print(f"  arch:          {m.target_arch}")
    print(f"  n_layers:      {m.n_layers}")
    print(f"  experts/layer: {m.n_experts_per_layer}")
    print(f"  top_k:         {m.target_top_k}")
    print(f"  hot_core:      {m.hot_core_bytes / 1e9:.2f} GB")
    print(f"  expert_bytes:  {m.expert_bytes / 1e9:.2f} GB")
    print(f"  draft:         {m.has_draft}")
    print(f"  router_prior:  {m.has_router_prior}")
    return 0


def register_subparsers(spec_parser: argparse.ArgumentParser) -> None:
    sub = spec_parser.add_subparsers(dest="spec_cmd", required=True)

    build = sub.add_parser("build", help="Build a .jangspec bundle from a source JANG MoE model")
    build.add_argument("source", help="Path to source JANG model directory")
    build.add_argument("--out", required=True, help="Path to output .jangspec directory")
    build.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    build.set_defaults(func=cmd_build)

    inspect = sub.add_parser("inspect", help="Inspect a .jangspec bundle")
    inspect.add_argument("bundle", help="Path to .jangspec directory")
    inspect.set_defaults(func=cmd_inspect)
```

- [ ] **Step 2: Wire the subcommand into `__main__.py`**

Find the existing subparser registration in `jang-tools/jang_tools/__main__.py` (search for `add_subparsers`). After the last existing subparser registration and before `args = parser.parse_args()`, add:

```python
    # --- spec subcommand (jang-spec bundle tooling) ---
    from .jangspec.cli import register_subparsers as _register_spec
    spec_parser = subparsers.add_parser("spec", help="jang-spec bundle tooling")
    _register_spec(spec_parser)
```

And in the dispatch block at the end of `main()` (where other subcommands call their `cmd_*` functions), add:

```python
    if args.command == "spec":
        return args.func(args)
```

(If the existing code uses `args.command` to dispatch, just add the new branch. If it uses `hasattr(args, "func")`, the existing dispatch will already pick it up.)

- [ ] **Step 3: Reinstall the editable package so the new subcommand is wired**

Run:
```bash
cd /Users/eric/jang/jang-tools && pip install -e . --quiet
```
Expected: `Successfully installed jang-...`.

- [ ] **Step 4: Smoke-test the CLI**

Run:
```bash
jang spec --help
```
Expected: help output showing `build` and `inspect` subcommands.

- [ ] **Step 5: End-to-end CLI build (skips if fixture missing)**

Run:
```bash
if [ -d /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M ]; then \
  jang spec build /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M \
    --out /tmp/fixture.jangspec --force && \
  jang spec inspect /tmp/fixture.jangspec; \
else \
  echo "fixture absent, skipping"; \
fi
```
Expected: bundle is built and inspected; summary output shows a nonzero number of experts.

- [ ] **Step 6: Commit**

```bash
git add jang-tools/jang_tools/jangspec/cli.py jang-tools/jang_tools/__main__.py
git commit -m "jang-spec: jang spec build/inspect CLI"
```

---

## Task 9: Swift `MTLIOCommandQueue` IO benchmark (Spike A)

**Files:**
- Modify: `jang-runtime/Package.swift`
- Create: `jang-runtime/Sources/jang-spec-iobench/main.swift`

- [ ] **Step 1: Add the executable target to Package.swift**

Open `jang-runtime/Package.swift` and add to `products`:
```swift
        .executable(name: "jang-spec-iobench", targets: ["JangSpecIOBench"]),
```

And to `targets`:
```swift
        .executableTarget(
            name: "JangSpecIOBench",
            dependencies: [],
            path: "Sources/jang-spec-iobench"
        ),
```

- [ ] **Step 2: Write the benchmark**

Write `jang-runtime/Sources/jang-spec-iobench/main.swift`:
```swift
// jang-spec-iobench — measure Mac NVMe -> unified memory throughput
// via MTLIOCommandQueue, compared to plain pread.
//
// Fixtures: creates N files of SIZE bytes each under a tmpdir, fills them
// with a fast PRNG (not zeros — apfs/nvme may shortcut zero reads), then:
//   (1) sequential MTLIOCommandQueue reads into MTLBuffer
//   (2) random-order MTLIOCommandQueue reads
//   (3) random-order pread into a Data buffer
// and reports GB/s + per-read latency p50/p99.

import Foundation
import Metal
import MetalKit

// ----- Config -----
let NUM_FILES = 256
let FILE_BYTES = 50 * 1024 * 1024   // 50 MB per file, ~one expert's worth
let ALIGN = 4096

@inline(__always)
func nowNs() -> UInt64 { DispatchTime.now().uptimeNanoseconds }

func xorshift(_ s: inout UInt64) -> UInt64 {
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    s = x
    return x
}

func fillRandom(_ buf: UnsafeMutableRawPointer, _ len: Int, seed: UInt64) {
    var s = seed | 1
    let p = buf.assumingMemoryBound(to: UInt64.self)
    let n = len / 8
    for i in 0..<n {
        p[i] = xorshift(&s)
    }
}

func percentiles(_ xs: [Double]) -> (p50: Double, p99: Double) {
    let s = xs.sorted()
    let p50 = s[s.count / 2]
    let p99 = s[min(s.count - 1, Int(Double(s.count) * 0.99))]
    return (p50, p99)
}

func makeFixtures(dir: URL) throws -> [URL] {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var urls: [URL] = []
    let buf = UnsafeMutableRawPointer.allocate(byteCount: FILE_BYTES, alignment: ALIGN)
    defer { buf.deallocate() }

    print("  creating \(NUM_FILES) × \(FILE_BYTES / 1024 / 1024) MB fixture files under \(dir.path) ...")
    let t0 = nowNs()
    for i in 0..<NUM_FILES {
        let url = dir.appendingPathComponent(String(format: "f-%05d.bin", i))
        fillRandom(buf, FILE_BYTES, seed: UInt64(i + 1))
        let data = Data(bytesNoCopy: buf, count: FILE_BYTES, deallocator: .none)
        try data.write(to: url)
        urls.append(url)
    }
    let elapsed = Double(nowNs() - t0) / 1e9
    let totalGB = Double(NUM_FILES * FILE_BYTES) / 1e9
    print(String(format: "    %.2f GB written in %.1fs (%.1f GB/s)", totalGB, elapsed, totalGB / elapsed))
    // Flush the page cache by running `purge` — optional, requires sudo, best-effort.
    return urls
}

func benchIOCommandQueue(device: MTLDevice, urls: [URL], random: Bool) throws -> (gbPerSec: Double, p50ms: Double, p99ms: Double) {
    guard #available(macOS 13.0, *) else {
        print("  MTLIOCommandQueue requires macOS 13+")
        exit(1)
    }
    let ioQueue: MTLIOCommandQueue
    let desc = MTLIOCommandQueueDescriptor()
    desc.type = .concurrent
    desc.priority = .normal
    ioQueue = try device.makeIOCommandQueue(descriptor: desc)

    // Buffers: one per file.
    var buffers: [MTLBuffer] = []
    buffers.reserveCapacity(urls.count)
    for _ in urls {
        guard let b = device.makeBuffer(length: FILE_BYTES, options: [.storageModeShared]) else {
            throw NSError(domain: "iobench", code: 1)
        }
        buffers.append(b)
    }

    // File handles for the IO queue.
    var handles: [MTLIOFileHandle] = []
    for url in urls {
        let h = try device.makeIOFileHandle(url: url)
        handles.append(h)
    }

    // Build an order.
    var order = Array(0..<urls.count)
    if random { order.shuffle() }

    var latencies: [Double] = []
    latencies.reserveCapacity(order.count)

    let tStart = nowNs()
    for i in order {
        let cb = ioQueue.makeCommandBuffer()
        cb.load(
            buffer: buffers[i],
            offset: 0,
            size: FILE_BYTES,
            sourceHandle: handles[i],
            sourceHandleOffset: 0
        )
        let before = nowNs()
        cb.commit()
        cb.waitUntilCompleted()
        let after = nowNs()
        latencies.append(Double(after - before) / 1e6) // ms
    }
    let elapsed = Double(nowNs() - tStart) / 1e9
    let totalGB = Double(order.count * FILE_BYTES) / 1e9
    let (p50, p99) = percentiles(latencies)
    return (totalGB / elapsed, p50, p99)
}

func benchPread(urls: [URL], random: Bool) throws -> (gbPerSec: Double, p50ms: Double, p99ms: Double) {
    var order = Array(0..<urls.count)
    if random { order.shuffle() }

    let buf = UnsafeMutableRawPointer.allocate(byteCount: FILE_BYTES, alignment: ALIGN)
    defer { buf.deallocate() }

    var latencies: [Double] = []
    latencies.reserveCapacity(order.count)

    let tStart = nowNs()
    for i in order {
        let fd = open(urls[i].path, O_RDONLY)
        if fd < 0 { throw NSError(domain: "iobench", code: 2) }
        defer { close(fd) }
        let before = nowNs()
        var off: off_t = 0
        var remaining = FILE_BYTES
        var p = buf
        while remaining > 0 {
            let n = pread(fd, p, remaining, off)
            if n <= 0 { break }
            remaining -= n
            off += off_t(n)
            p = p.advanced(by: n)
        }
        let after = nowNs()
        latencies.append(Double(after - before) / 1e6)
    }
    let elapsed = Double(nowNs() - tStart) / 1e9
    let totalGB = Double(order.count * FILE_BYTES) / 1e9
    let (p50, p99) = percentiles(latencies)
    return (totalGB / elapsed, p50, p99)
}

// ----- main -----
let tmp = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("jang-spec-iobench")
defer { try? FileManager.default.removeItem(at: tmp) }

guard let device = MTLCreateSystemDefaultDevice() else {
    print("no Metal device")
    exit(1)
}
print("  device: \(device.name)")

let urls = try makeFixtures(dir: tmp)

print("\n  (1) MTLIOCommandQueue, sequential order")
let seq = try benchIOCommandQueue(device: device, urls: urls, random: false)
print(String(format: "    throughput: %.2f GB/s   p50: %.2f ms   p99: %.2f ms", seq.gbPerSec, seq.p50ms, seq.p99ms))

print("\n  (2) MTLIOCommandQueue, random order")
let rnd = try benchIOCommandQueue(device: device, urls: urls, random: true)
print(String(format: "    throughput: %.2f GB/s   p50: %.2f ms   p99: %.2f ms", rnd.gbPerSec, rnd.p50ms, rnd.p99ms))

print("\n  (3) pread, random order")
let pr = try benchPread(urls: urls, random: true)
print(String(format: "    throughput: %.2f GB/s   p50: %.2f ms   p99: %.2f ms", pr.gbPerSec, pr.p50ms, pr.p99ms))

print("\n  verdict:")
if rnd.gbPerSec >= 3.0 && rnd.p99ms <= 5.0 {
    print("    GO — random-access streaming meets design thresholds")
} else {
    print(String(format: "    REVISIT — want >= 3 GB/s random and <= 5 ms p99, got %.2f GB/s / %.2f ms",
                 rnd.gbPerSec, rnd.p99ms))
}
```

- [ ] **Step 3: Build**

Run:
```bash
cd /Users/eric/jang/jang-runtime && swift build -c release --product jang-spec-iobench
```
Expected: build success.

- [ ] **Step 4: Run the benchmark**

Run:
```bash
cd /Users/eric/jang/jang-runtime && ./.build/release/jang-spec-iobench 2>&1 | tee /tmp/jang-spec-iobench.log
```
Expected: three measurement blocks and a verdict line.

- [ ] **Step 5: Commit**

```bash
git add jang-runtime/Package.swift jang-runtime/Sources/jang-spec-iobench/main.swift
git commit -m "jang-spec: Swift MTLIOCommandQueue vs pread benchmark"
```

---

## Task 10: Capture IO benchmark result in a note

**Files:**
- Create: `docs/superpowers/notes/2026-04-13-iobench-results.md`

- [ ] **Step 1: Write the note from the benchmark output**

Copy the relevant lines from `/tmp/jang-spec-iobench.log` into `docs/superpowers/notes/2026-04-13-iobench-results.md`:

```markdown
# jang-spec IO Benchmark — 2026-04-13

Spike A from `docs/superpowers/specs/2026-04-13-jang-spec-design.md` §14.

**Machine:** [fill in: Mac Studio model, SSD size]
**Device (Metal):** [paste from benchmark]

## Results

**MTLIOCommandQueue, sequential**
- throughput: `[paste]` GB/s
- p50: `[paste]` ms
- p99: `[paste]` ms

**MTLIOCommandQueue, random**
- throughput: `[paste]` GB/s
- p50: `[paste]` ms
- p99: `[paste]` ms

**pread, random**
- throughput: `[paste]` GB/s
- p50: `[paste]` ms
- p99: `[paste]` ms

## Verdict

[GO / REVISIT — paste verdict line]

## Implications for the spec

- [ ] If GO: proceed to Plan 2 (JANGCore Swift v2 loader) unchanged.
- [ ] If REVISIT: note which assumption breaks and how to adapt (fewer experts per verification, larger expert blobs, or fall back to pread with io_uring-style batching).
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/notes/2026-04-13-iobench-results.md
git commit -m "jang-spec: record IO benchmark spike A results"
```

---

## Task 11: README for the jangspec subpackage

**Files:**
- Create: `jang-tools/jang_tools/jangspec/README.md`

- [ ] **Step 1: Write the README**

Write `jang-tools/jang_tools/jangspec/README.md`:
```markdown
# jang_tools.jangspec

Bundle format and builder for jang-spec — SSD-streamed MoE speculative decoding.

See `docs/superpowers/specs/2026-04-13-jang-spec-design.md` for the full design.

## CLI

```bash
# Build a bundle from a source JANG MoE model
jang spec build /path/to/Qwen3.5-35B-A3B-JANG_2S --out /path/to/out.jangspec

# Inspect a bundle
jang spec inspect /path/to/out.jangspec
```

## Modules

| File | Purpose |
|---|---|
| `format.py` | On-disk constants, struct layouts, magic numbers, alignment |
| `blob.py` | Pack/unpack ExpertBlob (one expert: gate/up/down triples) |
| `index.py` | Flat experts.jsidx writer/reader |
| `manifest.py` | jangspec.json schema + JSON I/O |
| `tier.py` | Classify tensors: hot-core vs streamed experts |
| `builder.py` | JangSpecBuilder end-to-end |
| `reader.py` | JangSpecReader for tests and Swift-parity checks |
| `cli.py` | `jang spec build/inspect` subcommands |

## Format

A `.jangspec` directory contains:

```
<name>.jangspec/
  jangspec.json                 Manifest (bundle_version, tensor lists, sizes)
  tokenizer.json                Shared tokenizer
  tokenizer_config.json
  target/
    config.json                 Copied from source
    jang_config.json            Copied from source
    hot_core.safetensors        Pinned-resident tensors (attn, router, norms, embed, lm_head, shared experts)
    experts.jsidx               Flat binary index: (layer, expert_id) -> (file, offset, nbytes)
    experts-00001.bin           Raw ExpertBlob payloads, 4 KB-aligned
    experts-00002.bin           (rolled at 4 GB per file)
    ...
```

v1 does NOT include `draft/` or `router_prior/`. Those are populated by Plans 2 and 3.
```

- [ ] **Step 2: Commit**

```bash
git add jang-tools/jang_tools/jangspec/README.md
git commit -m "jang-spec: README for the jangspec Python subpackage"
```

---

## Task 12: Final sweep

**Files:** none

- [ ] **Step 1: Run the full jang-tools test suite**

Run:
```bash
cd /Users/eric/jang/jang-tools && python -m pytest tests/ -v
```
Expected: all tests pass or skip with fixture-absent messages; no failures.

- [ ] **Step 2: Rebuild the Swift target one more time**

Run:
```bash
cd /Users/eric/jang/jang-runtime && swift build -c release
```
Expected: build success with no warnings in `jang-spec-iobench`.

- [ ] **Step 3: Verify the CLI is installed**

Run:
```bash
jang spec --help && jang spec build --help && jang spec inspect --help
```
Expected: three help outputs.

- [ ] **Step 4: Push the branch**

Run:
```bash
cd /Users/eric/jang && git log --oneline main..HEAD
```
Expected: the 12-or-so commits from this plan, in order.

- [ ] **Step 5: Summarise the plan outcomes**

Write a one-paragraph summary for Eric:
- Bundle format shipped, round-trips against a real JANG MoE model.
- IO benchmark result (paste verdict).
- Next plan: JANGCore Swift v2 loader with MoE support.
