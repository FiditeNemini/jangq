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
