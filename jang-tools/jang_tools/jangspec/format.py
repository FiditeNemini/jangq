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
# M122 (iter 46): plain `assert` is stripped by `python -O`. These sizes
# gate the on-disk binary layout; a silent skip under -O would let a
# broken format change (e.g. adding a field to the struct but forgetting
# to update the size constant) through to runtime, where readers would
# misalign and corrupt tensors. Use an unconditional raise.
if BLOB_HEADER_SIZE != 32:
    raise ImportError(f"blob header must be 32 bytes, got {BLOB_HEADER_SIZE}")

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
if TENSOR_HEADER_SIZE != 36:
    raise ImportError(f"tensor header must be 36 bytes, got {TENSOR_HEADER_SIZE}")

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
if INDEX_ENTRY_SIZE != 28:
    raise ImportError(f"index entry must be 28 bytes, got {INDEX_ENTRY_SIZE}")

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
if INDEX_HEADER_SIZE != 24:
    raise ImportError(f"index header must be 24 bytes, got {INDEX_HEADER_SIZE}")


def align_up(n: int, align: int = BLOB_ALIGNMENT) -> int:
    """Round n up to the nearest multiple of align."""
    return (n + align - 1) & ~(align - 1)
