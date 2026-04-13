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
