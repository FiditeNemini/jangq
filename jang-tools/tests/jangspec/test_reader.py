"""Round-trip tests: build a bundle, reload it, verify bytes match source."""

from pathlib import Path

import numpy as np
from safetensors import safe_open

from jang_tools.jangspec.builder import JangSpecBuilder
from jang_tools.jangspec.reader import JangSpecReader


def test_reader_matches_source_bytes(jangspec_fixture_model: Path, tmp_path: Path):
    out = tmp_path / "fixture.jangspec"
    # JangSpecReader reads the per-blob streaming layout — opt in via write_streaming.
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out, write_streaming=True).build()

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
    # JangSpecReader reads the per-blob streaming layout — opt in via write_streaming.
    JangSpecBuilder(source_dir=jangspec_fixture_model, out_dir=out, write_streaming=True).build()

    reader = JangSpecReader(out)
    last_layer = reader.n_layers - 1
    last_expert = reader.n_experts_per_layer - 1
    unpacked = reader.load_expert(layer_idx=last_layer, expert_id=last_expert)
    assert unpacked.layer_idx == last_layer
    assert unpacked.expert_id == last_expert
    assert unpacked.tensors.gate[0].size > 0
