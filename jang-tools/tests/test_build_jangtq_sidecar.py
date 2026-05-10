import json

import numpy as np
from safetensors.numpy import save_file

from jang_tools import build_jangtq_sidecar as sidecar


def test_sidecar_scan_reads_tq_bits_by_shard_cache(tmp_path):
    model_dir = tmp_path / "bundle"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.tq_packed": np.zeros((4, 128), dtype=np.uint32),
        "model.layers.0.mlp.experts.0.gate_proj.tq_norms": np.ones((4,), dtype=np.float32),
        "model.layers.0.mlp.experts.0.gate_proj.tq_bits": np.array([1], dtype=np.int32),
        "model.layers.0.mlp.experts.0.down_proj.tq_packed": np.zeros((4, 96), dtype=np.uint32),
        "model.layers.0.mlp.experts.0.down_proj.tq_norms": np.ones((4,), dtype=np.float32),
        "model.layers.0.mlp.experts.0.down_proj.tq_bits": np.array([2], dtype=np.int32),
    }
    save_file(tensors, str(shard))
    weight_map = {key: shard.name for key in tensors}
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )

    loaded = sidecar._load_weight_map(model_dir)
    packed = sidecar._scan_tq_tensors(model_dir, loaded)
    by_shard = {}
    for fname, keys in sidecar._group_weight_map_by_shard(loaded).items():
        by_shard[fname] = {
            key: int(tensors[key][0])
            for key in keys
            if key.endswith(".tq_bits")
        }

    assert len(packed) == 2
    assert sidecar._read_tq_bits(
        model_dir,
        "model.layers.0.mlp.experts.0.gate_proj.tq_packed",
        loaded,
        by_shard,
    ) == 1
    assert sidecar._read_tq_bits(
        model_dir,
        "model.layers.0.mlp.experts.0.down_proj.tq_packed",
        loaded,
        by_shard,
    ) == 2
