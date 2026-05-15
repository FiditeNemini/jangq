import json

import numpy as np
from safetensors.numpy import save_file

from jang_tools.format.writer import write_jang_v2_model


def test_jang_v2_index_total_size_includes_preflushed_shards(tmp_path):
    preflushed = tmp_path / "model-00001-of-NNNNN.safetensors"
    save_file(
        {"pre.weight": np.zeros((2, 2), dtype=np.float16)},
        str(preflushed),
        metadata={"format": "mlx"},
    )

    write_jang_v2_model(
        output_dir=tmp_path,
        tensors={"tail.weight": np.zeros((4,), dtype=np.uint32)},
        model_config={},
        jang_config={"quantization": {"bit_widths_used": [4], "block_size": 64}},
        preflushed_map={"pre.weight": preflushed.name},
    )

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    shard_names = set(index["weight_map"].values())
    actual_size = sum((tmp_path / name).stat().st_size for name in shard_names)

    assert index["metadata"]["total_size"] == actual_size
    assert index["weight_map"]["pre.weight"] == "model-00001-of-00002.safetensors"
    assert index["weight_map"]["tail.weight"] == "model-00002-of-00002.safetensors"
