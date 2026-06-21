import json

import numpy as np
from safetensors.numpy import save_file

from jang_tools.prune_n2_jang_experts import (
    _compute_keep_indices,
    _slice_tensor,
    _update_jang_metadata,
)


def test_compute_keep_indices_accepts_sanitized_router_keys(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    router_key = "language_model.model.layers.0.mlp.gate.weight"
    save_file(
        {
            router_key: np.array(
                [
                    [0.1, 0.0],
                    [3.0, 0.0],
                    [0.2, 0.0],
                    [2.0, 0.0],
                ],
                dtype=np.float16,
            )
        },
        str(shard),
    )

    keep = _compute_keep_indices(tmp_path, {router_key: shard.name}, keep_experts=2)

    assert keep[0].tolist() == [1, 3]


def test_slice_tensor_prunes_switch_and_router_axis_zero():
    keep = {0: np.array([1, 3], dtype=np.int64)}

    switch = _slice_tensor(
        "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
        np.arange(4 * 2).reshape(4, 2),
        keep,
    )
    router = _slice_tensor(
        "language_model.model.layers.0.mlp.gate.weight",
        np.arange(4 * 2).reshape(4, 2),
        keep,
    )

    assert switch.tolist() == [[2, 3], [6, 7]]
    assert router.tolist() == [[2, 3], [6, 7]]


def test_slice_tensor_prunes_jangtq_payloads_but_keeps_tq_bits():
    keep = {0: np.array([1, 3], dtype=np.int64)}

    packed = _slice_tensor(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq_packed",
        np.arange(4 * 2).reshape(4, 2),
        keep,
    )
    norms = _slice_tensor(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq_norms",
        np.arange(4 * 2).reshape(4, 2),
        keep,
    )
    bits = _slice_tensor(
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.tq_bits",
        np.array([2], dtype=np.uint8),
        keep,
    )

    assert packed.tolist() == [[2, 3], [6, 7]]
    assert norms.tolist() == [[2, 3], [6, 7]]
    assert bits.tolist() == [2]


def test_update_jang_metadata_rewrites_manifest_shapes(tmp_path):
    jang_path = tmp_path / "jang_config.json"
    jang_path.write_text(
        json.dumps(
            {
                "quantization": {
                    "tensor_quantization_manifest": {
                        "language_model.model.layers.0.mlp.switch_mlp.down_proj": {
                            "weight_key": "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight",
                            "scales_key": "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales",
                            "biases_key": "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases",
                            "weight_shape": [512, 4096, 64],
                            "scales_shape": [512, 4096, 8],
                            "biases_shape": [512, 4096, 8],
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    _update_jang_metadata(
        tmp_path,
        keep_experts=435,
        original_experts=512,
        shape_updates={
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight": [435, 4096, 64],
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.scales": [435, 4096, 8],
            "language_model.model.layers.0.mlp.switch_mlp.down_proj.biases": [435, 4096, 8],
        },
    )

    updated = json.loads(jang_path.read_text(encoding="utf-8"))
    entry = updated["quantization"]["tensor_quantization_manifest"][
        "language_model.model.layers.0.mlp.switch_mlp.down_proj"
    ]

    assert updated["expert_pruning"]["num_experts"] == 435
    assert updated["expert_pruning"]["dropped_experts_per_layer"] == 77
    assert entry["weight_shape"] == [435, 4096, 64]
    assert entry["scales_shape"] == [435, 4096, 8]
    assert entry["biases_shape"] == [435, 4096, 8]
