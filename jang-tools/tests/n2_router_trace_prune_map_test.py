import json

from jang_tools.build_n2_prune_map_from_router_trace import build_prune_map
from jang_tools.convert import _load_expert_prune_map


def test_build_prune_map_from_layer_records(tmp_path):
    trace = {
        "artifact": "unpruned",
        "layers": {
            "0": {
                "records": [
                    {"indices": [4, 2, 7], "scores": [0.7, 0.2, 0.1]},
                    {"indices": [4, 9, 2], "scores": [0.6, 0.3, 0.1]},
                ]
            },
            "1": {
                "records": [
                    {"selected_experts": [[3, 1]], "selected_scores": [[0.8, 0.2]]},
                ]
            },
        },
    }

    prune_map = build_prune_map(trace, keep_experts=4, num_experts=10)
    out = tmp_path / "map.json"
    out.write_text(json.dumps(prune_map), encoding="utf-8")

    loaded = _load_expert_prune_map(out, keep_experts=4)

    assert loaded[0].tolist() == [4, 2, 9, 7]
    assert loaded[1].tolist() == [3, 1, 0, 2]


def test_build_prune_map_from_flat_records():
    trace = {
        "records": [
            {"layer": 0, "expert_indices": [1, 2], "weights": [0.4, 0.6]},
            {"layer": 0, "expert_indices": [2, 3], "weights": [0.9, 0.1]},
        ]
    }

    prune_map = build_prune_map(trace, keep_experts=3, num_experts=5)

    experts = prune_map["layers"]["0"]["keep_by_probability_coverage"]["3"]["experts"]
    assert experts == [2, 1, 3]
