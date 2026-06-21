from __future__ import annotations

from jang_tools.mimo_v2.router_trace import RouteAccumulator


def test_route_accumulator_ranks_probability_before_hit_count() -> None:
    acc = RouteAccumulator(num_experts=8)

    acc.add(layer=2, indices=[[1, 3], [1, 4]], weights=[[0.2, 0.7], [0.2, 0.1]])
    acc.add(layer=2, indices=[[5, 1]], weights=[[0.65, 0.2]])

    trace = acc.to_trace(source="fixture", prompts=["p0"])

    layer = trace["layers"]["2"]
    assert layer["token_count"] == 3
    assert layer["observed_experts"] == 4
    assert layer["prob_rank"][:4] == [3, 5, 1, 4]
    assert layer["hit_rank"][:4] == [1, 3, 5, 4]
    assert trace["records"][0] == {
        "layer": 2,
        "indices": [1, 3],
        "weights": [0.2, 0.7],
    }
