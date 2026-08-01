from types import SimpleNamespace

import numpy as np
import pytest


mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


@pytest.fixture(autouse=True)
def _run_dsv4_moe_contract_on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


class _FixedGate(nn.Module):
    def __init__(self, indices, scores):
        super().__init__()
        self.indices = indices
        self.scores = scores

    def __call__(self, _x, input_ids=None):
        del input_ids
        return self.indices, self.scores


class _FixedSharedExpert(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def __call__(self, _x):
        return self.output


def _official_limited_swiglu(gate, up, limit):
    gate = gate.astype(mx.float32)
    up = up.astype(mx.float32)
    up = mx.clip(up, a_min=-limit, a_max=limit)
    gate = mx.clip(gate, a_min=None, a_max=limit)
    return nn.silu(gate) * up


def _make_switch_glu(*, quantized, rng):
    from mlx_lm.models.switch_layers import SwitchGLU

    switch = SwitchGLU(32, 32, 3, bias=False)
    weight_dtype = np.float16 if quantized else np.float32
    for name in ("gate_proj", "up_proj", "down_proj"):
        projection = getattr(switch, name)
        projection.weight = mx.array(
            rng.normal(0.0, 0.35, size=projection.weight.shape).astype(
                weight_dtype
            )
        )
        if quantized:
            projection = projection.to_quantized(group_size=32, bits=4)
            projection.scales = projection.scales.astype(mx.float16)
            if projection.biases is not None:
                projection.biases = projection.biases.astype(mx.float16)
            setattr(switch, name, projection)
    return switch


@pytest.mark.parametrize("quantized", [False, True])
def test_dsv4_moe_matches_official_weight_before_down_formula(quantized):
    from jang_tools.dsv4.mlx_model import MoE

    rng = np.random.default_rng(0)
    switch = _make_switch_glu(quantized=quantized, rng=rng)
    dtype = mx.float16 if quantized else mx.float32
    np_dtype = np.float16 if quantized else np.float32
    x = mx.array(rng.normal(0.0, 1.2, size=(2, 20, 32)).astype(np_dtype))
    indices = mx.array(
        rng.integers(0, 3, size=(2, 20, 2), dtype=np.int32)
    )
    score_values = rng.uniform(0.03, 0.97, size=(2, 20, 2)).astype(np.float32)
    score_values /= score_values.sum(axis=-1, keepdims=True)
    scores = mx.array(score_values)
    shared = mx.array(rng.normal(0.0, 0.3, size=x.shape).astype(np_dtype))

    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.args = SimpleNamespace(swiglu_limit=10.0)
    moe.gate = _FixedGate(indices, scores)
    moe.switch_mlp = switch
    moe.shared_experts = _FixedSharedExpert(shared)

    actual = moe(x)

    selected = indices.astype(mx.uint32)
    expanded = mx.expand_dims(x, (-2, -3))
    up = switch.up_proj(expanded, selected, sorted_indices=False)
    gate = switch.gate_proj(expanded, selected, sorted_indices=False)
    activated = _official_limited_swiglu(gate, up, 10.0)
    activated = (activated * scores[..., None, None]).astype(dtype)
    routed = switch.down_proj(
        activated,
        selected,
        sorted_indices=False,
    ).squeeze(-2)
    if quantized:
        assert routed.dtype == mx.float16
        assert shared.dtype == mx.float16
        assert scores.dtype == mx.float32
    expected = (
        routed.astype(mx.float32).sum(axis=-2)
        + shared.astype(mx.float32)
    ).astype(dtype)

    # Negative control for the former implementation: F16 SwiGLU/down first,
    # route weighting afterward, then a cast before adding the shared expert.
    old_routed = switch(x, selected)
    old = (
        (old_routed * scores[..., None])
        .sum(axis=-2)
        .astype(old_routed.dtype)
        + shared
    )

    mx.eval(actual, expected, old)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.dtype == dtype
    assert np.any(np.asarray(old) != np.asarray(expected))


def test_dsv4_moe_accumulates_f16_routed_and_shared_outputs_in_fp32():
    from jang_tools.dsv4.mlx_model import _dsv4_accumulate_moe

    routed = mx.array([[[[2048.0], [1.0]]]], dtype=mx.float16)
    shared = mx.array([[[1.0]]], dtype=mx.float16)

    actual = _dsv4_accumulate_moe(
        routed,
        shared,
        (1, 1, 1),
        mx.float16,
    )
    mx.eval(actual)

    assert actual.dtype == mx.float16
    np.testing.assert_array_equal(np.asarray(actual), [[[2050.0]]])


def test_dsv4_router_softplus_is_finite_for_large_normal_and_hash_logits():
    from jang_tools.dsv4.mlx_model import Gate, sqrtsoftplus_select

    logits = mx.array([[[90.0, 100.0]]], dtype=mx.float32)
    indices, normal_scores = sqrtsoftplus_select(
        logits,
        mx.zeros((2,), dtype=mx.float32),
        top_k=2,
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
    )

    args = SimpleNamespace(
        hidden_size=1,
        n_routed_experts=2,
        num_experts_per_tok=2,
        num_hash_layers=1,
        vocab_size=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )
    hash_gate = Gate(args, layer_id=0)
    hash_gate.weight = mx.array([[90.0], [100.0]], dtype=mx.float32)
    hash_gate.tid2eid = mx.array([[0, 1]], dtype=mx.int32)
    hash_indices, hash_scores = hash_gate(
        mx.ones((1, 1, 1), dtype=mx.float32),
        input_ids=mx.zeros((1, 1), dtype=mx.int32),
    )

    mx.eval(indices, normal_scores, hash_indices, hash_scores)
    expected = np.sqrt(np.array([90.0, 100.0], dtype=np.float32))
    expected /= expected.sum()
    assert np.isfinite(np.asarray(normal_scores)).all()
    assert np.isfinite(np.asarray(hash_scores)).all()
    np.testing.assert_allclose(
        np.sort(np.asarray(normal_scores).reshape(-1)),
        np.sort(expected),
        rtol=1e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(hash_scores).reshape(-1),
        expected,
        rtol=1e-6,
        atol=0.0,
    )
