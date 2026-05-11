"""Unit tests for `jang_tools.topk_override`.

Exercises the model-walk + attribute-patch logic against synthetic
modules — no real model load needed."""
import os
import pytest

from jang_tools.topk_override import apply_topk_override, topk_override_from_env


class _FakeRouter:
    """Mimics a MoE router with a `top_k` int attribute."""

    def __init__(self, top_k: int):
        self.top_k = top_k


class _FakeMoE:
    """Mimics an outer MoE container with `num_experts_per_tok`."""

    def __init__(self, k: int):
        self.num_experts_per_tok = k
        self.gate = _FakeRouter(k)


class _FakeLayer:
    def __init__(self, k: int):
        self.mlp = _FakeMoE(k)


class _FakeModel:
    """Minimal `named_modules`-compatible model."""

    def __init__(self, num_layers: int, k: int):
        self.layers = [_FakeLayer(k) for _ in range(num_layers)]

    def named_modules(self):
        for i, layer in enumerate(self.layers):
            yield (f"layers.{i}", layer)
            yield (f"layers.{i}.mlp", layer.mlp)
            yield (f"layers.{i}.mlp.gate", layer.mlp.gate)


class _ConfigBackedRouter:
    """Mimics routers like Gemma 4 that read active K from config.top_k_experts."""

    def __init__(self, k: int):
        self.config = type("Config", (), {"top_k_experts": k})()


class _ConfigBackedModel:
    def __init__(self, k: int):
        self.router = _ConfigBackedRouter(k)

    def named_modules(self):
        yield ("", self)
        yield ("router", self.router)


def test_override_patches_router_and_outer():
    model = _FakeModel(num_layers=3, k=8)
    n = apply_topk_override(model, 4)
    # 3 layers × 2 attrs (router.top_k + mlp.num_experts_per_tok) = 6
    assert n == 6
    for layer in model.layers:
        assert layer.mlp.gate.top_k == 4
        assert layer.mlp.num_experts_per_tok == 4


def test_override_no_op_when_already_at_k():
    model = _FakeModel(num_layers=3, k=4)
    n = apply_topk_override(model, 4)
    # Same K → counter does not increment (paths visited but no change)
    assert n == 0
    for layer in model.layers:
        assert layer.mlp.gate.top_k == 4


def test_override_can_restore_original_k_but_not_increase_above_it():
    model = _FakeModel(num_layers=2, k=8)

    assert apply_topk_override(model, 4) == 4
    assert apply_topk_override(model, 8) == 4
    for layer in model.layers:
        assert layer.mlp.gate.top_k == 8
        assert layer.mlp.num_experts_per_tok == 8

    with pytest.raises(ValueError, match="refuses to increase"):
        apply_topk_override(model, 9)
    for layer in model.layers:
        assert layer.mlp.gate.top_k == 8
        assert layer.mlp.num_experts_per_tok == 8


def test_override_recognizes_config_backed_top_k_experts():
    model = _ConfigBackedModel(k=8)

    assert apply_topk_override(model, 4) == 1
    assert model.router.config.top_k_experts == 4
    assert apply_topk_override(model, 8) == 1
    assert model.router.config.top_k_experts == 8

    with pytest.raises(ValueError, match="refuses to increase"):
        apply_topk_override(model, 9)
    assert model.router.config.top_k_experts == 8


def test_override_recognizes_top1_moe_router_topk_and_refuses_increase():
    """ZAYA-style top-1 routers should be recognized as trained K=1.

    A K=4 override must fail loudly instead of silently no-oping, otherwise an
    app setting can look active while preserving a different trained runtime.
    """

    class _ZayaRouter:
        moe_router_topk = 1

    class _ZayaModel:
        def __init__(self):
            self.router = _ZayaRouter()

        def named_modules(self):
            yield ("", self)
            yield ("router", self.router)

    model = _ZayaModel()

    assert apply_topk_override(model, 1) == 0
    with pytest.raises(ValueError, match="refuses to increase"):
        apply_topk_override(model, 4)
    assert model.router.moe_router_topk == 1


def test_override_none_is_no_op():
    model = _FakeModel(num_layers=3, k=8)
    assert apply_topk_override(model, None) == 0
    for layer in model.layers:
        assert layer.mlp.gate.top_k == 8


def test_override_rejects_non_positive():
    model = _FakeModel(num_layers=1, k=8)
    with pytest.raises(ValueError):
        apply_topk_override(model, 0)
    with pytest.raises(ValueError):
        apply_topk_override(model, -1)
    with pytest.raises(ValueError):
        apply_topk_override(model, "4")  # type: ignore[arg-type]


def test_override_skips_top1_models_silently():
    """Top-1 MoE families (ZAYA) have no `top_k` attribute. Override must not
    raise or patch anything; it just returns 0."""

    class _ZayaRouter:
        # ZAYA's router uses argmax, no top_k
        balancing_biases = None

    class _ZayaModel:
        def __init__(self):
            self.router = _ZayaRouter()

        def named_modules(self):
            yield ("", self)
            yield ("router", self.router)

    n = apply_topk_override(_ZayaModel(), 4)
    assert n == 0


def test_env_var_parses(monkeypatch):
    monkeypatch.delenv("JANGTQ_TOPK_OVERRIDE", raising=False)
    assert topk_override_from_env() is None
    monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "")
    assert topk_override_from_env() is None
    monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "0")
    assert topk_override_from_env() is None
    monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "4")
    assert topk_override_from_env() == 4
    monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", " 6 ")
    assert topk_override_from_env() == 6
    monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "abc")
    with pytest.raises(ValueError):
        topk_override_from_env()
    monkeypatch.setenv("JANGTQ_TOPK_OVERRIDE", "-2")
    with pytest.raises(ValueError):
        topk_override_from_env()
