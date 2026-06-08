import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn


class _SwitchMLP(nn.Module):
    def __init__(self, hidden=64, intermediate=32):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.switch_mlp = _SwitchMLP()


class _LanguageInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Layer(), _Layer()]


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _LanguageInner()


class _MiMoVLMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=48))
        self.visual = SimpleNamespace(config=SimpleNamespace(depth=28))
        self.language_model = _LanguageModel()


def test_load_jangtq_vlm_accepts_mimo_visual_without_vision_tower(tmp_path, monkeypatch):
    from jang_tools import load_jangtq
    from jang_tools import load_jangtq_vlm

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mimo_v2",
                "text_config": {"num_hidden_layers": 48},
                "vision_config": {"depth": 28},
            }
        )
    )
    (tmp_path / "jang_config.json").write_text(
        json.dumps({"format": "jangtq", "mxtq_seed": 42, "mxtq_bits": {}})
    )

    model = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=48)),
        visual=SimpleNamespace(config=SimpleNamespace(depth=28)),
        language_model=SimpleNamespace(),
    )
    processor = object()
    model_config = object()
    hydrate_calls = []

    monkeypatch.setattr(
        load_jangtq_vlm,
        "_mlx_vlm_skeleton",
        lambda model_path: (model, processor, object(), model_config),
    )
    monkeypatch.setattr(
        load_jangtq,
        "_hydrate_jangtq_model",
        lambda **kwargs: hydrate_calls.append(kwargs),
    )

    loaded_model, loaded_processor = load_jangtq_vlm.load_jangtq_vlm_model(tmp_path)

    assert loaded_model is model
    assert loaded_processor is processor
    assert hydrate_calls
    assert hydrate_calls[0]["model"] is model
    assert hydrate_calls[0]["model_config"] is model_config


def test_jangtq_hydrate_resolves_mimo_vlm_text_decoder_path(tmp_path, monkeypatch):
    from jang_tools import load_jangtq
    from jang_tools.turboquant.tq_kernel import TurboQuantSwitchLinear

    bits = 2
    hidden = 64
    intermediate = 32
    vals_per_u32 = 32 // bits
    weights = {
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_packed": mx.zeros(
            (2, intermediate, hidden // vals_per_u32), dtype=mx.uint32
        ),
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_norms": mx.zeros(
            (2, intermediate), dtype=mx.float16
        ),
        "model.layers.1.mlp.switch_mlp.gate_proj.tq_bits": mx.array(
            [bits], dtype=mx.uint8
        ),
    }
    model = _MiMoVLMModel()
    shard = tmp_path / "model-00001.safetensors"
    shard.write_bytes(b"not-used-mx-load-is-monkeypatched")

    monkeypatch.setattr(load_jangtq.mx, "load", lambda path: weights)

    load_jangtq._hydrate_jangtq_model(
        model=model,
        model_path=tmp_path,
        mxtq_seed=42,
        mxtq_bits_map={},
        model_config={"model_type": "mimo_v2"},
        skip_params_eval=True,
    )

    hydrated = model.language_model.model.layers[1].mlp.switch_mlp.gate_proj
    assert isinstance(hydrated, TurboQuantSwitchLinear)
    assert hydrated.packed.shape == (2, intermediate, hidden // vals_per_u32)
