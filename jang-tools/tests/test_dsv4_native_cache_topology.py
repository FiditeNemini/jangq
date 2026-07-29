from types import SimpleNamespace


def _fake_model(ratios, *, sliding_window=128):
    return SimpleNamespace(
        args=SimpleNamespace(sliding_window=sliding_window),
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(compress_ratio=ratio)
                )
                for ratio in ratios
            ]
        ),
    )


def test_dsv4_long_context_uses_swa_ring_for_zero_compression_layers(
    monkeypatch,
):
    from jang_tools.dsv4.mlx_model import DeepseekV4Cache, Model
    from mlx_lm.models.cache import RotatingKVCache

    monkeypatch.setenv("DSV4_LONG_CTX", "1")
    monkeypatch.setenv("DSV4_POOL_QUANT", "0")

    cache = Model.make_cache(_fake_model([0, 0, 4, 128, 4]))

    assert [type(layer).__name__ for layer in cache] == [
        "RotatingKVCache",
        "RotatingKVCache",
        "DeepseekV4Cache",
        "DeepseekV4Cache",
        "DeepseekV4Cache",
    ]
    assert all(
        layer.max_size == 128 and layer.keep == 0
        for layer in cache[:2]
        if isinstance(layer, RotatingKVCache)
    )
    assert all(
        isinstance(layer, DeepseekV4Cache)
        for layer in cache[2:]
    )
    assert [layer.compress_ratio for layer in cache[2:]] == [4, 128, 4]


def test_dsv4_legacy_short_mode_keeps_plain_kv_cache(monkeypatch):
    from jang_tools.dsv4.mlx_model import Model

    monkeypatch.setenv("DSV4_LONG_CTX", "0")
    monkeypatch.setenv("DSV4_POOL_QUANT", "0")

    cache = Model.make_cache(_fake_model([0, 4, 128]))

    assert [type(layer).__name__ for layer in cache] == [
        "KVCache",
        "KVCache",
        "KVCache",
    ]
