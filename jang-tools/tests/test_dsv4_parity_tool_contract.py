import pytest

from jang_tools.dsv4.verify_bf16_identical import _compress_ratio_for_layer, run
from jang_tools.dsv4.verify_mlx_vs_torch import _window_size_for_mask


def test_dsv4_parity_probe_detects_compressed_layers_from_config():
    cfg = {
        "num_hidden_layers": 43,
        "compress_ratios": [0, 0, 4, 128, 4],
    }

    assert _compress_ratio_for_layer(cfg, 0) == 0
    assert _compress_ratio_for_layer(cfg, 1) == 0
    assert _compress_ratio_for_layer(cfg, 2) == 4
    assert _compress_ratio_for_layer(cfg, 3) == 128


def test_dsv4_parity_probe_refuses_compressed_layers_before_indexing(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"num_hidden_layers": 43, "compress_ratios": [0, 0, 4]}'
    )

    with pytest.raises(SystemExit) as exc:
        run(tmp_path, 2)

    assert "compress_ratio=4" in str(exc.value)
    assert "CSA/HSA compressor/indexer sparse attention" in str(exc.value)


def test_dsv4_mlx_vs_torch_mask_window_comes_from_mlx_layer_args():
    class Args:
        sliding_window = 128

    class Layer:
        args = Args()

    assert _window_size_for_mask(Layer(), fallback=256) == 128


def test_dsv4_mlx_vs_torch_mask_window_falls_back_to_torch_config():
    class Layer:
        pass

    assert _window_size_for_mask(Layer(), fallback=256) == 256
