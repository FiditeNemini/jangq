"""Tests for JANG TurboQuant configuration and cache factory."""
import pytest
import mlx.core as mx

from jang_tools.turboquant.config import TurboQuantConfig, make_turboquant_cache


class TestTurboQuantConfig:
    """Test per-layer configuration."""

    def test_default_config(self):
        cfg = TurboQuantConfig()
        assert cfg.default_key_bits == 3
        assert cfg.default_value_bits == 3

    def test_layer_bits(self):
        """Critical layers should get more bits."""
        cfg = TurboQuantConfig(
            n_layers=56,
            critical_key_bits=4,
            critical_value_bits=4,
            default_key_bits=3,
            default_value_bits=3,
            critical_layers=[0, 1, 2, 53, 54, 55],
        )
        assert cfg.key_bits_for_layer(0) == 4    # critical
        assert cfg.key_bits_for_layer(1) == 4    # critical
        assert cfg.key_bits_for_layer(10) == 3   # default
        assert cfg.key_bits_for_layer(55) == 4   # critical

    def test_negative_layer_indices(self):
        """Negative indices should resolve from end."""
        cfg = TurboQuantConfig(
            n_layers=56,
            critical_layers=[0, 1, 2, -3, -2, -1],
        )
        assert cfg.key_bits_for_layer(53) == 4   # -3
        assert cfg.key_bits_for_layer(54) == 4   # -2
        assert cfg.key_bits_for_layer(55) == 4   # -1
        assert cfg.key_bits_for_layer(30) == 3   # middle

    def test_from_jang_config(self):
        """Should parse turboquant section from jang_config.json."""
        jang_cfg = {
            "turboquant": {
                "enabled": True,
                "default_key_bits": 3,
                "default_value_bits": 3,
                "critical_key_bits": 4,
                "critical_value_bits": 4,
                "critical_layers": [0, 1, 2, -3, -2, -1],
            }
        }
        cfg = TurboQuantConfig.from_jang_config(jang_cfg, n_layers=56)
        assert cfg is not None
        assert cfg.key_bits_for_layer(0) == 4
        assert cfg.key_bits_for_layer(30) == 3

    def test_disabled_returns_none(self):
        """No turboquant config -> returns None."""
        cfg = TurboQuantConfig.from_jang_config({}, n_layers=56)
        assert cfg is None

    def test_disabled_explicit(self):
        """enabled=false -> returns None."""
        cfg = TurboQuantConfig.from_jang_config(
            {"turboquant": {"enabled": False}}, n_layers=56
        )
        assert cfg is None


class TestMakeCache:
    """Test cache factory for hybrid models."""

    def test_make_cache_all_attention(self):
        """All-attention model: every layer gets TurboQuantKVCache."""
        cfg = TurboQuantConfig(n_layers=4)
        caches = make_turboquant_cache(
            cfg, n_layers=4,
            key_dims=[128]*4, value_dims=[128]*4,
            layer_types=["attention"]*4,
        )
        assert len(caches) == 4
        from jang_tools.turboquant.cache import TurboQuantKVCache
        assert all(isinstance(c, TurboQuantKVCache) for c in caches)

    def test_make_cache_hybrid(self):
        """Hybrid model: attention gets TurboQuant, SSM gets ArraysCache."""
        cfg = TurboQuantConfig(n_layers=4)
        caches = make_turboquant_cache(
            cfg, n_layers=4,
            key_dims=[128]*4, value_dims=[128]*4,
            layer_types=["attention", "ssm", "attention", "ssm"],
        )
        from jang_tools.turboquant.cache import TurboQuantKVCache
        assert isinstance(caches[0], TurboQuantKVCache)
        assert not isinstance(caches[1], TurboQuantKVCache)  # SSM
        assert isinstance(caches[2], TurboQuantKVCache)
        assert not isinstance(caches[3], TurboQuantKVCache)  # SSM

    def test_per_layer_bits(self):
        """Critical layers should get different bits than default."""
        cfg = TurboQuantConfig(
            n_layers=4,
            critical_layers=[0, 3],
            critical_key_bits=4,
            default_key_bits=3,
        )
        caches = make_turboquant_cache(
            cfg, n_layers=4,
            key_dims=[128]*4, value_dims=[128]*4,
            layer_types=["attention"]*4,
        )
        assert caches[0].key_bits == 4   # critical
        assert caches[1].key_bits == 3   # default
        assert caches[2].key_bits == 3   # default
        assert caches[3].key_bits == 4   # critical
