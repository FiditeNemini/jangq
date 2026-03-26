"""
TurboQuant per-layer configuration.

Mirrors JANG's CRITICAL/COMPRESS philosophy for KV cache bits.
Config is stored in jang_config.json -- only JANG models have this.
No jang_config = no TurboQuant. This is the gating mechanism.
"""

from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx


@dataclass
class TurboQuantConfig:
    """Per-layer TurboQuant configuration."""
    n_layers: int = 32
    default_key_bits: int = 3
    default_value_bits: int = 3
    critical_key_bits: int = 4
    critical_value_bits: int = 4
    critical_layers: list[int] = field(default_factory=lambda: [0, 1, 2])
    sink_tokens: int = 4
    seed: int = 42

    def key_bits_for_layer(self, layer_idx: int) -> int:
        resolved = [l if l >= 0 else self.n_layers + l for l in self.critical_layers]
        return self.critical_key_bits if layer_idx in resolved else self.default_key_bits

    def value_bits_for_layer(self, layer_idx: int) -> int:
        resolved = [l if l >= 0 else self.n_layers + l for l in self.critical_layers]
        return self.critical_value_bits if layer_idx in resolved else self.default_value_bits

    @classmethod
    def from_jang_config(cls, jang_cfg: dict, n_layers: int) -> Optional["TurboQuantConfig"]:
        tq = jang_cfg.get("turboquant")
        if not tq or not tq.get("enabled", False):
            return None
        return cls(
            n_layers=n_layers,
            default_key_bits=tq.get("default_key_bits", 3),
            default_value_bits=tq.get("default_value_bits", 3),
            critical_key_bits=tq.get("critical_key_bits", 4),
            critical_value_bits=tq.get("critical_value_bits", 4),
            critical_layers=tq.get("critical_layers", [0, 1, 2, -3, -2, -1]),
            sink_tokens=tq.get("sink_tokens", 4),
            seed=tq.get("seed", 42),
        )


def make_turboquant_cache(
    config: TurboQuantConfig,
    n_layers: int,
    key_dims: list[int],
    value_dims: list[int],
    layer_types: list[str],
) -> list:
    """Create per-layer cache list: TurboQuantKVCache for attention, ArraysCache for SSM.

    Args:
        config: TurboQuant configuration.
        n_layers: Total number of layers.
        key_dims: Key dimension per layer.
        value_dims: Value dimension per layer.
        layer_types: "attention" or "ssm" per layer.

    Returns:
        List of cache objects, one per layer.
    """
    from .cache import TurboQuantKVCache

    # Try to import ArraysCache for SSM layers
    try:
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        ArraysCache = None

    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        KVCache = None

    caches = []
    for i in range(n_layers):
        if layer_types[i] == "attention":
            caches.append(TurboQuantKVCache(
                key_dim=key_dims[i],
                value_dim=value_dims[i],
                key_bits=config.key_bits_for_layer(i),
                value_bits=config.value_bits_for_layer(i),
                seed=config.seed + i,
                sink_tokens=config.sink_tokens,
            ))
        else:
            # SSM layer: use ArraysCache or KVCache as placeholder
            if ArraysCache is not None:
                caches.append(ArraysCache(size=2))
            elif KVCache is not None:
                caches.append(KVCache())
            else:
                raise ImportError("Neither ArraysCache nor KVCache available from mlx_lm")
    return caches
