"""
TurboQuant — JANG-exclusive KV Cache Compression for Apple Silicon.

First implementation of Google DeepMind's TurboQuant (ICLR 2026) on MLX.
Random rotation + optimal scalar codebooks + QJL residual correction.

5.3x KV cache compression at 3-bit with zero speed overhead.
Only activates for JANG models (requires jang_config.json).

Usage:
    # Add to model's jang_config.json:
    # "turboquant": {"enabled": true, "default_key_bits": 3, ...}
    #
    # Then load normally:
    from jang_tools.loader import load_jang_model
    model, tokenizer = load_jang_model("path/to/jang/model")
    # TurboQuant activates automatically if config present.
    #
    # Compress cache on demand (for memory savings at long contexts):
    cache = model.make_cache()
    # ... generate tokens ...
    for c in cache:
        if hasattr(c, 'compress'):
            c.compress()  # 5.3x memory reduction
"""

from .cache import TurboQuantKVCache
from .config import TurboQuantConfig, make_turboquant_cache
from .pipeline import TurboQuantEncoder, encode_keys, decode_keys, encode_values, decode_values
from .generate import compress_cache, cache_memory_report

__all__ = [
    "TurboQuantKVCache",
    "TurboQuantConfig",
    "TurboQuantEncoder",
    "make_turboquant_cache",
    "encode_keys",
    "decode_keys",
    "encode_values",
    "decode_values",
    "compress_cache",
    "cache_memory_report",
]
