# TurboQuant — Final Status
**Author:** Jinho Jang (eric@jangq.ai)
**Date:** 2026-03-25
**Version:** JANG 2.3.0

---

## What It Is

First implementation of Google DeepMind's TurboQuant (ICLR 2026) on Apple Silicon.
Random Hadamard rotation + optimal Lloyd-Max codebooks + QJL residual correction.
JANG-exclusive: only activates for JANG models via `jang_config.json`.

## What It Does

1. **Zero-overhead generation** — TurboQuant cache runs at baseline KVCache speed during generation. No encode/decode on the hot path.
2. **5x compressed storage** — `compress_cache()` converts float KV to packed 3-bit format for serialization, prefix cache, disk paging.
3. **90% baseline post-compress generation** — After compress, new tokens generate at 90% of baseline speed via pre-allocated joined buffer.
4. **Automatic hybrid model detection** — Correctly identifies attention vs SSM vs MoE-MLP layers across all tested architectures.

## Models Tested

| Model | Params | Profile | Architecture | TQ Layers | Gen tok/s | Quality |
|-------|--------|---------|-------------|-----------|-----------|---------|
| **Mistral Small 4** | **119B** | **JANG_2L** | **MLA + 128-expert MoE** | **36/36** | **77.9** | **Correct** |
| Qwen3.5-35B-A3B | 35B | JANG_4S | GQA + GatedDeltaNet SSM | 10/40 | 109 | Correct |
| Qwen3.5-122B-A10B | 122B | JANG_3L | GQA + GatedDeltaNet SSM | 12/48 | 57 | Correct |
| Qwen3.5-122B-A10B | 122B | JANG_4K | GQA + GatedDeltaNet SSM | 12/48 | 57 | Correct |
| MiniMax M2.5 | ~120B | JANG_2L | GQA + 256-expert MoE | 62/62 | 53 | Correct |
| Nemotron Cascade 2 | 30B | JANG_4M | GQA + Mamba + 128-expert MoE | 6/52 | 140 | Correct |
| Nemotron Cascade 2 | 30B | JANG_2L | GQA + Mamba + 128-expert MoE | 6/52 | 131 | Correct |

**8 model configurations tested across 6 architectures. All produce correct output.**

## Compression Results (actual)

| Model | Context | Float Cache | Packed | Savings | Ratio |
|-------|---------|-----------|--------|---------|-------|
| **Mistral 4** | **1.9K** | **1,208 MB** | **244 MB** | **964 MB** | **4.9x** |
| **Mistral 4** | **32K (proj)** | **20.2 GB** | **4.1 GB** | **16.1 GB** | **4.9x** |
| Qwen3.5-35B | 32K | 655 MB | 135 MB | 520 MB | 4.9x |
| MiniMax M2.5 | 32K | 8,127 MB | 1,719 MB | 6,408 MB | 4.7x |

## Architecture Detection Matrix

| Pattern | Meaning | Cache | Detection |
|---------|---------|-------|-----------|
| `full_attention` in layer_types | Standard attention | TurboQuantKVCache | Qwen3.5 |
| `linear_attention` in layer_types | GatedDeltaNet SSM | ArraysCache | Qwen3.5 |
| `*` in hybrid_override_pattern | Attention layer | TurboQuantKVCache | Nemotron |
| `M` in hybrid_override_pattern | Mamba SSM | ArraysCache | Nemotron |
| `E` in hybrid_override_pattern | MoE MLP (no attn) | No cache entry | Nemotron |
| All layers attention | Standard model | TurboQuantKVCache | MiniMax, Llama |

## Code

```
jang_tools/turboquant/        (9 files, 1148 lines)
tests/test_turboquant_*.py    (8 files, 722 lines)
Total: 17 files, 1870 lines, 66 tests passing
```

## Known Limitations

1. **Mistral 4 (MLA)** — WORKING. 77.9 tok/s, perfect quality, 4.9x compression. 36 TQ layers. At 32K: saves 16.1 GB.
2. **Runtime memory savings** — After compress, the decoded buffer stays in memory for fast generation. Net runtime memory is float + packed (not smaller). Packed storage enables serialization/eviction savings.
3. **Nemotron .mlxstudio copy** — Has corrupted gate scales (all zeros) from early conversion. Use JANGQ-Library copies which work correctly.
4. **No VLM testing yet** — Should work (TQ only touches KV cache, not vision encoder) but untested.

## How to Enable

Add to model's `jang_config.json`:
```json
"turboquant": {
    "enabled": true,
    "default_key_bits": 3,
    "default_value_bits": 3,
    "critical_key_bits": 4,
    "critical_value_bits": 4,
    "critical_layers": [0, 1, 2, -3, -2, -1],
    "seed": 42
}
```

Load normally:
```python
from jang_tools.loader import load_jang_model
model, tokenizer = load_jang_model("path/to/model")
# TurboQuant activates automatically
```

Compress for storage:
```python
from jang_tools.turboquant import compress_cache
cache = model.make_cache()
# ... generate ...
compress_cache(cache)  # 5x packed storage
```

## Files Modified (from base JANG 2.1.5)

- `jang_tools/__init__.py` — Added TurboQuant exports, version 2.3.0
- `jang_tools/loader.py` — TurboQuant make_cache patch, gate dequant fix (.mixer.gate.), hybrid pattern detection, Nemotron fc1/fc2 rename
- `jang_tools/turboquant/` — New package (9 files)
- `pyproject.toml` — Version 2.3.0
- `tests/test_turboquant_*.py` — 8 test files, 66 tests
