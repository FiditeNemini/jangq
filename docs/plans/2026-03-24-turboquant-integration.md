# TurboQuant KV Cache Compression — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Google DeepMind's TurboQuant (ICLR 2026) as a JANG-exclusive KV cache compression system for Apple Silicon, making JANG the first and only quantization system that optimally compresses both model weights AND KV cache.

**Architecture:** A standalone `jang_tools/turboquant/` package implementing full TurboQuant (random rotation + optimal scalar codebooks + QJL residual correction for keys). Integrates with JANG loader via `make_cache()` monkey-patch. Uses the rotate-query trick for O(1) per-step cost. JANG-gated: only activates for JANG models (requires `jang_config.json`). Pure MLX ops first, Metal kernel optimization later.

**Tech Stack:** MLX (mx.array, mx.fast), NumPy (codebook precomputation), SciPy (Beta distribution CDF/PDF for Lloyd-Max), pytest

**Primary target:** Mistral Small 4 (119B) — MLA architecture, 128 KV heads, 192-dim keys, 128-dim values, 56 layers. Cache at 4K context = 18 GB FP16.

---

## Research Documents (already written)

All research is at `/Users/eric/jang/research/turboquant/`:
- `00-RESEARCH-PLAN.md` — Master plan
- `01-RELATED-PAPERS.md` — TurboQuant, KIVI, QJL, GEAR, QuIP, QuIP#, PolarQuant, KVLinC, KVSplit
- `02-MLX-KV-CACHE-ANALYSIS.md` — Complete MLX cache reverse-engineering
- `03-KV-CACHE-MATH-FOUNDATIONS.md` — Mathematical foundations, formulas, error analysis

## Key Mathematical Facts

- Coordinates after random rotation on S^(d-1) follow Beta((d-1)/2, (d-1)/2) approx N(0, 1/d)
- Optimal codebook: Lloyd-Max quantizer on this distribution, precomputed per (d, b)
- MSE distortion: D_mse <= 2.72/4^b (within 2.7x of Shannon bound)
- Inner-product distortion (with QJL): D_prod <= sqrt(3)*pi^2*||y||^2/(d*4^b), UNBIASED
- Keys use TurboQuant_prod: (b-1)-bit MSE + 1-bit QJL residual
- Values use TurboQuant_mse: b-bit MSE
- Rotate-query trick: <Pi*q, Pi*k> = <q, k> (rotation preserves inner products)

## JANG-Gating Strategy

TurboQuant only activates when:
1. Model loaded via `load_jang_model()` (requires `jang_config.json`)
2. `turboquant.enabled = true` is set in jang_config
3. TurboQuant per-layer config stored in `jang_config.json` under `"turboquant"` key

Regular MLX models, oQ models, competitor quants get nothing — standard naive cache.

---

## Task 1: Hadamard Rotation Module

**Files:**
- Create: `jang-tools/jang_tools/turboquant/__init__.py`
- Create: `jang-tools/jang_tools/turboquant/rotation.py`
- Create: `jang-tools/tests/test_turboquant_rotation.py`

### Tests (8 tests):
- `test_roundtrip_identity` — rotate then inverse = original (atol 1e-5)
- `test_preserves_norm` — L2 norm unchanged (orthogonal transform)
- `test_preserves_inner_product` — <Pi*a, Pi*b> = <a, b>
- `test_output_shape_matches_input` — shape preserved
- `test_different_dimensions` — parametrized: 64, 128, 192, 256
- `test_spreads_outliers` — max coordinate drops after rotation

### Implementation:
- `generate_random_signs(dim, seed)` — random +/-1 vector for Randomized Hadamard
- `_hadamard_transform(x)` — O(d*log d) butterfly FFT-like transform on last dim
- `hadamard_rotate(x, signs)` — H * diag(signs) * x, with padding for non-power-of-2
- `hadamard_inverse(y, signs)` — inverse: signs * H * y (H is self-inverse)

Non-power-of-2 handling (for Mistral 4 key_dim=192): pad to 256, apply Hadamard, slice back to 192.

---

## Task 2: Optimal Scalar Codebook

**Files:**
- Create: `jang-tools/jang_tools/turboquant/codebook.py`
- Create: `jang-tools/tests/test_turboquant_codebook.py`

### Tests (7 tests):
- `test_codebook_size` — 2^b entries per codebook
- `test_codebook_sorted` — ascending order
- `test_codebook_symmetric` — symmetric around 0 (distribution is symmetric)
- `test_quantize_roundtrip_low_error` — MSE matches paper bounds
- `test_quantize_indices_valid` — all indices in [0, 2^b - 1]
- `test_codebook_different_dims` — parametrized: 64, 128, 192, 256

### Implementation:
- `_beta_pdf(x, d)` — PDF of coordinate on unit sphere S^(d-1)
- `compute_codebook(dim, bits)` — Lloyd-Max algorithm on Beta distribution (cached)
- `quantize_scalar(x, codebook)` — nearest codebook entry per element
- `dequantize_scalar(indices, codebook)` — lookup via mx.take()

---

## Task 3: QJL Transform

**Files:**
- Create: `jang-tools/jang_tools/turboquant/qjl.py`
- Create: `jang-tools/tests/test_turboquant_qjl.py`

### Tests (5 tests):
- `test_encode_output_is_signs` — produces +/-1 values only
- `test_encode_returns_correct_norm` — stores L2 norm of input
- `test_unbiased_inner_product` — E[estimate] = true IP (5000 trials, <15% relative error on mean)
- `test_projection_shape` — S is (dim, dim)
- `test_different_seeds_different_projections`

### Implementation:
- `generate_qjl_projection(dim, seed)` — random Gaussian matrix S
- `qjl_encode(x, S)` — returns (sign(S*x), ||x||_2)
- `qjl_decode(signs, norm, S)` — sqrt(pi/2)/d * norm * S^T * signs
- `qjl_inner_product(query, key_signs, key_norm, S)` — sqrt(pi/2)/d * ||k|| * <S*q, signs>

---

## Task 4: Encode/Decode Pipeline

**Files:**
- Create: `jang-tools/jang_tools/turboquant/pipeline.py`
- Create: `jang-tools/tests/test_turboquant_pipeline.py`

### Tests (8 tests):
- `test_key_roundtrip_shape` — shape preserved
- `test_key_mse_within_bound` — MSE < 0.1 at 3-bit
- `test_key_inner_product_unbiased` — mean error < 0.05 over 200 vectors
- `test_value_roundtrip_shape` — shape preserved
- `test_value_mse_within_bound` — MSE < 0.08 at 3-bit
- `test_encoder_init` — no errors
- `test_encoder_different_dims` — parametrized: 64, 128, 192

### Implementation:
- `TurboQuantEncoder` dataclass: precomputes rotation signs, codebooks, QJL projection
- `EncodedKeys` NamedTuple: mse_indices, qjl_signs, residual_norms, vector_norms
- `EncodedValues` NamedTuple: mse_indices, vector_norms
- `encode_keys(keys, enc)` — rotate, MSE quantize (b-1 bits), compute residual, QJL encode
- `decode_keys(encoded, enc)` — MSE dequant + QJL dequant, inverse rotate, rescale
- `encode_values(values, enc)` — rotate, MSE quantize (b bits)
- `decode_values(encoded, enc)` — MSE dequant, inverse rotate, rescale

Key pipeline: normalize to unit sphere -> rotate -> quantize -> QJL on residual -> store
Value pipeline: normalize to unit sphere -> rotate -> quantize -> store

---

## Task 5: TurboQuantKVCache Class

**Files:**
- Create: `jang-tools/jang_tools/turboquant/cache.py`
- Create: `jang-tools/tests/test_turboquant_cache.py`

### Tests (9 tests):
- `test_empty_on_init`
- `test_update_single_token` — insert 1 token, verify offset and shape
- `test_update_multiple_tokens` — prefill 8, decode 1, verify offset=9 and shapes
- `test_mistral4_dimensions` — keys=192, values=128, 128 KV heads
- `test_reconstruction_quality` — relative MSE < 50% at 4-bit
- `test_has_offset_property`
- `test_is_trimmable`
- `test_trim` — trim reduces offset correctly

### Implementation:
- `TurboQuantKVCache` class with:
  - `__init__(key_dim, value_dim, key_bits, value_bits, seed)`
  - `update_and_fetch(keys, values)` — encode new tokens, decode all, return full cache
  - `make_mask(N, ...)` — delegates to mlx_lm's create_attention_mask
  - `empty()`, `is_trimmable()`, `trim(n)`, `size()`
  - `state`/`meta_state` properties for serialization
  - Stores encoded chunks as lists, concatenates on fetch

---

## Task 6: Custom SDPA

**Files:**
- Create: `jang-tools/jang_tools/turboquant/attention.py`
- Create: `jang-tools/tests/test_turboquant_attention.py`

### Tests (3 tests):
- `test_output_shape` — correct shape with GQA
- `test_attention_not_nan` — no NaN/Inf in output
- `test_close_to_exact_attention` — cosine similarity > 0.7 with exact FP32 attention at 4-bit

### Implementation:
- `turboquant_sdpa(queries, cache, scale, mask)` — decodes keys/values from cache, runs standard attention
- First version: full decode then standard matmul (correct but not optimized)
- Future: rotate-query trick to avoid per-token inverse rotation

---

## Task 7: JANG Loader Integration

**Files:**
- Modify: `jang-tools/jang_tools/loader.py` (add TurboQuant cache patching)
- Create: `jang-tools/jang_tools/turboquant/config.py`
- Create: `jang-tools/tests/test_turboquant_loader.py`

### Tests (6 tests):
- `test_default_config` — sensible defaults
- `test_layer_bits` — critical layers get more bits
- `test_from_jang_config` — parses turboquant section
- `test_disabled_returns_none` — no config = no TurboQuant
- `test_make_cache_all_attention` — all attention layers get TurboQuantKVCache
- `test_make_cache_hybrid` — SSM layers get ArraysCache, attention gets TurboQuant

### Implementation:
- `TurboQuantConfig` dataclass with `from_jang_config()` classmethod
- `make_turboquant_cache()` — per-layer cache factory
- Loader patch: after model load, check jang_config for turboquant section, monkey-patch `model.make_cache()`

### Gating mechanism:
- No `jang_config.json` = not a JANG model = no TurboQuant
- No `turboquant.enabled` in config = TurboQuant not configured = no TurboQuant
- Both present = monkey-patch make_cache with TurboQuant caches

---

## Task 8: End-to-End Test (Mistral 4 on Mac Studio)

**Files:**
- Create: `jang-tools/tests/test_turboquant_e2e.py`

### Tests (2 tests, require model on disk):
- `test_generate_short` — load Mistral 4 JANG_2L, generate response, verify coherent
- `test_cache_memory_savings` — verify TurboQuantKVCache instances created

### Setup:
1. Add `turboquant` section to Mistral 4 JANG_2L's `jang_config.json`
2. Run on Mac Studio where model exists

---

## Task 9: Benchmarks and Validation

Run on Mac Studio with Mistral Small 4 JANG_2L:

| Benchmark | What | Success Criteria |
|-----------|------|-----------------|
| Perplexity | WikiText2 sample | Within 5% of FP16 cache at 3-bit |
| MMLU | 200 questions | Within 1% of FP16 cache |
| Speed | Generate 200 tokens | > 80% of FP16 cache tok/s |
| Memory | Cache size at 4K context | < 40% of FP16 |
| Long context | 8K, 16K, 32K | No crashes, coherent output |
| NIAH | Needle retrieval | 100% at 4-bit, >95% at 3-bit |

Document all results in `research/turboquant/05-EXPERIMENT-LOG.md`.

---

## Task 10: Production Polish

- Clean up `__init__.py` exports
- Add scipy to pyproject.toml dependencies
- Version bump
- Final commit

---

## Dependency Chain

```
Task 1 (rotation) -> Task 2 (codebook) -> Task 3 (QJL)
                                               |
                                               v
                                    Task 4 (pipeline)
                                               |
                                               v
                                    Task 5 (cache class)
                                               |
                                               v
                                    Task 6 (SDPA)
                                               |
                                               v
                                    Task 7 (loader integration)
                                               |
                                               v
                                    Task 8 (E2E test)
                                               |
                                               v
                                    Task 9 (benchmarks)
                                               |
                                               v
                                    Task 10 (polish)
```

## Total: 48 automated tests across 8 test files
