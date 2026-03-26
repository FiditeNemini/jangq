# TurboQuant Phase 2 — Complete Production Implementation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make TurboQuant fully production-ready for all use cases: fast compressed generation at long context, Mistral 4 MLA support, Nemotron gate fix, sink token preservation, vmlx compatibility, and proper benchmarking.

**Architecture:** The core bottleneck is `_get_full_cache()` which decodes ALL compressed tokens every step (O(n*d^2)). Fix: cache the decoded compressed region as a read-only float buffer. Compress once → decode once → buffer persists. New tokens append to a separate float window. Attention reads the concatenation of (decoded_buffer, float_window) — both are float arrays, standard SDPA at full Metal speed.

**Tech Stack:** MLX, mlx_lm, numpy, pytest. Mac Studio M3 Ultra for E2E tests.

**Current state:** 9 source files, 8 test files, 59 tests passing, 1870 lines. Compression 5x, speed baseline, quality perfect. But post-compress generation is slow (decodes every step).

---

## CRITICAL BUG: Post-Compress Generation Speed

### The Problem
After `compress()`, every call to `update_and_fetch()` hits `_get_full_cache()` which calls `decode_keys()` and `decode_values()` — O(n * d^2) for QJL matrix multiply on ALL compressed tokens. At 10K compressed tokens: ~100ms per step per layer. With 62 layers (MiniMax): 6.2 seconds per token. Unusable.

### The Fix
Cache the decoded compressed region as a persistent float buffer. Only decode ONCE after `compress()`. Subsequent `update_and_fetch()` reads from the cached decoded buffer + appends new tokens to the float window.

```
compress() called:
  1. Encode [0..n) → compressed storage (5x smaller)
  2. Decode compressed → _decoded_k_buffer, _decoded_v_buffer (float, read-only)
  3. Shrink float window to [n..offset)
  4. Free original float [0..n) — memory reclaimed

update_and_fetch() after compress:
  1. Write new token to float window (fast, in-place)
  2. Return concat(_decoded_k_buffer, float_window) — both float, standard SDPA
  3. NO re-decode, NO QJL matmul. O(1) per step.
```

Memory: compressed storage (5x smaller) + decoded buffer (same as before) + float window.
Net savings: the decoded buffer is the same size as before compress, BUT the compressed storage
lets us evict the decoded buffer later (disk page, prefix cache) and reconstruct on demand.
The REAL memory savings come when vmlx evicts the decoded buffer to disk and only keeps compressed.

---

## Task 1: Fix Post-Compress Speed — Decoded Buffer Cache

**Files:**
- Modify: `jang-tools/jang_tools/turboquant/cache.py`
- Modify: `jang-tools/tests/test_turboquant_cache.py`
- Modify: `jang-tools/tests/test_turboquant_generate.py`

**What changes in cache.py:**

`compress()` now also stores `_decoded_k_buffer` and `_decoded_v_buffer` — the decoded float arrays of the compressed region. `_get_full_cache()` reads from these buffers instead of re-decoding.

**New tests:**
- `test_generate_after_compress_produces_output` — insert 100 tokens, compress(50), insert 20 more, verify shapes and no NaN
- `test_generate_after_compress_speed` — measure per-step time, must be < 5ms (not 100ms+)

---

## Task 2: Sink Token Preservation

**Files:**
- Modify: `jang-tools/jang_tools/turboquant/cache.py`
- Add tests: `jang-tools/tests/test_turboquant_cache.py`

**What:** When `compress(n)` is called, the first `sink_tokens` (default 4) tokens stay in the float window at full precision — never compressed. These are the BOS/system-prompt tokens that all subsequent tokens attend to heavily.

**Implementation:** `compress()` skips the first `sink_tokens` positions. Compresses `[sink..n)` instead of `[0..n)`. Float buffer keeps `[0..sink) + [n..offset)`.

**New tests:**
- `test_compress_preserves_sink_tokens` — compress with sink=4, verify first 4 tokens unchanged
- `test_sink_tokens_config` — verify sink_tokens flows from TurboQuantConfig

---

## Task 3: Nemotron Gate Dequant Fix (Task #39)

**Files:**
- Modify: `jang-tools/jang_tools/loader.py`

**What:** The Nemotron MoE gate weight stays as quantized uint32 instead of being dequantized. The current gate dequant code in the loader checks for `.mlp.gate.` patterns but Nemotron's gate is at `.block_sparse_moe.gate.` or similar path.

**Steps:**
1. SSH to Mac Studio, inspect the actual gate weight key names in Nemotron model
2. Fix the pattern matching in loader.py to catch Nemotron gate patterns
3. Verify Nemotron generates correctly
4. Then verify TurboQuant works on Nemotron end-to-end

---

## Task 4: Mistral 4 MLX-LM Patches

**Files:**
- Create or copy: `mistral4.py` in mlx-lm models (from `jang_tools/mistral4_mlx.py`)
- Modify: `mistral3.py` routing (add `mistral4` text_config type)

**What:** Apply the 7 patches needed for Mistral 4 on the MacBook's mlx-lm install so we can test locally without Mac Studio dependency. The patches are documented in `research/MISTRAL4-FIX-LOG.md`.

**Key issue found during Phase 1:** Our `mistral4_mlx.py` needs `input_embeddings` parameter added to `Model.__call__` and `Mistral4TextModel.__call__`. Also, the MLA `embed_q` weight shapes don't match between our native implementation and how JANG stores the weights.

**Approach:** Use our native `mistral4_mlx.py` (NOT the vmlx version which has different weight format expectations). Fix the `input_embeddings` param. Test with JANG_2L model on MacBook.

---

## Task 5: Proper Speed Benchmarking

**Files:**
- Create: `jang-tools/jang_tools/turboquant/benchmark.py`
- Create: `jang-tools/tests/test_turboquant_benchmark.py`

**What:** The Phase 1 benchmark measured `total_time / gen_tokens` which includes prefill in the denominator — making generation look 10x slower than it actually is. Need proper separation.

**Implementation:**
```python
def benchmark_generation(model, tokenizer, prompt, gen_tokens=100):
    """Returns dict with separated prefill_tok_s, gen_tok_s, peak_memory_gb."""
    # Prefill: time the model(prompt_tokens, cache=cache) call
    # Decode: time the token-by-token loop
    # Report separately
```

**Tests:**
- `test_benchmark_returns_all_fields`
- `test_benchmark_gen_speed_matches_baseline` — TQ gen speed within 5% of no-TQ

---

## Task 6: Compressed Generation E2E Test

**Files:**
- Create: `jang-tools/tests/test_turboquant_e2e_compressed.py`

**What:** The killer test — generate at long context WITH compression active, verify:
1. Output is coherent (not garbage)
2. Speed is close to baseline (not the 2.5 tok/s regression)
3. Memory is actually reduced vs baseline
4. Works on at least 2 model architectures

**Test scenarios:**
- Prefill 2K tokens → compress → generate 100 tokens → verify output quality
- Prefill 5K tokens → compress → generate 50 tokens → verify speed > 50% baseline
- Compare memory: baseline 5K float cache vs TQ compressed 5K cache

---

## Task 7: vmlx Cache Type Integration

**Files:**
- Modify: `vmlx_engine/utils/cache_types.py` (on Mac Studio at `/Users/eric/mlx/vllm-mlx/`)
- Modify: `vmlx_engine/utils/jang_loader.py`

**What:** vmlx's `detect_cache_type()` needs to recognize TurboQuantKVCache. The cache type detection routes to different code paths for batch operations, serialization, and memory management.

**Implementation:**
- Add `TURBOQUANT_KV_CACHE` to the CacheType enum
- Detect by checking `isinstance(cache, TurboQuantKVCache)` or class name
- For `is_positional_cache()`: return True (TQ cache is position-indexed like KVCache)
- For `estimate_kv_cache_memory()`: use `cache.nbytes` property
- For `_merge_caches()`: treat as KVCache (uses float buffer during generation)

---

## Task 8: Auto-Compress in vmlx Scheduler

**Files:**
- Modify: `vmlx_engine/mllm_scheduler.py`

**What:** When a request finishes and its cache is stored for prefix reuse, auto-compress the TurboQuant layers. This reduces prefix cache memory by 5x.

**Implementation:**
```python
# In _cleanup_finished_requests():
if cache_stored:
    from jang_tools.turboquant import compress_cache
    compress_cache(request.cache)  # 5x smaller for prefix storage
```

---

## Task 9: MMLU Benchmark with TurboQuant

**Files:**
- Use existing: `benchmark_mmlu.py` (on Mac Studio)

**What:** Run MMLU on at least 2 models with and without TurboQuant to prove zero quality degradation with actual benchmark numbers (not just "output looks correct").

**Models to test:**
- Qwen3.5-35B-A3B JANG_4S: should match baseline MMLU score
- MiniMax M2.5 JANG_2L: should match baseline MMLU score

**Success criteria:** TurboQuant MMLU within 0.5% of baseline (same as without TQ).

---

## Task 10: Documentation Cleanup

**Files:**
- Update: `research/turboquant/05-EXPERIMENT-LOG.md`
- Update: `research/turboquant/07-FULL-BENCHMARK-RESULTS.md`
- Update: `research/turboquant/06-EDGE-CASES-AND-ARCHITECTURE-ANALYSIS.md`
- Create: `research/turboquant/08-FINAL-STATUS.md`

**What:** Consolidate all findings, remove outdated projections (replace with actuals), document every known limitation, and write the final status doc.

---

## Dependency Chain

```
Task 1 (fix post-compress speed) ─────────────────────┐
Task 2 (sink tokens) ─────────────────────────────────┤
                                                       ├─→ Task 6 (compressed E2E)
Task 3 (Nemotron gate fix) ───────────────────────────┤
Task 4 (Mistral 4 patches) ───────────────────────────┘

Task 5 (benchmark tool) ──→ Task 9 (MMLU benchmark) ──→ Task 10 (docs)

Task 7 (vmlx cache type) ──→ Task 8 (vmlx auto-compress)
```

**Critical path:** Tasks 1 → 6 (fix speed, then prove it works)
**Parallel:** Tasks 3, 4 can be done independently
**vmlx track:** Tasks 7 → 8 (separate from core TQ work)

## Total estimated new tests: ~25
## Total tests after completion: ~84
