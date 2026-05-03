# Changelog

## 2.5.15 — 2026-05-03

### Fixed
- `DeepseekV4Cache.trim(n)` now does **proportional** pool-row truncation
  instead of v2.5.14's full reset. The cache accepts a new
  `compress_ratio` constructor arg (per-layer) and uses it to compute
  how many trailing `pooled` rows correspond to the trimmed KV tokens
  (`rows_to_drop = max(1, n // ratio)`). The kept-prefix pool survives
  across multi-turn — so long-context chats no longer pay full pool
  re-derivation on every turn.

  Strategy mirrors llama.cpp's
  [antirez/llama.cpp-deepseek-v4-flash](https://github.com/antirez/llama.cpp-deepseek-v4-flash)
  `dsv4_clear_rows` in `src/llama-memory-hybrid-iswa.cpp`:
  `row_begin = p0 / ratio`, `row_end = ceil(p1 / ratio)`. Same principle
  in MLX form: slice `pooled[:, :keep, :]` where `keep = n_rows -
  rows_to_drop`.

  `buffer_kv` and `buffer_gate` partial-window buffers are still
  cleared unconditionally — their start_pos invariants are invalidated
  by any trim, and `accumulate_windows` already handles None init.

  Backward-compatible: when `compress_ratio` is None (legacy single-arg
  construction), `trim()` falls back to v2.5.14's full reset — still
  correct, just less efficient.

### Notes
- This only affects `cache_type="kv"` DSV4-Flash multi-turn `/v1/chat/
  completions` performance, not correctness. v2.5.14 already fixed the
  chat-mode loop root cause; v2.5.15 just makes the fix cheaper for
  long-context multi-turn.

## 2.5.14 — 2026-05-03

### Fixed
- `DeepseekV4Cache.trim(n)` now resets the cumulative compressor +
  indexer pool state in addition to delegating to the inner
  `RotatingKVCache.trim(n)`. Pre-fix the pool buffers (`buffer_kv`,
  `buffer_gate`, `pooled` keys in `compressor_state` and
  `indexer_state`) survived the truncation reflecting pre-trim KV
  positions — including output-side tokens the trim was meant to
  discard. On multi-turn `/v1/chat/completions` the scheduler's
  prefix-cache reuse restored that contaminated pool on the next
  turn, and the model's HSA / CSA pool-attention path read
  global-context vectors built from prior turns' GENERATED OUTPUT.
  Symptom: DSV4-Flash drifted into the polite-assistant attractor
  on chat completions ("How are things with you? Let me know if
  there's anything I can help with." repeated until max_tokens).
  Bench harness (SimpleEngine, no cache reuse across requests) was
  unaffected — that's how we know the model itself is sound and
  the bug was specifically in cross-turn pool-state survival.

  Reset semantics: pool state goes to None on every trim. Next
  forward pass re-derives via `accumulate_windows` + `update_pool`
  from the kept KV. Both helpers already handle None-pool init
  cleanly. Cost: marginal first-forward latency on the next turn;
  coherence preserved across arbitrary multi-turn chats.

  This makes the existing engine-side classification at
  `vmlx_engine/scheduler.py:770` (`non_kv.discard("DeepseekV4Cache")`)
  correct — the cache's external surface is now genuinely KV-shaped,
  so prefix-cache reuse is safe.

### Notes
- Engine-side rep_penalty bandaids in `_FAMILY_FALLBACK_DEFAULTS`
  for `deepseek_v4` (1.15) were masking this same cumulative-pool
  contamination via diversification pressure rather than fixing it.
  Once 2.5.14 lands the engine can revert that to 1.05 (the MMLU
  91 % baseline value).

## 2.5.13 — 2026-05-03

### Fixed
- `load_jangtq.py`: P18 QKV-fusion patch crashed on attention classes
  whose head-count attribute isn't `num_attention_heads`. NemotronH /
  DeepSeek use `num_heads`; Qwen3 uses `n_heads`. Mirrored the
  pre-patch safety check's getattr fallback inside the patched
  `__call__` body so the reshape resolves correctly across families.
  Also bails to the original call when no recognised name is found.
- `load_jangtq.py`: P18 unconditionally called `self.rope(queries,
  ...)` but NemotronHAttention has no rope (cache update happens
  directly before SDPA). Gated the rope step behind `hasattr` and
  added a fallback for `self.scale` (1/sqrt(head_dim)) for classes
  that don't expose `scale` or use `softmax_scale`.

These fixes make Nemotron-H bundles bootable through the canonical
`load_jangtq_model` path. Without them the engine emitted empty
responses with no decoded tokens because every prefill aborted on
the AttributeError.


## 2.5.12 — 2026-05-02

### Added
- `jang_tools.jangrt.jangtq_hydrate` — pure helper that swaps
  `nn.Linear` / `SwitchLinear` modules carrying `.tq_packed` /
  `.tq_norms` / `.tq_bits` keys to `TurboQuantLinear` /
  `TurboQuantSwitchLinear` and returns the leftover regular weights.

### Fixed
- Laguna JANGTQ (`weight_format=mxtq`) bundles now load. Previous
  releases hit `ValueError: Module does not have parameter named
  "experts"` at `model.update` because the runtime's nn.quantize
  predicate-based affine path could not bind `.tq_packed` keys to
  bare `nn.Linear` / `SwitchLinear` modules.
- Mistral-Medium-3.5 (`ministral3`) JANGTQ — same fix shape (dense,
  TurboQuantLinear only).

## 2.5.4 — 2026-04-24

### Verified
- Long-context generation works coherently up to 700+ token prompts on
  default plain-`KVCache` path. Compressor auto-triggers during prefill
  via `L >= compress_ratio` check.
- HumanEval-style code generation produces correct `is_palindrome`,
  list ops, etc. — model is bench-ready.

### Investigated
- `mx.fast.rope` fast path tried for forward RoPE (saves ~129 ops/token
  across 43 layers × 3 rope calls). Produced incoherent output — likely
  YaRN inv_freq scale convention mismatch with `mx.fast.rope` expectations.
  Kept manual cos/sin path; documented in code comment for future.


## 2.5.3 — 2026-04-24

### Added
- `Model.make_cache()` for `deepseek_v4` returns proper per-layer cache list.
  Defaults to plain `KVCache` (short-prompt-safe). Set env `DSV4_LONG_CTX=1`
  to enable `DeepseekV4Cache` for compress_ratio>0 layers (long-context path
  still under refinement).
- `DeepseekV4Cache.state` / `keys` / `meta_state` properties for proper
  `mlx_lm.generate` pipelined evaluation.
- `jang_tools.load_jangtq.load_jangtq_model` auto-registers `deepseek_v4`
  via `import jang_tools.dsv4` when the bundle's model_type matches.

### Known issue
- Long-context (>128 token) generation through Compressor + Indexer path is
  experimental; works for short prompts (≤128 tokens) coherently. Long-prompt
  decode polish is a follow-up.

## 2.5.2 — 2026-04-24

### Removed
- Removed unused experimental scaffolding modules. Re-released without them.

## 2.5.1 — 2026-04-24

### Renamed
- `jang_tools.dsv4_prune` → `jang_tools.dsv4` (the folder is quantization +
  runtime, not pruning — the original name was misleading).

## 2.5.0 — 2026-04-24

### Added — DeepSeek-V4-Flash support
- New `jang_tools.dsv4` package with full DSV4 support:
  - `convert_dsv4_jangtq.py` — convert FP4+FP8 (or BF16-dequant) source to JANG/JANGTQ bundles. Profiles: 2L (2-bit affine), 4 (4-bit affine), JANGTQ2 (2-bit MXTQ codebook + 8-bit attn), JANGTQ4 (4-bit affine routed + 8-bit attn).
  - `mlx_model.py` — full DSV4 runtime with MLA head_dim=512 + mHC + Compressor/Indexer + per-layer RoPE.
  - `mlx_register.py` — registers `deepseek_v4` model_type with mlx_lm.
  - 13 architecture-specific bug fixes vs naïve port (see `research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md`).
- Native fused Metal kernel for `hc_split_sinkhorn` (mHC Sinkhorn doubly-stochastic normalization). Replaces ~40 MLX ops per call with single GPU dispatch.
- Coherent generation verified on JANG_2L (107 GB), JANG4 (173 GB), JANGTQ2 (74 GB), JANGTQ4 (173 GB) bundle formats.

### Added — Kimi K2.6 expert pruning
- New `jang_tools.kimi_prune` package: routing-aware expert pruning (REAP-style) with absorb-merge.
  - `build_calib.py` / `build_calib_v2.py` — calibration corpus assembly.
  - `profile.py` — captures routing freq + coact + output_energy per layer.
  - `score.py` — per-expert importance.
  - `prune.py` — drops experts + absorb-merge + router rewrite + FP8 reshard.
  - `convert_kimi_jangtq.py` — JANGTQ conversion for pruned Kimi.
  - `bench_humaneval*.py`, `bench_mmlu*.py` — eval harnesses.

### Changed
- `convert_dsv4_jangtq.py`: BF16-source aware (graceful fallthrough to `mx.quantize` when `.scale` siblings are absent — no more direct-copy MXFP4 attempt on dequant sources).

### Removed
- Hardcoded user paths from new code (DSV4 + Kimi). All examples now use `<path/to/...>` placeholders or env vars.

### Known limitations
- DSV4-Flash decode steady-state ~21 tok/s on Mac Studio M3 Ultra (Python). Architecture-inherent (MLA head_dim=512). Native Swift via vMLX targets 40-50 tok/s.
- DSV4 HP bundle format (mxfp4 direct-copy + bf16 non-routed) is unstable — residual stream explodes over 43 layers. Use JANGTQ2/JANGTQ4/JANG4 instead.

## 2.4.2

(prior release — see git history)
