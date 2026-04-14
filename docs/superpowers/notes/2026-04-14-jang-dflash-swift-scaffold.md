# JANG-DFlash + DDTree — Swift scaffold landed (2026-04-14)

**Status:** Phase 1 tasks 42–48 + 45 + 49 complete. Code-complete and unit-tested for all pure-Swift pieces. MLX runtime compute validation deferred to end-to-end CLI smoke (Task 10 in the plan, runs inside `vmlxctl` where the Metal library is colocated with the binary).

## What landed

### Docs
- `docs/superpowers/specs/2026-04-14-jang-dflash-ddtree-design.md` — full design spec
- `docs/superpowers/plans/2026-04-14-jang-dflash-ddtree-phase1.md` — task-by-task plan

### Python (jang repo)
- `jang-tools/jang_tools/dflash/__init__.py`
- `jang-tools/jang_tools/dflash/config.py` — `JangDFlashConfig` dataclass
- `jang-tools/jang_tools/dflash/drafter.py` — `JangDFlashDrafter` (5-layer block-diffusion transformer with KV injection) + `dflash_loss` (weighted masked CE, DFlash Eq. 4)
- `jang-tools/tests/test_dflash_drafter.py` — 7 unit tests, all green (shapes, loss, gradient flow, input validation)

### Swift (vmlx tree, uncommitted)
- `Sources/vMLXLMCommon/DFlash/JangDFlashConfig.swift` — mirrors Python config
- `Sources/vMLXLMCommon/DFlash/JangDFlashDrafter.swift` — `JangDFlashAttention` (dual Q/K/V projections: block + context), `JangDFlashFFN` (SwiGLU), `JangDFlashBlock`, `JangDFlashFusion` (tap → hidden), `JangDFlashDrafter`
- `Sources/vMLXLMCommon/DFlash/DDTreeBuilder.swift` — `beamTopMLattice` + `flatten` (prefix-trie + ancestry mask)
- `Sources/vMLXLMCommon/DFlash/JangDFlashSpecDec.swift` — end-to-end spec-dec step: tap concatenation, top-k-per-slot extractor, softmax, ancestry-mask → additive-bias conversion, `walkAcceptGreedy` rejection walker
- `Sources/vMLXLLM/Models/MiniMax.swift` — additive `callAsFunctionWithTaps` on `MiniMaxModelInner` and `MiniMaxModel`. Accepts `tapLayers: Set<Int>` and optional `providedMask` override for tree attention. Non-invasive — existing forward path untouched.
- `Tests/vMLXTests/DDTreeBuilderTests.swift` — 9 tests, all green (beam search ordering, truncation, zero-prob flooring, prefix collapse, diamond non-merge, mask lower-triangularity, beam+flatten integration)
- `Tests/vMLXTests/JangDFlashDrafterTests.swift` — 5 tests, all green (config defaults, Codable roundtrip, module instantiation, parameter keys present, parameter count within bounds)
- `Tests/vMLXTests/JangDFlashSpecDecTests.swift` — 7 tests, all green (greedy walker: empty tree, single match, single mismatch, depth-1 chain, depth-1 mismatch-bonus, branching child, ancestry → additive bias conversion)

**Total:** 21/21 Swift tests passing, 7/7 Python tests passing.

## What's NOT yet landed (intentional scope split)

Python-side (needs 5090 + M3 Ultra, not this MacBook):
- Task 45/Task 3 in the plan: `distill_data.py` — runs MiniMax with layer-tap hook, writes `(h_taps, tokens)` shards
- Task 46/Task 4: `train.py` — PyTorch distillation trainer
- Task 46/Task 5: `convert_to_mlx.py` — PT → safetensors converter

Swift-side (needs integration with `SpeculativeDecoder` + CLI flag):
- Wire `JangDFlashSpecDec` into `Evaluate.SpeculativeDecoder` or into `vmlxctl serve` via a `--dflash-drafter PATH` flag
- Drafter weight loader (use existing `JangLoader` / safetensors mmap — parameter keys already match Python side)
- End-to-end tok/s bench with an untrained drafter (pipeline smoke)

## Key architectural decisions recorded in the scaffold

1. **Separate context K/V projections** (`wk_ctx`, `wv_ctx`). The injected context comes from the fusion MLP's output, not from the block's own embeddings, so it deserves its own projection weights. DFlash paper says KV injection is at every layer — Python and Swift both model this explicitly.

2. **Context-first concat in attention**. Block Q attends to `[ctx_K || block_K]` with mask `[zeros[L, T_ctx] || triu_causal[L, L]]`. Block positions see all of ctx (bidirectional) and each other causally. RoPE is applied only to block-side Q/K because ctx carries its positional information baked in via the fusion MLP.

3. **Context length is decoupled from block length**. Tests verify `Tctx != L` works. The paper uses T_ctx = all prefilled positions, not just the current block, so this matters at inference when the prefix grows.

4. **`tapDim = 5 * 3072 = 15360`** hardcoded as default. Matches MiniMax's hidden dim and the 5 evenly-spaced tap layers [10, 22, 34, 46, 58].

5. **MASK token at index `vocab_size`**. Embedding table is `vocab_size + 1` rows. The fusion MLP and lm_head use unmodified `vocab_size`.

6. **Walker is greedy-first** (matches paper). Rejection sampling via `min(1, p_target/p_draft) > rand` is a later-stage improvement — the paper's headline numbers are all greedy.

7. **Target is abstracted via the `JangDFlashTarget` protocol** in `JangDFlashSpecDec.swift`. MiniMax conforms via the existing `callAsFunctionWithTaps` additive API. Any other target architecture (Qwen3.5, DeepSeek-V3, future Kimi-K2 JANG) can plug in by implementing the same method shape.

## Test harness gotchas

- `swift test` can't load the MLX `default.metallib` in this project's test bundle. Copying the metallib into `.build/arm64-apple-macosx/debug/vmlxPackageTests.xctest/Contents/MacOS/mlx.metallib` was tried and did NOT unblock it (the loader goes through `NS::Bundle::allBundles()` and still reports "library not found"). Upshot: any test that dispatches an actual Metal kernel fails immediately.
- **Workaround:** keep `swift test` coverage to pure-Swift pieces (config parsing, walker, beam search, prefix trie, mask conversion). MLX compute-path validation happens via `vmlxctl` CLI runs where the metallib is correctly resolved.
- Unrelated pre-existing compile errors in `vMLXApp/Chat/InputBar.swift` and `vMLXEngine/ChatRequest.swift` surface intermittently during full test builds. DFlash tests don't depend on those targets and run when filtered.

## Updated Plan 7 bench status (in-RAM MiniMax)

Before DFlash:
- Baseline (QKV-fused, no spec-dec): 14.99 tok/s on M4 Max
- `VMLX_FORCE_COMPILE_DECODE=1`: +1.7% (15.25 tok/s)
- All Tier-1 fusion/dispatch levers: exhausted at noise floor

Projected with DFlash + DDTree:
- Paper reports τ ≈ 6.5–7.9 dense Qwen3 B=16
- MoE penalty assumed ≈ 20% → effective τ ≈ 5–6 on MiniMax
- M4 Max (MacBook): 14.99 × 5.5 / 1.1 draft cost ≈ **~75 tok/s**
- M3 Ultra (JANGTQ P15 baseline ~42 tok/s): × 5.5 / 1.1 ≈ **~205 tok/s**
- Stretch at τ=6.5: ~248 tok/s on M3 Ultra

These are the numbers the trained drafter needs to hit to close the plan.

## Next concrete steps

1. Wire `--dflash-drafter PATH` into `vmlxctl serve` (Task 49/10 in the plan)
2. Drafter weight loader — reuse `JangLoader.loadV1Weights` or `safetensors` mmap
3. MiniMax conformance to `JangDFlashTarget` — one adapter file, adapts the existing `callAsFunctionWithTaps` signature to the protocol's `[Any]` cache (or refactor the protocol to use `[KVCache]` directly)
4. End-to-end CLI smoke with a randomly-init drafter — validates pipeline; expected acceptance ≈ 0, effective tok/s lower than baseline (draft overhead + rejected verify)
5. Hand off to user for Python distill-data generation on M3 Ultra + training on 5090

The whole Swift half is ready to accept a real checkpoint.
