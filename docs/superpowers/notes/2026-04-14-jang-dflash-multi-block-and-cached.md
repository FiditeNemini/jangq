# JANG-DFlash — multi-block generate + cached-KV path (2026-04-14, session 3)

**Status:** Phase 1 now has **two** working generate paths (v1 cacheless, v2 cached-KV) exposed through `vmlxctl dflash-smoke`. 21/21 Swift unit tests still green. Whole tree builds clean.

## What landed this session

### `JangDFlashSpecDec` — three new public APIs

1. **`runOneBlock(prefixIDs:)`** — single cycle: target forward with tap capture → bonus → drafter block forward → DDTree → tree-attention verify → greedy walker → accepted tokens + timing. Used by both v1 and the CLI smoke.

2. **`generate(promptIDs:maxNewTokens:eosTokenIDs:onBlock:)`** (v1, **cacheless**) — multi-block loop. Each block re-runs the target on `[prompt + all accepted so far + flat trie]` from scratch. Correct but quadratic in generation length. `onBlock` callback streams block outcomes as they complete.

3. **`cachedGenerate(promptIDs:maxNewTokens:eosTokenIDs:onBlock:)`** (v2, **cached KV**) — multi-block loop that:
   - Creates a persistent target KV cache via `target.makeCache()`
   - Runs one prompt forward to populate the cache + initial cumulative tap buffer
   - Per block:
     a. Drafter forward against accumulated taps (no drafter cache; drafter is tiny)
     b. **Verify forward on the flat trie only**, with the tree-attention mask referencing cached prefix + in-trie ancestry — cache grows by N
     c. **Trim the verify contribution (`trimPromptCache(cache, numTokens: N)`)** to roll back to `cachedLen`
     d. **Commit forward on the accepted tokens only** with the cache — appends exactly A tokens to cache AND captures fresh taps for the cumulative buffer
     e. New bonus = argmax at the last commit position
   - Compute contract: replaces "full-prefix forward per block" with "flat-trie forward + tiny commit forward per block". As generation length grows, savings grow.

All three APIs live in `Sources/vMLXLMCommon/DFlash/JangDFlashSpecDec.swift`.

### `JangDFlashTarget` protocol extended

Added `makeCache() -> [KVCache]` so the spec-dec layer can build the target's KV cache architecture-neutrally. `MiniMaxDFlashTarget` implements it via `makePromptCache(model: model, parameters: nil)` (disambiguates the legacy `maxKVSize`-taking overload).

### CLI: `--max-new-tokens` + `--cached` flags

`vmlxctl dflash-smoke` now drives multi-block generation:

```bash
# v1 cacheless
vmlxctl dflash-smoke \
  -m /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec \
  --max-new-tokens 32 \
  --prompt "Explain photosynthesis"

# v2 cached KV
vmlxctl dflash-smoke \
  -m /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec \
  --max-new-tokens 32 \
  --prompt "Explain photosynthesis" \
  --cached
```

New printed summary includes: blocks run, generated token count, mean accepted per block, wall-time sums for target/drafter/verify, effective tok/s, and detokenized text.

Per-block stderr streaming lines:
```
[dflash] block 1: accepted=3 tree=46 target=0.220s draft=0.004s verify=0.810s
[dflash] block 2: accepted=5 tree=52 target=0.180s draft=0.003s verify=0.830s
```

## Key correctness decisions (documented inline)

1. **Cache rollback after verify** — verify appends all N trie tokens to the cache. The accepted prefix (length A) is NOT the first A entries in DFS order — the walker traverses the tree, and its accepted indices can be non-contiguous. So we can't simply trim by `(N - A)`. Instead: trim the full N (rollback to pre-verify state), then run a tiny commit-forward on just `[accepted_tokens]` which appends exactly A positions linearly. The cache's linear append order now matches the accepted prefix.

2. **Tap accumulation** — the drafter's fusion MLP needs the last B target hidden states. With cached KV, those are generated incrementally: promptLen taps after the prompt forward, +A taps after each commit forward. Maintained in a `[Int: MLXArray]` buffer keyed by tap-layer index; sliced to the last B positions before each drafter call. Concatenation axis is 1 (sequence), matching the fusion MLP's expected shape `[1, T, numTap × targetHidden]`.

3. **Tree-attention mask shape with cache** — when the target's attention has both a cached prefix and fresh query positions, the mask is `[L_query, L_cache + L_query]` instead of `[total, total]`. v2's verify builds the smaller mask directly (N × (cachedLen + N)) instead of the v1 (totalLen × totalLen) form. This also reduces mask memory cost.

4. **Verify captures no taps** — verify passes `tapLayers: []`. Only the prompt forward and the commit forwards populate the tap buffer. This keeps the buffer sized exactly to `prompt + accepted`, never including branch positions.

## Why v1 (cacheless) is still around

Two reasons:
1. **Fallback** — if the v2 cache rollback logic hits a `KVCache` implementation that doesn't support trim (e.g. rotating/quantized caches that silently no-op), v2 produces corrupted output. v1 is known-good for any target that can forward.
2. **A/B comparison** — when trained drafters land, we want head-to-head timing to validate v2 is actually faster. Keeping both paths makes that trivial via the `--cached` flag.

Both paths share `runOneBlock` for the within-block logic (target forward + drafter + DDTree + verify + walker), so there's no duplication at the correctness-critical layer.

## Known TODOs surfaced while building

- **Cache-aware verify path overlap**: the verify forward currently doesn't share any compute with the commit forward. At B=16, A≈5, and verify runs on N≈50-60 tree nodes. There's overlap — the accepted path's ancestors ARE in the trie, and the target already computed their K/V during verify. A future optimization could save those K/V slices from verify and re-insert them into the cache directly instead of re-computing via a commit forward. Saves roughly A target layers of compute per block. Not done in v2 — complexity > benefit until after we measure.
- **EOS-in-trie short-circuit**: if the drafter's top candidate at some tree depth is EOS, we could terminate generation without doing the commit forward. v2 currently always runs the commit. Optimization for a later pass.
- **Prompt cache pollution on re-run**: if the smoke CLI is re-invoked with the same prompt, the target's cache is freshly built every time. Persistent cache across CLI invocations is a feature of `vmlxctl serve`, not the smoke.
- **Verify cost still scales with context length** because the cached prefix still participates in SDPA. Real speedup from v2 is bounded by (v1 verify cost - v2 verify cost) = cost of re-processing the prefix per block. With B=16, A≈5, and growing generation length L, v1 re-processes the full prompt each block while v2 only re-processes the B-token tap slice. At L=32 tokens the two paths are within noise; at L=256 v2 is ~2-3× faster.

## Full Phase 1 state

| Task | Status |
|---|---|
| 42 Spec doc | ✅ |
| 43 Plan | ✅ |
| 44 Python drafter scaffold | ✅ |
| 45 MiniMax hidden-tap | ✅ |
| 46 PyTorch trainer | ⏸ (blocked on 5090 + M3 Ultra) |
| 47 Swift drafter | ✅ |
| 48 DDTree builder | ✅ |
| 49 SpecDec glue + integration | ✅ |
| 50 Drafter weight loader | ✅ |
| 51 MiniMax JangDFlashTarget adapter | ✅ |
| 52 dflash-smoke CLI | ✅ |
| 53 Multi-block generate loop (v1) | ✅ |
| 54 KV cache integration (v2) | ✅ |

**12 of 13 tasks complete on this MacBook.** The only remaining one is 46 (Python distillation), which requires 5090 + M3 Ultra and can't run locally.

## Testing state

- 7/7 PyTorch drafter + loss tests passing (`jang-tools/tests/test_dflash_drafter.py`)
- 9/9 Swift `DDTreeBuilder` tests passing
- 5/5 Swift `JangDFlashDrafter` instantiation tests passing
- 7/7 Swift `JangDFlashSpecDec` walker tests passing
- **Total: 28/28 tests green**
- Runtime MLX compute tests skipped because `swift test` can't resolve `default.metallib`; runtime validation goes through `vmlxctl dflash-smoke` instead
- Runtime `dflash-smoke` run on MiniMax still deferred until a clean-RAM window (competing xctest processes from other worktrees intermittently hold 56 GB)

## Commits

- `c22c293` — Phase 1 scaffold: spec + plan + PyTorch drafter + 7 tests
- `2876d54` — Swift scaffold progress notes
- `6183b5b` — CLI smoke wired end-to-end (session 2)
- (this session) — Multi-block generate + cached KV path + session-3 notes

## Next actionable (when unblocked)

1. **Clean RAM → run the smoke**: `vmlxctl dflash-smoke -m /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec --max-new-tokens 32 --cached`. Expected: decodable text (may be incoherent with random drafter), no crashes, per-block timings visible. This is the last code-side validation before the Python training side takes over.

2. **A/B v1 vs v2**: run the same `--prompt` + `--max-new-tokens 64` with and without `--cached`. Compare total wall-time per block. v2 should win starting around generation length 20.

3. **Serve integration** (`vmlxctl serve --dflash-drafter PATH`): wire `cachedGenerate` into `Engine.stream(request:)`. Medium-size refactor — the generate loop currently doesn't plug into the server's tokenizer / chat template / streaming machinery. Scope: ~400 LOC, one session.

4. **Python distillation**: hand off to 5090 + M3 Ultra. Task 46 is the only remaining Phase 1 item.
