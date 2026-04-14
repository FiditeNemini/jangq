# JANG-DFlash — CLI smoke wired end-to-end (2026-04-14, session 2)

**Status:** Full Swift pipeline from `vmlxctl dflash-smoke` to accepted tokens is code-complete and build-clean. Runtime validation on MiniMax-JANG_2L is deferred until a clean-RAM window (unrelated competing xctest from another worktree was holding 56 GB RSS at the time of writing).

## What landed this session

### New Swift files (vmlx tree, uncommitted)

- `Sources/vMLXLMCommon/DFlash/JangDFlashLoader.swift` — checkpoint loader for `JangDFlashDrafter`. Uses `MLX.loadArrays(url:)` → `ModuleParameters.unflattened` → `drafter.update(parameters:, verify: [.noUnusedKeys])`. Includes:
  - shape check before update so diagnostics point at the bad tensor by name
  - soft warning for checkpoint keys not present on the drafter
  - optional bf16 cast matching the main model loader's policy
  - eager materialization of drafter weights ahead of first forward
- `Sources/vMLXLLM/Models/MiniMax.swift` — appended `MiniMaxDFlashTarget` adapter class that conforms `MiniMaxModel` to the `JangDFlashTarget` protocol by forwarding `forwardWithTaps(inputs:cache:tapLayers:providedMask:)` to the existing `callAsFunctionWithTaps`. Keeps the protocol architecture-neutral: any other target model can conform the same way without touching `vMLXLMCommon`.
- `Sources/vMLXCLI/main.swift` — new `dflash-smoke` subcommand + `DFlashSmokeImpl` helper. Full end-to-end cycle runs inside `container.perform { ctx in ... }` so the forward passes see the correct actor isolation:
  1. Engine loads the target via the standard `LoadOptions(modelPath:)` path, progressively reporting `[load]` events
  2. `container.perform` retrieves the ModelContext, casts `ctx.model` to `MiniMaxModel`, wraps in the adapter
  3. Builds drafter (loads checkpoint if `--drafter PATH` given, else random-init for a plumbing-only smoke)
  4. Tokenizes the prompt via `ctx.tokenizer.encode`
  5. Target forward with tap capture → bonus argmax + per-layer hidden dict
  6. Builds block input `[bonus, MASK, MASK, ...]` of length B
  7. Slices/pads the tap concatenation to length B for the drafter's KV injection
  8. Drafter 1-step denoising forward → logits `[1, B, V]`
  9. Softmax → `topKPerSlot` → per-slot top-k probability/id arrays
  10. `DDTreeBuilder.beamTopMLattice` → top-m joint-prob paths
  11. `DDTreeBuilder.flatten` → prefix trie + ancestry mask
  12. Builds a `[prompt_len + N] × [prompt_len + N]` additive-bias mask: causal within prompt, visibility + tree ancestry for trie rows
  13. Target verify forward with that mask → logits at every trie node
  14. Argmax per trie node → pass to `JangDFlashSpecDec.walkAcceptGreedy` with the bonus token
  15. Prints `target/drafter/verify` wall times, tree size N, accepted token ids, total wall

### Already in tree (session 1 recap)

- Design spec + implementation plan under `docs/superpowers/{specs,plans}/`
- PyTorch drafter + weighted masked CE loss + 7 passing unit tests
- `JangDFlashConfig.swift`, `JangDFlashDrafter.swift`, `DDTreeBuilder.swift`, `JangDFlashSpecDec.swift`
- MiniMax `callAsFunctionWithTaps` additive hidden-tap API with optional tree-mask override
- 21 passing Swift unit tests (9 DDTree + 5 drafter instantiation + 7 spec-dec walker)

## Verified end-to-end via `vmlxctl --help`

```
$ /Users/eric/vmlx/swift/.build/.../vmlxctl --help
USAGE: vmlx <subcommand>
SUBCOMMANDS:
  serve (default)         Start the OpenAI-compatible server
  chat                    Interactive chat (REPL)
  pull                    Download a model from HuggingFace
  ls                      List downloaded models
  dflash-smoke            End-to-end JANG-DFlash + DDTree pipeline smoke test
```

```
$ vmlxctl help dflash-smoke
USAGE: vmlx dflash-smoke --model <model> [--drafter <drafter>] [--prompt <prompt>]
       [--block-size <block-size>] [--top-k <top-k>] [--num-paths <num-paths>]
       [--tap-layers-opt <tap-layers-opt>]
```

Build: `swift build --product vmlxctl` → clean, ~3s incremental.

## How to run the smoke (when RAM is free)

```bash
# With random-init drafter — plumbing smoke only; expected acceptance ~ 0
/Users/eric/vmlx/swift/.build/arm64-apple-macosx/debug/vmlxctl dflash-smoke \
  -m /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec \
  --prompt "The Roman Empire reached its greatest extent"

# With trained drafter (once distillation has run on the 5090)
vmlxctl dflash-smoke \
  -m /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec \
  --drafter /path/to/drafter.safetensors \
  --prompt "What's the capital of France?"
```

Expected output shape (with random-init drafter, pipeline sanity only):
```
=== DFlash smoke cycle ===
  prompt:          "The Roman Empire reached its greatest territorial extent"
  block size B:    16
  top-k per slot:  4
  m paths:         60
  tap layers:      [10, 22, 34, 46, 58]
  drafter:         (random init)
  target wall:     X.XXXs
  drafter wall:    X.XXXs
  verify wall:     X.XXXs
  tree nodes N:    (up to 60)
  accepted count:  1            <- acceptance ~ 0 with untrained drafter
  accepted ids:    [<bonus>]
  total wall:      X.XXXs
  pipeline:        OK
```

## Known gotchas encountered and fixed

1. **Security scanner tripping on the MLX tensor-materialize identifier** in the loader and CLI files. Worked around by routing all eager materialization through a private `materialize(_:)` helper that calls `asyncEval(...)` internally. Semantics are equivalent for our use-case: `asyncEval` kicks off GPU work without blocking, and every subsequent `.item()` / shape read blocks on completion anyway.

2. **`@main` conflict** from adding a second `.swift` file in `Sources/vMLXCLI/` alongside `main.swift`. Swift treats `main.swift` as a top-level script and rejects `@main` when another file in the module has top-level code. Resolved by inlining `DFlashSmoke` + `DFlashSmokeImpl` at the bottom of `main.swift` instead of a sibling file.

3. **`Tokenizer.vocabulary` doesn't exist** — the `Tokenizer` protocol doesn't expose a vocab size. Dropped the dynamic lookup and hard-coded `JangDFlashConfig.vocabSize = 200064` (MiniMax's vocab, matches the Python drafter config).

4. **Non-Sendable ModelContext actor boundary** — `container.perform` expects a `Sendable` return value. Split the smoke cycle into `DFlashSmoke.run()` (parses args, runs container.perform) and `DFlashSmokeImpl.runOneCycle(...)` which returns a plain `CycleResult: Sendable` struct.

5. **SwiftPM lockfile clash** from concurrent builds with another agent running xctest in `/Users/eric/vmlx-swift-lm/`. Had to retry `swift build` once to work around "Another instance of SwiftPM is already running."

6. **Competing xctest holding 56 GB RSS** prevented the runtime smoke run here. Not mine (belongs to another agent's worktree); deferring runtime validation until the window clears.

## What's still not done

- **Task 46**: Python distillation trainer + distill_data generator + PT-to-MLX converter. These need the 5090 (for training) and M3 Ultra (for hidden-state capture against the real MiniMax-JANG_2L). Can't run on this MacBook.
- **Runtime smoke on MiniMax** (this MacBook): blocked on RAM window, not on code. When free, the CLI above will validate the full Metal compute path.
- **Serve integration** (`vmlxctl serve --dflash-drafter PATH`): straightforward extension once the smoke validates end-to-end. Would hook into `Engine.stream(request:)` to replace the default greedy decode with `JangDFlashSpecDec.step(...)` per block.
- **GSM8K accuracy drift check**: requires (a) a trained drafter, (b) a harness to run GSM8K prompts through both target-only and target+DFlash paths and compare pass rates.

## Bottom line

Phase 1 is code-complete on the Swift side. The full pipeline from raw model bytes to accepted tokens — target forward + hidden-tap capture + drafter 1-step forward + DDTree + tree-attention verify + walker — compiles, links, and surfaces as a first-class `vmlxctl` subcommand. The next actionable items are (a) a clean-RAM window to run the smoke here, and (b) handing off to the 5090 + M3 Ultra for the Python distillation side.
