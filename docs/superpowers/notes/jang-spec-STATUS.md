# jang-spec — Running Status

Last updated: 2026-04-13

**Single source of truth for what's shipped, what's in progress, and what's next on the jang-spec project.** Every plan updates this file in its final commit.

---

## TL;DR

SSD-streamed MoE speculative decoding for JANG models. v1 = streaming + Medusa drafting + prior-based prefetch. Plans 1 and 2 done, Plan 3 starting. Three real MoE bundles built and verified (Gemma-4-26B, Qwen3.5-35B-A3B, MiniMax M2.7 228B). Streaming IO spike GO on M4 Max (12.54 GB/s random).

---

## Design

- **Spec:** `docs/superpowers/specs/2026-04-13-jang-spec-design.md` (14 sections)
- **Framing:** "SSD-Streamed MoE Speculative Decoding" — router-aware drafting deferred to v2 (§13)
- **DFlash relationship:** parallel block drafting cited as prior art only; no code, checkpoints, or recipes reused. jang-spec's novelty is the streaming runtime + Apple-Silicon-native Metal/unified-memory path.

---

## Plans

| # | Plan | Status | Branch | Artifacts |
|---|---|---|---|---|
| 1 | Bundle format + Python builder + IO spike | **DONE** | `jang-spec-plan1-bundle` | `jang_tools.jangspec`, `jang spec build/inspect` CLI, `jang-spec-iobench` |
| 2 | Swift bundle loader (JANGCore) | **DONE** | `jang-spec-plan2-swift-loader` | `JANGCore` library, `jang-core inspect` CLI |
| 3 | Hot-core loader (Swift, v2 safetensors, no Metal yet) | **STARTING** | `jang-spec-plan3-hotcore` (next) | TBD |
| 4 | Metal v2 quantized matmul kernel | queued | — | — |
| 5 | Dense JANG v2 forward pass | queued | — | — |
| 6 | MoE layer (router, shared expert, switch_mlp, RAM-resident) | queued | — | — |
| 7 | ExpertStreamer — swap RAM experts for SSD streaming | queued | — | — |
| 8 | Medusa drafter + speculative decoding loop | queued | — | — |
| 9 | `jang-specd` daemon + vmlx integration | queued | — | — |

---

## Bundles built

Real `.jangspec` bundles produced by Plan 1's `jang spec build`, verified by Plan 2's `jang-core inspect`. Format is identical across all three.

| Source | Arch | Layers | Experts/layer | Top-k | Bits | Hot core | Experts | Total | Build time |
|---|---|---|---|---|---|---|---|---|---|
| Gemma-4-26B-A4B-it-JANG_4M | `gemma4` | 30 | 128 | 2 | 4 | 3.32 GB | 12.85 GB | 16.2 GB | ~6.5 s |
| Qwen3.5-35B-A3B-JANG_2S-TEXT | `qwen3_5_moe` | 40 | 256 | 2 | 2 | 1.44 GB | 9.10 GB | 9.8 GB | ~5 s |
| MiniMax-M2.7-JANG_2L | `minimax_m2` | 62 | 256 | **8** | 2 | 4.03 GB | 63.26 GB | 63 GB | 1:38 |

Bundle locations:
- Gemma: `/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec`
- Qwen:  `/tmp/jangcore-fixtures/qwen35-a3b.jangspec`
- MiniMax: `/Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec`

MiniMax is the real-world target: 228 B params, 15 872 expert blobs, 4.03 GB hot core that fits on any M-series Mac.

---

## Spike A — `MTLIOCommandQueue` benchmark

Machine: **Apple M4 Max**
Config: 256 × 50 MB files, random-order reads into unified-memory `MTLBuffer`s.

| Test | Throughput | p50 | p99 |
|---|---|---|---|
| MTLIOCommandQueue sequential | 11.85 GB/s | 4.38 ms | 4.96 ms |
| MTLIOCommandQueue random | **12.54 GB/s** | 4.15 ms | **4.74 ms** |
| pread random (warm cache) | 20.04 GB/s | 2.57 ms | 2.75 ms |

**Verdict: GO.** Random access is ~4× the 3 GB/s design threshold, p99 under the 5 ms ceiling. Random ≈ sequential on MTL path, which is the critical signal — reading experts in router-decided order is not penalized. Full report: `docs/superpowers/notes/2026-04-13-iobench-results.md`.

---

## Tests (running totals)

- **Python:** 14 tests under `jang-tools/tests/jangspec/` — unit (blob, index, manifest, tier) + integration (builder, reader round-trip against Gemma fixture)
- **Swift:** 11 tests under `jang-runtime/Tests/JANGCoreTests/` — unit (format, index, blob) + integration (bundle open, expert load against Gemma fixture)
- **Swift benchmark:** `jang-spec-iobench` (standalone, not an XCTest)

All green as of last run.

---

## Gotchas logged

Things discovered during implementation that future plans must remember:

1. **Swift 6.2 on arm64e traps on misaligned `load`.** Use `raw.loadUnaligned(fromByteOffset:as:)` — not `load(fromByteOffset:as:)` — when reading u32/u64 from packed `Data` via `withUnsafeBytes`. Affects any on-disk format parsing.
2. **SwiftPM refuses empty targets.** A product whose backing target has no `.swift` files fails `swift package describe`. Always ship at least a stub source file when registering a new library or executable target.
3. **`test_*` is globally gitignored.** `git add -f` is required for test files under `tests/jangspec/`. Non-test files in the same directory add normally.
4. **Apple Python on macOS blocks `pip install -e .`** via PEP 668. The editable install from Plan 0's scaffold is sufficient for later tasks because new files under an installed package are picked up via path.
5. **MiniMax tensor names are nested.** `model.layers.N.block_sparse_moe.switch_mlp.gate_proj.*` instead of the shallower `model.layers.N.switch_mlp.gate_proj.*` that Gemma/Qwen use. The Python tier regex `\.switch_mlp\.(gate_proj|up_proj|down_proj)\b` and the layer regex `\.layers\.(\d+)\.` both handle this without changes.

---

## Known divergence from DFlash

This is what makes jang-spec a new thing, not a DFlash port:

| Dimension | DFlash | jang-spec v1 |
|---|---|---|
| Target location | VRAM-resident | SSD-streamed |
| Drafting | Block diffusion, iterative refinement | Medusa masked-prediction heads, one forward pass |
| Platform | CUDA (vLLM, SGLang) | Swift + Metal unified memory |
| Bundle format | Standard HF safetensors | Purpose-built `.jangspec` with 4 KB-aligned expert blobs and flat index |
| Prefetch | N/A | Prior-driven (coactivation + transition) |

v2 will add router-aware drafting as a distinct ML contribution once v1 proves out.

---

## Immediate next

**Plan 3: Hot-core loader.** Pure Swift, no Metal yet.
- Parse v2 MLX-native safetensors (`uint32` qweight + `float16` scales/biases, mixed bit widths per tensor)
- Infer per-tensor bit width from shape ratios (no metadata needed — same trick MLX loader uses)
- Load `target/hot_core.safetensors` from a bundle into typed Swift handles
- Integration test against the real MiniMax bundle: verify every hot-core base name is present, with the right shape and inferred bit width
- `jang-core hot-core` CLI subcommand prints a per-tensor summary

Plan 4 adds the Metal dequant/matmul kernel on top of these handles.

Plans 5–9 build up dense → MoE → streamed → drafted → shipped.
