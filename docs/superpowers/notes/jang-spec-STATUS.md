# jang-spec — Running Status

Last updated: 2026-04-13

**Single source of truth for what's shipped, what's in progress, and what's next on the jang-spec project.** Every plan updates this file in its final commit.

---

## TL;DR

SSD-streamed MoE speculative decoding for JANG models. v1 = streaming + Medusa drafting + prior-based prefetch. Plans 1–4 done — Plan 5 (dense forward pass) is next. Three real MoE bundles built and verified (Gemma-4-26B, Qwen3.5-35B-A3B, MiniMax M2.7 228B). Streaming IO spike GO on M4 Max (12.54 GB/s random). Hot-core loader parses JANG v2 safetensors and classifies every tensor in the MiniMax hot core in ~0.5 s via mmap. First Metal compute kernel (4-bit GEMV) now validated against an MLX reference within fp16 drift (max abs 7.5e-3 vs 1e-2 bound).

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
| 3 | Hot-core loader (Swift, v2 safetensors, no Metal yet) | **DONE** | `jang-spec-plan3-hotcore` | `SafetensorsV2File`, `BitInference`, `QuantizedTensorView`, `HotCoreLoader`, `jang-core hot-core` CLI |
| 4 | Metal v2 quantized matmul kernel (4-bit GEMV) | **DONE** | `jang-spec-plan4-metal-matmul` | `JANGCoreMetal` library, `JangV2QuantMatmul.metal`, `QuantizedMatmul4` |
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
- **Swift:** 20 tests total — 19 under `jang-runtime/Tests/JANGCoreTests/` (format, index, blob, safetensors v2, bit inference, bundle open, expert/hot-core load) + 1 under `jang-runtime/Tests/JANGCoreMetalTests/` (4-bit GEMV correctness vs MLX reference)
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

**Plan 5: Dense JANG v2 forward pass.** Wire `QuantizedMatmul4` into a real forward pass on a small dense JANG model, using it for all linear layers (q/k/v/o projections, MLP gate/up/down).
- Load a tiny dense JANG v2 bundle via `SafetensorsV2File` + `HotCoreLoader`
- Replace Plan 4's synthetic fixture with real per-layer projections and validate against an MLX reference forward
- Extend `JANGCoreMetal` with the companion kernels (RMSNorm, SiLU, residual add, rope) as needed
- CLI surface: `jang-core forward <bundle>` runs one token end-to-end and prints logits
- Plan 6 generalizes this to MoE with a router + switch_mlp gather variant

Plans 6–9 build up MoE → streamed → drafted → shipped.

### Plan 4 notes

- First Metal compute code in the repo. New `JANGCoreMetal` library target depends on `JANGCore`, ships `JangV2QuantMatmul.metal` as an SPM resource, and exposes `MetalContext`, `MetalBuffer`, and `QuantizedMatmul4`.
- Test: `QuantizedMatmul4Tests.testMatchesMLXReference` loads a committed safetensors fixture generated by `jang-tools/scripts/gen_matmul_fixture.py` (deterministic `mx.quantize` on a 128×64 weight matrix, group_size 64) and compares the kernel output against `mx.dequantize`-based reference. **Measured: max abs 7.492e-3, max rel 5.266e-3 on M4 Max, well under the 1e-2 bound.** The residual is fp16 scale/bias drift — MLX computes dequant in fp32 while the kernel reads `half` scales/biases.
- Kernel is deliberately naive (1 thread per output row, no SIMD/threadgroup tricks). Correctness before perf. The 4-bit LSB-first packing convention is now verified end-to-end against MLX — Plan 5 can safely build on it without re-deriving the bit layout.
- **SwiftPM 6.2 does not auto-compile `.metal` files into a default metallib for library targets** (that's an Xcode-only build rule). Workaround: the `.metal` file is shipped as a plain `.copy` resource and compiled at runtime via `device.makeLibrary(source:)` (~1 ms on first use). `MetalContext` tries `makeDefaultLibrary(bundle:)` first so a future SPM that adds native support will pick it up transparently.
- **Metal resource path:** the plan's preferred layout `jang-runtime/Metal/` uses a `..` path from inside `Sources/JANGCoreMetal` which SwiftPM 6 rejects for safety. Fallback co-located layout (`Sources/JANGCoreMetal/JangV2QuantMatmul.metal`) is used instead and works cleanly. Plan 5 should keep future kernels in the same directory.
- Release build (`swift build -c release`) produces `jang`, `jang-spec-iobench`, `jang-core`, `JANGCore`, and `JANGCoreMetal` cleanly.

### Plan 3 notes

- `SafetensorsV2File` is a pure-Foundation mmap reader, completely independent of the existing v1 reader in the `JANG` target. Uses `raw.loadUnaligned` for the u64 header size to dodge Swift 6's misaligned-load trap.
- `BitInference.infer(qweightShape:scalesShape:groupSize:)` works for both 2D dense and 3D expert-stacked shapes; the rule only looks at the last two axes.
- `HotCoreLoader.load(bundle:groupSize:)` groups `{base}.weight/.scales/.biases` triples into `QuantizedTensorView`s and emits everything else (norms, lone biases) as `RawTensorView`s. Returned handles are zero-copy slices into the mmap.
- **MiniMax uses `block_size: 128`**, not 64. The CLI takes `--group-size 128` for JANG_2L; the test helper reads `target/jang_config.json` to pick the right size automatically. Plans 4+ should plumb group size through from the manifest / config rather than assume 64.
- Hot-core summary from `jang-core hot-core --group-size 128 /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec`: 250 quantized base tensors (249 × 8-bit attention/output + 1 × 6-bit embedding) + 373 raw tensors (norms, routers) = 4.03 GB, matching `jang-core inspect`.
