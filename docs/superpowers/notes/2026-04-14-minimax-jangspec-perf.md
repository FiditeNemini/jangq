# MiniMax-M2.7-JANG_2L on vmlx — decode perf baseline + fat-layout bundle fix

**Date:** 2026-04-14
**Hardware:** M4 Max, 128 GB unified memory
**Runtime:** vmlx-swift (in-tree Sources/vMLX* targets at `/Users/eric/vmlx/swift/`)
**Model:** MiniMax-M2.7-JANG_2L
  - 228 B params, 62 layers, 256 experts/layer, top-8 routing
  - 2-bit mixed-precision JANG format, ~63 GB on disk, ~30 GB working set

## TL;DR

- **Source directory load:** 19 s wall, 55 GB resident, **14.35 tok/s decode**
- **Bundle (fat layout) load:** 13 s wall, 56 GB resident, **14.23 tok/s decode**
- **Bundle (old per-blob layout):** 1:20+ wall, peaked at 89 GB RSS, heavy compressed-memory thrashing, decode never measured
- **Decode parity:** bundle is within 1% of source after the fat-layout fix
- **Bundle loads 31% faster** than source (fewer files to mmap, Metal shaders compile once)

The old per-blob bundle layout was actively *hurting* us on memory-constrained machines because it required restacking 256 per-expert MLXArrays into 3D tensors at load time, peaking at ~3× the steady-state size. The fat layout writes the experts as pre-stacked 3D tensors in a single `experts.safetensors` at the bundle root, and the loader mmaps it directly.

Once loaded, both paths feed identical weights into identical model code and identical Metal kernels, so **the bundle path cannot be meaningfully faster than the source path on decode alone** — they are both bottlenecked by the same mixed-bitwidth MoE kernels.

## What "faster than minimax itself" means after this measurement

The goal of beating source-dir decode via the bundle path is misframed. With the fat layout, they converge to parity. Any decode speedup now has to come from inference-path optimizations that apply to *both* paths equally:

1. **Compile fusion** (`mx.compile(shapeless: true)`) — free gain if it's not already on for MiniMax in the serve path. Typically 15–25% on mlx-swift decode. Under investigation (task #36).
2. **JANGTQ P13+ Metal kernel stack** — extension of the existing P1-P12 optimization work in `Sources/Cmlx/mlx-generated/metal/`. Weeks of work, applies directly to both paths.
3. **Cold expert pruning / gate softmax truncation** — skip top-k experts with routing weights below a threshold. ~1% quality cost for potentially 20–40% speedup.
4. **Metal-resident draft model + speculative decoding** — the Plan 8 Medusa idea with ANE removed (see §ANE verdict below). 5–6 week scope, ceiling ~3× on decode.

Plan 7 (decode optimization) covers these levers explicitly.

## How the bundle loader now works

Two code paths in `vMLXLMCommon/JangSpecBundleLoader.swift`:

### Fat layout (preferred, built by every new bundle)

`<bundle>/experts.safetensors` — one consolidated safetensors file holding every expert tensor under its original name in original 3D stacked `[E, ...]` form. Swift loader mmaps it via the existing `loadArraysAndMetadata` helper, hands the resulting dict to the model factory unchanged. Zero restack cost, zero transient RAM peak.

### Per-blob streaming layout (legacy, still readable)

`<bundle>/target/experts.jsidx` + `experts-NNNNN.bin` — one blob per `(layer, expert_id)`. Swift loader walks the flat index, mmaps each shard, parses per-expert blob headers, calls `MLX.stacked(_:axis: 0)` per layer base. Used only when `experts.safetensors` is absent. Peaks at ~3× RAM during load because of the per-expert temporaries.

The builder currently writes BOTH formats (126 GB total for MiniMax). Task #35 will default to fat-only with `--streaming` opt-in for the per-blob layout.

## Measurements (M4 Max, 113 GB free at test time)

### Source directory path

```
model: /Users/eric/models/MiniMax-M2.7-JANG_2L/
load wallclock:          ~19 s  (from "Loading" → "Ready")
resident RSS after warmup: ~55 GB
prompt:                  60 tokens ("Write a short paragraph about the Roman Empire...")
generated:               256 tokens
generation wallclock:    17.84 s
decode tok/s:            14.35
```

### Bundle fat-layout path

```
bundle: /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec/
  fat file: experts.safetensors (63.19 GB)
  plus:     hot_core.safetensors (4.03 GB)
  plus:     target/experts-*.bin per-blob format (legacy copy, 63 GB, unused by loader)

load wallclock:          ~13 s  (from "Loading" → "Ready")
resident RSS after warmup: ~56 GB
prompt:                  60 tokens (same)
generated:               256 tokens
generation wallclock:    17.99 s
decode tok/s:            14.23
```

### Bundle per-blob path (pre-fat-layout)

```
load wallclock:          >1:20 (process hit SIGSTOP via memory pressure, never completed)
resident RSS peak:       89 GB during restack phase
compressed memory peak:  77 GB (OS paging out to compressed memory)
observations:            all 62 layers restacked via MLX.stacked per base name.
                         autoreleasepool boundary per-layer bounded the transient
                         peak but the steady-state from ~62 stacked tensors plus
                         in-flight MLX allocation metadata pushed us past physical
                         RAM on a machine that had 60 GB free.
```

### Gemma-4-26B-A4B (smaller MoE, for scale calibration)

Same harness, run earlier in the session before the fat-layout fix:

```
model:  /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec (old layout)
(pre-fat layout worked for Gemma because the 16 GB bundle fit in RAM)
load:   ~30 s
prompt: 50 tokens
gen:    188 tokens (natural EOS stop)
wall:   4.77 s
tok/s:  39.4
```

Gemma runs ~2.8× faster than MiniMax per token because:
- 30 layers vs 62 (2×)
- 4-bit MoE vs 2-bit mixed (2-bit kernels are slower per-op on Metal)
- smaller intermediate size per expert

## Swift code changes landed this session

Files modified in `/Users/eric/vmlx/swift/` (all uncommitted, vmlx git repo has no commits yet):

- `Sources/vMLXLMCommon/JangSpecBundleLoader.swift` — **NEW** file, 470+ lines. Bundle format detection, manifest/index/blob parsers, hot-core mmap, fat-layout direct mmap, per-blob fallback restack path. All autoreleasepool-bounded.
- `Sources/vMLXLMCommon/Load.swift` — one-branch insertion at `loadWeights()`: `if JangSpecBundleLoader.isBundle(at: modelDirectory) { weights = try JangSpecBundleLoader.loadWeights(...) }`. Runs before the existing JANG v1 / safetensors enumeration branches.
- `Sources/vMLXLMCommon/JangLoader.swift` — `findConfigPath` defensive fallback into `<root>/target/` for older bundles that only had configs under target/.
- `Sources/vMLXLMCommon/Evaluate.swift` line 768 — pre-existing type-inference breakage on `MLXArray([[Int32(...)]])`. Fix: use `MLXArray([Int32(...)])[.newAxis]`. Unblocked full vmlx build. Not my original bug.
- `Sources/vMLXLMCommon/BatchEngine/BatchEngine.swift` line 440 — same pre-existing fix.

Files modified in `/Users/eric/jang/jang-tools/`:

- `jang_tools/jangspec/builder.py` — `_write_experts_and_index` now also collects every expert tensor in its original 3D stacked form and writes `experts.safetensors` at the bundle root via `save_file`. The per-blob write path is preserved. `_copy_tokenizer` copies every non-safetensors file from the source model dir (not a hardcoded list) so VLM models pick up `processor_config.json`, `chat_template.jinja`, `modeling_*.py`, etc. without per-architecture carve-outs.

## ANE feasibility spike — VERDICT: no-go

Downloaded `anemll/anemll-Llama-3.2-1B-FP16-iOS` (a 1 B fp16 Llama precompiled for Core ML on ANE) and benchmarked it three ways on this M4 Max. Real numbers, same model, same Core ML graph:

| Per-token chunk | **ANE** (cpuANE) | **Metal** (cpuGPU) | auto-route |
|---|---:|---:|---:|
| embeddings | 0.064 ms | 0.266 ms | 0.309 ms |
| ffn_chunk_01 | 34.87 ms | 10.95 ms | 34.63 ms |
| ffn_chunk_02 | 34.70 ms | 8.48 ms | 33.89 ms |
| lm_head | 4.49 ms | 1.78 ms | 4.50 ms |
| **total** | **74.12 ms** | **21.48 ms** | 73.33 ms |
| **decode tok/s** | **13.5** | **46.6** | 13.6 |

**Metal is 3.45× faster than ANE** on the same 1 B fp16 Llama. Core ML's auto-router (`.all`) picked ANE for the FFN chunks and matched the ANE number — Core ML's heuristic ALSO gets this wrong, defaulting to the slow path when both engines are available.

### Why ANE lost

- ANE can't run JANG's variable-bit-width quantized weights natively. To use ANE at all we'd have to dequantize to fp16, throwing away the memory savings that make JANG interesting in the first place.
- Dynamic MoE dispatch (top-k per token, 256 experts) doesn't map to ANE's static-graph model.
- Core ML's implementation of SDPA and attention primitives for transformers has ops that fall back to CPU/GPU mid-kernel on ANE, costing round-trip bandwidth.
- Even the "best case" fp16 dense transformer loses to a well-tuned Metal path on this hardware.

### The ANE opportunity that might still matter (deferred)

Plan 8's "speculative decoding with ANE-hosted draft" idea dies as originally framed, but the **same Plan 8 architecture with a Metal-hosted draft** is still viable. A small dense JANG model running at 80–120 tok/s on Metal + Medusa-style block heads + the MiniMax target at 14 tok/s = effective ~100+ tok/s if accept rate is ≥ 60% on a B=8 block. That's the realistic path to beating MiniMax source decode. 5–6 weeks scope.

Full ANE-bench harness preserved at `/tmp/ane-bench.swift` + `/tmp/ane-one.swift` for future re-tests on other precompiled packages (anemll-Llama-FAST-iOS, LUT8 variants, etc.).

## Immediate next steps (Plan 7 execution)

### ✅ Done this session

1. **Default bundle builder to fat-only.** `--streaming` opt-in flag added for the per-blob layout. Saves 63 GB per MiniMax bundle. Jang repo `jang-spec-plan5-bundle-python-validation` branch commit `58f21c0`.
2. **Compile-decode fusion investigation.** `HardwareInfo.isCompiledDecodeSupported` was hardcoded `false` across all hardware due to a macOS Tahoe Metal JIT bug (MLX #3329/#3201/#3256). Added a `VMLX_FORCE_COMPILE_DECODE=1` env override in `Sources/vMLXLMCommon/HardwareInfo.swift`. **M4 Max does NOT hit the Tahoe bug** — compile-decode runs to completion with no crash. Measured effect on MiniMax-M2.7-JANG_2L decode: **14.23 → 14.42 tok/s (+1.3%, within noise).** The SwiGLU/GeGLU activation-level fusion is not where the time is spent. Conclusion: compile fusion is a free small win on M4 Max but not the "beat minimax itself" move. The actual bottleneck is the quantized MoE expert matmul kernel.

### 🔜 Remaining levers

3. **L2 — Cold expert pruning.** Threshold-drop experts with routing weight below (say) 5% of the max. Predicted 20–40% decode speedup at <1% MMLU cost. This is the highest-ROI item left on the menu and applies equally to bundle and source paths.
4. **L3 — JANGTQ P13+ Metal kernel extensions.** 2-bit-specific gather_qmm variant, fused expert dispatch, activation caching. Weeks of work but targets the actual decode bottleneck.
5. **Plan 8 — Metal-resident speculative decoding.** Deferred to its own plan. Ceiling ~3× on effective decode. 5–6 week scope.
6. **Long-term parity check** — once decode is optimized, rebenchmark bundle vs source to confirm they stay at parity.

### Key insight from the compile-decode experiment

The decode hot path for a JANG MoE model is NOT dominated by activation ops — it's dominated by **the quantized-matmul kernel calls for the 8 active experts per token**. Fusing SwiGLU (2 Metal dispatches collapsed to 1) across 62 layers saves ~60 dispatches per token out of a total count dominated by 62 × 8 × 3 = ~1500 quantized-matmul dispatches. The 1.3% measured delta is consistent with that math.

**What would actually move the needle** at the kernel level:
- Fuse the 8-way gather+matmul inside a single dispatch (currently launches separate `gather_qmm` per expert call)
- Reuse dequantized expert weights across consecutive decode steps where the router picks the same expert
- A 2-bit-specific kernel that avoids the variable-bit dispatch overhead of the current mixed-bit `gather_qmm`

These are all P13+ kernel work — not config flags, not loader changes, not compile-fusion tweaks.

## Parking lot — not today

- MiniMax `.jangspec` per-blob SSD streaming path. The per-blob format is preserved in the bundle for a future streaming runtime, but won't be the default load path. If we ever need to run a model that doesn't fit in unified memory, the per-blob + `MTLIOCommandQueue` streaming path from the original jang-spec design is still on the table.
- ANE retry with a hand-tuned Core ML conversion. Possible ~2× improvement over anemll's pipeline, but ceiling is still slower than Metal for this workload.
- Gemma-4-26B rebench on the new fat layout. Low priority — Gemma already worked with the old layout.
