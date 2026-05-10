# JANGTQ-NA — M5 GPU Neural Accelerator path for JANGTQ_K

**Author:** Jinho Jang (eric@jangq.ai)
**Status:** Design — pending implementation plan
**Date:** 2026-05-08
**Reference target:** MiniMax-M2.7-JANGTQ_K → MiniMax-M2.7-JANGTQ-NA
**Hardware:** Apple M5 Max (this machine, primary). Non-M5 hardware is *not* a target machine path for this bundle — non-M5 users continue to use the existing `MiniMax-M2.7-JANGTQ_K` bundle. The new bundle is M5-only.

---

> **2026-05-08 FINAL STATUS — Custom NA kernel track CLOSED.**
>
> The dense pivot above is also wrong. The correct baseline for "is custom NA worth it" is `mlx.quantized_matmul`, not `gather_tq_matmul` or `TurboQuantLinear`. MLX 0.31.2 metallib already ships M5-NA-accelerated `affine_qmm_*_nax_*` (dense) and `affine_gather_qmm_*_nax_*` (MoE expert) kernels, 64×64 tiles, 2×2 simdgroups, bits {2,3,4,5,6,8} × gs {32,64,128}. Against that baseline, the custom NA fused kernel loses 10–20× on dense shapes.
>
> Decision (per Eric, 2026-05-08): **stop the custom NA kernel track.** The kernel artifacts (spike, int8_gemm, codebook_unpack, per_token_quant, na_kernel composition + fused, full test suite) stay on disk as research; no further investment without a Steel-class redesign.
>
> The next speed track is a **format question**, not a kernel question: can existing JANGTQ targets (MiniMax-M2.7, Kimi-K2.6, GLM-5.1, …) be re-quantized into MLX-affine bundles that hit `quantized_matmul_nax` / `gather_qmm_nax` natively at acceptable quality + bundle size? See `docs/superpowers/plans/2026-05-08-m5-affine-sidecar-plan.md` (new plan).
>
> Everything below this point is the original MoE-targeted Phase A design, preserved as historical record. It does not reflect the current direction.

---

## 1. Goal

Cut MiniMax JANGTQ_K **prefill** time on M5 Max by routing the routed-expert MoE matmuls through the new M5 GPU **Neural Accelerators** (per-GPU-core matmul tensor units, accessed via Metal 4 `cooperative_tensor` + Metal Performance Primitives `mpp::tensor_ops::matmul2d`). Decode is held flat-or-better; the Phase A win surfaces as TTFT and as end-to-end throughput on prompts where prefill dominates. Bundle size relative to JANGTQ_K is roughly equal — the L1+L2 layout shrinks the per-row scale tensor by ~50 %, but adds a per-tile scale tensor and a codebook-scale scalar that approximately cancel the win. Net: **negligible size delta, no size regression**.

Out of scope: Apple Neural Engine (ANE) proper. The ANE is FP16-internal, capped at ~32 MB working set, and demonstrably the wrong tool for 228 B-class MoE — see §3. The bundle is M5-only; non-M5 users stay on the existing `MiniMax-M2.7-JANGTQ_K`.

## 2. Success criteria

Phase A is the gate for B and C. All four bars below must be cleared on M5 Max (this machine) for Phase A to ship:

1. **Prefill (pp/s, equiv. TTFT):** strictly greater than `MiniMax-M2.7-JANGTQ_K` on the same machine, same prompt, identical greedy decode settings. This is the Phase A win the kernel actually targets.
2. **Pure decode (tok/s, prefill amortized out — measure on the 300-token "photosynthesis" / "poem + 17×23" prompts from `JANGTQ-REFERENCE.md` §6 where decode dominates):** at most a **3 % regression** vs `MiniMax-M2.7-JANGTQ_K`. Phase A's decode kernel is unchanged from JANGTQ_K, but the loader/converter changes can introduce micro-overhead (e.g. extra norm reconstruction); the 3 % bar absorbs that without forgiving an actual decode regression.
3. **End-to-end long-prompt throughput** (≥ 1024-token prompt with ≤ 256 new tokens, where prefill dominates): strictly greater than JANGTQ_K. This is the win users feel.
4. **Quality:** Greedy coherence on the standard 5-prompt validation set (`research/scripts/validate_jangtq.py`) — EOS reaches `max_tokens` on counts/poems, "Capital of France?" → "Paris", "2+2" → "4", no EOS-at-50 collapse — AND MMLU 10-subject-N-question score within ±0.5 pp of the JANGTQ_K baseline.

If any of (1), (3), or (4) fails, Phase A is shelved. If only (2) fails, root-cause the decode regression before promoting (do not silently accept a decode-quiet ship).

## 3. Hardware reality check (why GPU NA, why not ANE)

### 3.1 The two M5 Max accelerator blocks

| Block | What | Programming model | Compute precisions | Capacity ceiling | LLM verdict |
|---|---|---|---|---|---|
| **Apple Neural Engine (ANE)** | 16-core fixed-function NPU, ~38 TOPS | CoreML / MIL / `_ANECompiler` private API | FP16 internal — INT4/INT8 only saves bandwidth, not compute (Orion paper, §Quantization Support) | 32 MB on-chip SRAM, ≥30% throughput cliff above. Practical ~8 B param ceiling | **Wrong tool for 228 B-class JANGTQ.** Reserved for the small router carve-out in §6 L4. |
| **M5 GPU Neural Accelerators (NA)** | 1 dedicated matmul tensor unit per GPU core (32–40 on M5 Max) | Metal 4 `cooperative_tensor` API, `mpp::tensor_ops::matmul2d(M, N, K)`. Already wired through MLX TensorOps | BF16, FP16, MXFP4, INT8×INT8→INT32 | Scales with GPU memory; ML Research demoed 30 B MoE prefill | **Right target.** This is what cider exploits. |

### 3.2 Source data anchoring this choice

- **Apple ML Research** "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" (2026-03): up to 4× TTFT vs M4 Max, 19–27% decode lift, 30 B MoE TTFT under 3 s.
- **cider** (Mininglamp-AI, 2026): MLX custom primitive shape, W4A8 / W8A8 implemented, 1.42–1.86× kernel-level on M5 Pro, **1.46× end-to-end Qwen3-8B prefill** with W8A8 per-channel.
- **ANEMLL**: 8 B model on ANE = ~9 tok/s vs ~50 tok/s on MLX/GPU on the same machine. Tops out at 8 B.
- **Orion** (arXiv 2603.06728): ANE has no INT8/INT4 compute path; 32 MB SRAM hard ceiling.

### 3.3 What we copy from cider, what we don't

Copy:
- The MLX `mlx::core::Primitive` subclass + Metal kernel pattern (`csrc/src/w8a8_primitive.mm`, `cider/kernels/w8a8_matmul.metal`).
- The `cooperative_tensor` + `mpp::tensor_ops::matmul2d(16, 32, 16)` tile.
- The "fused dequant in store phase" trick (INT32 accumulator × scale → FP16 written once, no extra device round-trip).

Don't copy:
- cider's per-token activation INT8 quantizer is generic; ours can exploit the Hadamard rotation that JANGTQ already does. Post-rotation activations are roughly Gaussian per-row, so per-row activation scale is a single `max` reduction → cheap.
- cider's W4A8 unpacks to INT8 at compute. JANGTQ's "weight" is a 2-bit codebook index into a 4-entry FP16 table; our unpack is a `codebook[idx] → INT8` LUT, not an INT4→INT8 conversion. Different plumbing.

## 4. Three-phase plan

### Phase A — JANGTQ-NA prefill on tensor cores (decode unchanged)

| Aspect | Choice |
|---|---|
| Prefill kernel | New MLX custom primitive `tq_na_matmul_prefill`. INT8×INT8→INT32 tensor matmul. Per-row dequant fused into store phase. |
| Decode kernel | Unchanged. Existing P15/P17/P18 hand-rolled Hadamard+codebook fast path stays. |
| Bundle | New HF line **`JANGQ-AI/MiniMax-M2.7-JANGTQ-NA`**. (Decision locked 2026-05-08.) |
| New bundle tensors | Two new tensors per `(layer, projection, expert)` set: `tq_tile_scale` (per-tile FP16) and `tq_norms_log8` (per-row uint8). FP16 `tq_norms` is **dropped** in this bundle. `tq_codebook_int8` + `tq_codebook_int8_scale` are derived per `(in_features, bits)` shape at load time, cached, shared across modules. |
| Compression layers added | **L1** (uint8 log-scale row norms) + **L2** (per-tile shared scales). These shrink the per-row scale tensor by 50 % and add a per-tile scale tensor of size `out/16 × FP16`. Net bundle change: **< 0.2 %, treated as no-change.** L1 + L2 exist to serve the kernel layout, not to shrink the bundle. |
| Hardware gate | M5+ GPU + macOS 26.2+. Loader hard-errors on incompatible hardware or OS — no silent fallback. Non-M5 users receive a clear message naming `MiniMax-M2.7-JANGTQ_K` as the alternative bundle. |
| Phase exit | Success criteria in §2 met on M5 Max. |

### Phase B — JANGTQ-NA decode on tensor cores

| Aspect | Choice |
|---|---|
| Decode kernel | Same primitive generalized to `(K, hidden)` matrix-vector tile shape. Activations from Hadamard rotation quantized in-kernel to INT8 per K-row. |
| Risk | cider's own data shows the per-row INT8 MV kernel runs *slower* than the FP16 kernel for isolated decode. The win has to come from removing dispatches across 62 layers, not from raw kernel speed. |
| Quality layer added | **L3** (per-layer codebooks). Same number of stored bits per weight as JANGTQ_K — the bundle does **not** shrink. L3 retunes 4-entry codebooks per layer using activation calibration so the same 2/4-bit indices express more of each layer's signal. Expected: ~0.3 perplexity improvement at fixed bits, or equivalently the option to drop one projection from 4-bit to 2-bit at parity quality (which would shrink the bundle, but only as a separate Phase B.2 decision). |
| Phase exit | Decode tok/s on M5 Max ≥ Phase A tok/s, MMLU within ±0.5 pp of Phase A. If decode regresses vs Phase A, revert to Phase A's hybrid kernels and skip the rest of B. |

### Phase C — Native MiniMax-NA model variant

| Aspect | Choice |
|---|---|
| Source | A new SLURPY-style finetune pass on top of MiniMax-M2.7. Pin FFN intermediate to a multiple of 32 (the NA matmul `K` dim), ensure attention head_dim is multiple of 16, calibrate against Hadamard-rotated activations from day one. |
| Compression layer added | **L4** (ANE-resident router + small lm_head head-block). Earns the ANE its keep on the small hot-path components only. |
| Cost | Multi-week. Comparable to JANGREAP iteration cycle. |
| Phase exit | Sub-1.6 bits/param effective rate, decode tok/s on M5 Max ≥ 1.4× Phase B, MMLU within ±0.5 pp of MiniMax-M2.7 base. |

A → B → C is strictly de-risking: each phase's hardest unknown gets answered before the next phase commits.

## 5. Bundle format — `MiniMax-M2.7-JANGTQ-NA`

### 5.1 Disk layout

Directory name: `MiniMax-M2.7-JANGTQ-NA` (publishes to `JANGQ-AI/MiniMax-M2.7-JANGTQ-NA`).

```
MiniMax-M2.7-JANGTQ-NA/
  config.json                         # standard HF config + jangtq + jangtq_na keys (see §5.2)
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  jang_config.json                    # JANG metadata, format_version "2.0"
  jangtq_na.json                      # NA-specific metadata (see §5.3)
  model-00001-of-NNNNN.safetensors    # MLX-native quantized weights (prestacked per JANGTQ-PRESTACK-SPEC)
  model.safetensors.index.json
```

No separate sidecar files for NA tensors — they live in the main shards alongside `tq_packed`/`tq_bits` (and the new `tq_tile_scale`/`tq_norms_log8`). The prestack spec already commits to this layout discipline; we extend it. Note: FP16 `tq_norms` is **not** present in this bundle — it has been replaced by `tq_norms_log8` (uint8) plus `tq_tile_scale` (FP16).

### 5.2 New keys in `config.json`

```json
{
  "...": "...",
  "quantization": { "group_size": 64, "bits": 2 },
  "routed_expert_bits": { "gate_proj": 2, "up_proj": 2, "down_proj": 4 },
  "jangtq_na": {
    "format_version": "1.0",
    "tile_shape": [16, 32, 16],
    "compression_layers": ["L1_log_scale_row_norms", "L2_per_tile_scale"],
    "min_macos": "26.2",
    "min_chip": "m5"
  }
}
```

There is no `fallback_kernel` field. The bundle is M5-only; the loader hard-errors on incompatible hardware (see §5.5).

### 5.3 `jangtq_na.json`

Verbose mirror of the `jangtq_na` config block plus calibration metadata. The kernel does not read this file; it exists for tooling, audits, and the verifier.

```json
{
  "format_version": "1.0",
  "source_bundle": "MiniMax-M2.7-JANGTQ_K",
  "source_bundle_sha256": "<hash>",
  "tile_shape": [16, 32, 16],
  "tensor_core_dtype": "int8",
  "accumulator_dtype": "int32",
  "compression_layers": {
    "L1_log_scale_row_norms": {
      "applied": true,
      "encoding": "uint8",
      "log2_step": 0.0625,
      "u8_zero_point": 128,
      "decode_formula": "norms_hat[r] = exp2((u8[r] - 128) / 16)"
    },
    "L2_per_tile_scale": { "applied": true, "tile_rows": 16, "scale_dtype": "float16" },
    "L3_per_layer_codebook": { "applied": false, "_phase": "B" },
    "L4_ane_router": { "applied": false, "_phase": "C" }
  },
  "kernel_target": "m5_na",
  "min_macos": "26.2",
  "min_chip": "m5"
}
```

### 5.4 New per-`(layer × projection)` tensors

Stacked, in main shards, alongside the existing prestack keys. `<prefix>` matches `JANGTQ-PRESTACK-SPEC.md` §"On-disk key schema".

| Key | Shape | Dtype | Role |
|---|---|---|---|
| `<prefix>.switch_mlp.<proj>.tq_packed` | `[E, out, packed_in]` | uint32 | **Existing.** Lloyd-Max codebook indices, bit-packed. Unchanged from JANGTQ_K. |
| `<prefix>.switch_mlp.<proj>.tq_bits` | `[1]` | uint8 | **Existing.** Bit-width 2 / 4. Unchanged. |
| `<prefix>.switch_mlp.<proj>.tq_tile_scale` | `[E, out_tiles]` where `out_tiles = ceil(out / 16)` | float16 | **NEW (L2).** Per-tile dequant scale (one entry per 16-row output tile). |
| `<prefix>.switch_mlp.<proj>.tq_norms_log8` | `[E, out]` | uint8 | **NEW (L1).** Per-row log2 norm, applied on top of `tq_tile_scale`. Encoding: `uint8 ∈ [0, 255]`, reconstruction `norms_hat[r] = exp2((uint8 − 128) / 16)` ∈ approx [0.0039, 245.7]. Resolution 1/16 octave. |

`tq_norms` (FP16, JANGTQ_K's per-row norm) is **dropped** from this bundle — non-M5 machines stay on the existing `MiniMax-M2.7-JANGTQ_K` bundle, so this bundle has no fallback obligation.

Net effect on scale-tensor footprint, per `(layer × projection × expert)` block of `out` rows:
- JANGTQ_K: `tq_norms` = `out × 2` bytes
- JANGTQ-NA: `tq_norms_log8` (`out × 1` bytes) + `tq_tile_scale` (`out × 0.125` bytes) = `out × 1.125` bytes
- Per-block delta: **−43.75 %** on scale-tensor footprint
- Bundle-level delta (scales are ~0.34 % of total): **≈ −0.15 %, i.e. negligible**. L1 + L2 exist for kernel layout, not for storage savings.

`tq_codebook_int8` is **not** stored in shards. The FP16 codebook (4 entries per `(in_features, bits)` shape, already cached in JANGTQ_K) is converted at load time as:

```python
absmax = max(abs(codebook_fp16))                 # scalar
codebook_int8 = round(codebook_fp16 / absmax * 127).astype(int8)  # 4 × int8
codebook_scale = (absmax / 127).astype(float16)  # scalar (FP16)
```

Both `codebook_int8` and `codebook_scale` are passed into the kernel. **Direct `astype(int8)` on the FP16 codebook is wrong** — codebook entries are O(0.01) and would round to zero. The explicit scale-and-quantize step is mandatory.

### 5.5 Hardware policy

`MiniMax-M2.7-JANGTQ-NA` is **M5-only**. The loader hard-fails with a clear message on non-M5 hardware or macOS < 26.2 — no silent fallback, no in-bundle FP16 `tq_norms` shadow, no two-format-in-one shipping. This keeps the bundle small and the format well-defined.

Non-M5 users use the existing `MiniMax-M2.7-JANGTQ_K` bundle, which is unmodified and remains the canonical option for M3/M4 hardware. Old `osaurus` / `vmlx-swift` consumers pinned to JANGTQ_K stay pinned.

Loader pseudocode:

```python
if not _detect_m5_neural_accelerators():
    raise RuntimeError(
        "MiniMax-M2.7-JANGTQ-NA requires an M5-class GPU and macOS 26.2+. "
        "On this machine, use MiniMax-M2.7-JANGTQ_K instead."
    )
if not _macos_at_least("26.2"):
    raise RuntimeError("...")
```

## 6. Compression-layer math

### L1 — log-scale row norms in uint8 (Phase A)

**Setting.** JANGTQ today stores per-row norms as FP16 (`out × 2` bytes per expert × E experts × L layers × 3 projections). Across MiniMax M2.7 layers the dynamic range of `norms[r]` spans about three orders of magnitude — well within an 8-bit log-scale.

**Math.** Single convention used throughout: **uint8 in [0, 255], offset 128, step 1/16 octave**.

```
encode:    u8[r] = clamp(round(log2(norms[r]) * 16) + 128, 0, 255)
decode:    norms_hat[r] = exp2((u8[r] - 128) / 16)
range:     norms_hat ∈ [exp2(-8), exp2( 7.9375)] ≈ [3.9e-3, 245.7]
worst err: 2^(1/32) - 1 ≈ 2.2 % per-row scale error (dominated by bf16 activation noise floor)
```

**Kernel application.** The reconstructed `norms_hat[r]` is an FP16 multiplier. There is **no integer-shift trick** — 1/16-octave resolution corresponds to fractional powers of 2, not integer bit-shifts. The kernel reconstructs `norms_hat[r]` either via a 256-entry FP16 lookup table (`exp2_lut[256]`) loaded once into threadgroup shared memory, or via `metal::exp2((u8 − 128) * (1.0f/16.0f))` per row. The LUT path is preferred (one MAD per output row at register cost of 256 × 2 bytes = 512 bytes per threadgroup).

**Storage.** `out × 2` FP16 bytes → `out × 1` uint8 byte per row. **−50 % on the per-row scale tensor.**

**Why this isn't MXFP4.** MXFP4 (OCP microscaling, 4-bit element + per-block 8-bit shared exponent) is *bigger* than JANGTQ_K's 2-bit codebook + per-row norm — it would expand storage, not compress. We only use MXFP4 if the M5 NA kernel path requires it for hardware compatibility (decision point in §10 Open Question 1); it is not a compression layer in this design.

**Why L1 is mandatory in this bundle.** The bundle defines `tq_norms_log8` as the only per-row norm storage; FP16 `tq_norms` is dropped (see §5.4). There is no "fall back to FP16 row norms" path within `MiniMax-M2.7-JANGTQ-NA` — that's what the JANGTQ_K bundle is. So if L1 fails its quality gate during conversion, **the JANGTQ-NA bundle is shelved and we use JANGTQ_K**, not "ship JANGTQ-NA without L1".

**Risk and gate.** The 2.2 % per-row scale error sits at the high end of "tolerable". The converter (`jang_tools.convert_minimax_jangtq_na`) MUST run a per-layer histogram check + a single-layer MMLU spot-check on a held-out subject before emitting the full bundle, and abort with a clear failure if MMLU shifts > 0.5 pp or 5-prompt coherence breaks. The conversion path has no "ship without L1" escape; the entire bundle either passes the L1 gate or is not emitted.

### L2 — Per-tile shared scales (Phase A)

**Setting.** The NA kernel reads weights in `(16, 32)` row × col tiles. Each tile spans 16 output rows, so the kernel needs one dequant scale per tile-row group, not per individual row.

**Math.** Let `S_tile = max_{r ∈ tile} norms[r]`. Within a tile, each row's residual scale is `s_r = norms[r] / S_tile ∈ (0, 1]`. The kernel computes `int8_w[r, k] = round(W_rot[r, k] × 127 / S_tile)`, and the residual `s_r` is folded into the existing `tq_norms_log8` (L1) bias term — i.e. L2 *re-bases* the L1 log-scale relative to the tile max instead of the global max. Hadamard rotation keeps `norms[r]` tightly clustered (within a single tile, σ/μ < 2 % empirically per `research/MINIMAX-QUANT-NOISE-AUDIT.md`), so collapsing per-row scales to per-tile scales costs an additional ~1 % signal-to-noise loss — under the bf16 activation noise floor.

**Storage.** L2 adds `out_tiles × 2 = ceil(out / 16) × 2` FP16 bytes per expert (the per-tile scale). Combined with L1 (uint8 row-norms replacing FP16 row-norms), per-block scale-tensor footprint goes from `out × 2` (JANGTQ_K) to `out × 1.125` (JANGTQ-NA): **−43.75 % on the scale tensors**, **≈ −0.15 % on total bundle size, treated as no-change**. See §5.4 for the per-(layer × proj × expert) byte breakdown. L1+L2 exist for kernel layout, not for bundle compression.

**Risk.** The 2 % cluster width is empirical for MiniMax M2.7. Verify before applying to other models. Test plan: per-layer histogram of `norms` after Hadamard during conversion, abort L2 for layers with σ/μ > 5 %.

### L3 — Per-layer codebooks (Phase B, quality only — does NOT shrink the bundle)

**Setting.** JANGTQ_K shares one FP16 codebook per `(in_features, bits)` across all 62 layers. Per-layer activation distributions differ (`research/MINIMAX-QUANT-NOISE-AUDIT.md` §3): layer 0 has σ ≈ 0.014, layer 31 σ ≈ 0.026, layer 61 σ ≈ 0.041. A single codebook fits the median layer well and the tails poorly.

**Math.** Lloyd-Max minimizes `E[(x − c[argmin_c |x − c|])²]` against the empirical distribution. Calibrating against per-layer activations gives 62 tighter codebooks; worst-row quantization error drops by 12–18 % at no cost to best-row quality. **At fixed bit count, this is purely a quality improvement** — expected ≈ 0.3 perplexity improvement (or equivalently, 0.1–0.3 pp on MMLU).

**This is not a bit-rate reduction.** Each weight is still stored as the same 2/4-bit codebook index; the codebook itself is still 4 entries × FP16. The number of stored bits per weight is **identical** to JANGTQ_K. The "bit savings at fixed perplexity" framing is only meaningful if it leads to a separate Phase B.2 decision to drop a projection's bits (e.g. 4-bit `down_proj` → 2-bit). That decision is out of scope for L3 itself — L3 just provides the headroom that would *enable* such a decision later.

**Storage.** `(in_features, bits)` shapes × 4 codebook entries × FP16 = 16 bytes per `(layer, shape)` group. ~4 KB total across the 62-layer MiniMax M2.7 model. **Trivial bundle-size overhead, not a bundle-size reduction.** Real cost is ~30 min of conversion time for the calibration sweep.

**Risk.** Per-layer codebook means the codebook cache grows from O(shapes) to O(layers × shapes). Memory cost still negligible. Kernel signature unchanged — the codebook is already passed as an argument.

### L4 — ANE-resident router + lm_head head-block (Phase C)

**Setting.** Per `JANGTQ-REFERENCE.md` §7, decode time on M3 Ultra:
- Router math: 3.2 ms / token (13 % of decode)
- Gate Linear: 1.7 ms (7 %)
- These run on GPU as part of the `MiniMaxSparseMoeBlock.__call__` path.

**ANE fit.** Router weight is `(256 experts, 3072 hidden)` × FP16 = 1.5 MB. lm_head split into 4 vocab chunks fits 50 MB each — too big for ANE 32 MB SRAM in one piece, but a 16 MB chunk fits and the remaining 3 chunks stay on GPU. Net: router fits cleanly, lm_head needs case-by-case.

**Programming model.** Compile the router (input: hidden state `(1, 3072)` FP16, output: `(K, indices)` int32, `(K, scores)` FP16) as a CoreML mlpackage via `coremltools.convert(...)`. Wire into the MoE block with a `try ANE / else GPU` switch keyed off availability + sequence-length (decode only — prefill stays on GPU, since long-sequence router latency on ANE is dominated by IOSurface marshaling).

**Win.** Removes 124 dispatches/token from GPU command queue. Empirically that's worth +5–8 % tok/s on M3 Ultra. M5 Max has higher GPU dispatch overhead headroom (Neural Accelerators leave more SIMD groups idle for control flow), so the win likely scales similarly or better.

**Risk.** ANE's 119-compilations-per-process limit (Orion paper). Mitigation: compile once at load time and persist mlpackage in the bundle.

## 7. Kernel design (Phase A)

Two new MLX primitives, both subclassing `mlx::core::Primitive` like cider's `W8A8Linear`:

### 7.1 `tq_na_matmul_prefill`

Inputs (per dispatch):
- `x_rot` `(B, in_features)` FP16 — rotated activations from the existing Hadamard kernel
- `tq_packed` `(out, packed_in)` uint32 — bit-packed codebook indices (existing JANGTQ_K layout)
- `tq_tile_scale` `(out_tiles,)` FP16 — per-output-tile scale (L2)
- `tq_norms_log8` `(out,)` uint8 — per-row log-scale residual on top of `tq_tile_scale` (L1)
- `tq_codebook_int8` `(4,)` int8 — symmetrically-quantized codebook entries
- `tq_codebook_int8_scale` `()` FP16 scalar — codebook dequant scale (`absmax / 127`)
- `per_token_scale` `(B,)` FP16 — per-token activation INT8 quantization scale

Output: `(B, out)` FP16.

Per-output reconstruction (mathematically exact):

```
W_dequant[r, k] = (codebook_int8[idx[r, k]] * codebook_int8_scale)
                  * tile_scale[r // 16]
                  * exp2((u8_norms[r] - 128) / 16)
out[b, r] = sum_k(x_rot[b, k] * W_dequant[r, k])
         = (per_token_scale[b] * tile_scale[r // 16] * codebook_int8_scale * exp2(...))
            * sum_k(x_int8[b, k] * codebook_int8[idx[r, k]])
```

The per-output combined scale is `per_token_scale[b] × tile_scale[r // 16] × codebook_int8_scale × norms_hat[r]`. The kernel accumulates `sum_k` in INT32 via `mpp::tensor_ops::matmul2d`, then multiplies by the combined scale once when storing the FP16 output.

Kernel sketch:

```metal
// One threadgroup per (B-tile, out-tile) pair. Tile = (16 rows, 32 cols).
// Threadgroup memory: 256 × FP16 LUT for exp2_lut[u8] (L1 reconstruction).
// Stage 1: load (16, 32) packed weights, unpack codebook indices to INT8 via codebook_int8 LUT.
// Stage 2: load x_rot (B-tile, 32), per-token quantize to INT8 (using per_token_scale).
// Stage 3: mpp::tensor_ops::matmul2d(16, 32, 16) over INT8 -> INT32 accumulator.
// Stage 4: per-output combined-scale dequant on store.

constexpr auto md = mpp::tensor_ops::matmul2d_descriptor(/*M=*/16, /*N=*/32, /*K=*/16,
                                                          /*A=*/int8_t, /*B=*/int8_t,
                                                          /*C=*/int32_t);
auto matmul_op = mpp::tensor_ops::matmul2d<md, threads_per_simdgroup>{};
auto a_tile = cooperative_tensor::load<int8_t>(x_int8 + ...);
auto b_tile = cooperative_tensor::load<int8_t>(w_int8_unpacked + ...);
auto c_tile = matmul_op(a_tile, b_tile);

// Fused store: combined scale = per_token * tile * codebook * exp2(u8 - 128) / 16
half cb_scale = tq_codebook_int8_scale;
for (int i = 0; i < tile_size; ++i) {
    int r_local = i / B_tile;
    int b_local = i % B_tile;
    half norms_hat = exp2_lut[u8_norms[row_base + r_local]];
    half combined = per_token_scale[b_base + b_local]
                    * tile_scale[(row_base + r_local) >> 4]
                    * cb_scale
                    * norms_hat;
    out_fp16[i] = half(float(c_tile[i]) * combined);
}
```

Reference for the descriptor + `cooperative_tensor` API: `liuliu/example_matmul_metal4` (minimal working example) and cider's `csrc/src/w8a8_primitive.mm`.

### 7.2 `tq_na_router_prefill`

Generalizes the router's gate Linear to use `tq_na_matmul_prefill` once it's stable. Out of scope for Phase A initial landing — router stays on standard `nn.QuantizedLinear` in Phase A v1 and gets folded in once decode coherence is verified.

### 7.3 What stays on the existing kernels in Phase A

- `hadamard_rotate_metal` — multi-block butterfly. Not a matmul, no NA path.
- `fused_gate_up_swiglu_decode` — decode fast path, kept verbatim.
- `gather_tq_decode_per_row` — decode fast path, kept verbatim.
- `MLA QKV fused matmul` (P18) — could move to `tq_na_matmul_prefill` opportunistically. Defer to Phase B.

## 8. Implementation milestones

| Phase | Milestone | Owning files |
|---|---|---|
| A.1 | New `jang-tools/jang_tools/turboquant/na_kernel.py` with `tq_na_matmul_prefill` MLX primitive (Python side) and Metal source string | new file |
| A.2 | C++ shim `jang-tools/csrc/jangtq_na_primitive.mm` if needed for Metal 4 features not exposed via Python kernel API yet | new file (TBD — first attempt is pure-Python via `mx.fast.metal_kernel`, fall back to C++ shim only if `cooperative_tensor` isn't reachable) |
| A.3 | `jang_tools.convert_minimax_jangtq_na` converter — reads MiniMax-M2.7-JANGTQ_K bundle, applies L1+L2, writes new bundle | new file |
| A.4 | `jang_tools.load_jangtq_na` loader hook — extends `load_jangtq_model` with `na_path` capability detection + monkeypatch swap | edit `load_jangtq.py` |
| A.5 | Verifier `jang_tools.verify_jangtq_na` — checks all §5 invariants on a bundle | new file |
| A.6 | Bench `research/scripts/bench_jangtq_na.py` — pp/s + tok/s + MMLU vs JANGTQ_K | new file |
| A.7 | Smoke test on small JANGTQ_K bundle (Holo3-35B-A3B-JANGTQ if available, else Qwen3.6-35B-A3B-JANGTQ) before applying to MiniMax M2.7 | uses A.6 |
| A.8 | Convert + bench MiniMax-M2.7-JANGTQ-NA. Gate decision per §2 | uses A.3 + A.6 |
| B.x | Decode kernel generalization, L3 per-layer codebook calibration, Holo3 + Qwen3.6 ports | future |
| C.x | Native MiniMax-NA finetune + ANE router export + lm_head head-block | future |

A.1–A.7 are deliberately small, individually testable units. The only place where a deep design decision could still bite is A.2 — whether `mx.fast.metal_kernel` can express `cooperative_tensor` bindings, or whether we need C++ glue. Investigate first and adjust the spec if it forces a C++ build into the converter pipeline.

## 9. Risks and rollback

| Risk | Likelihood | Mitigation |
|---|---|---|
| `cooperative_tensor` not accessible from `mx.fast.metal_kernel` source string | Medium — MLX issue #2693 still open as of 2026-05 | Fall back to a C++ MLX primitive (cider's pattern). Adds ~200 LOC of nanobind glue + per-platform `.mm` build step. Confirmed feasible. |
| Phase A passes pp/s + e2e bars but pure decode regresses > 3 % vs JANGTQ_K | Medium — Phase A's decode kernel is unchanged from JANGTQ_K, but converter / loader micro-overhead can leak into decode (e.g. extra tensor materialization, monkey-patch ordering). | Profile per-call attribution (see `JANGTQ-REFERENCE.md` §7 method); root-cause and fix before promoting. Do NOT silently ship a decode regression. |
| L1 uint8 log-scale shifts MMLU > 0.5 pp | Low — bf16 activation noise floor exceeds the 2.2 % per-row scale error | L1 is mandatory in this bundle (§6 L1). If the converter's L1 gate fails, the JANGTQ-NA bundle is shelved and we ship JANGTQ_K instead. There is no "ship without L1" path within JANGTQ-NA. |
| L2 per-tile scale fails on a non-MiniMax model | Medium-high for cross-model transfer | Per-layer histogram check in the converter, abort if σ/μ > 5 % for any layer. MiniMax M2.7 is the reference target; transfer to other JANGTQ models requires re-validation. |
| macOS 26.2 not GA on a user's machine | Medium | Loader hard-errors with a clear message naming the JANGTQ_K bundle as the alternative; no silent fallback. |
| Per-token activation INT8 quant overflows on outlier activations | Medium | Clip to `[-127, 127]`; track clip rate during the smoke test; if > 0.1 % of activations clip, switch to per-row activation scales (per-channel cider mode) instead of per-token. |
| ANE compilation cap (119/process, Orion) hits L4 in Phase C | High if router recompiles per session | Cache mlpackage in bundle, never recompile at runtime. |

Rollback strategy: the JANGTQ-NA bundle is a separate HF bundle line. If the NA path fails the §2 gate, do not publish — `MiniMax-M2.7-JANGTQ_K` continues to serve all hardware. There is no consumer-visible regression because nothing depends on the new bundle until the §2 gate passes.

## 10. Open questions for implementation

1. **Does `mx.fast.metal_kernel` accept Metal 4 `mpp::tensor_ops::matmul2d` source?** If yes, milestone A.2 is empty. If no, we need a build-system change in `jang-tools` to compile a `.mm` into a `.dylib` (or per-platform Metal library) and dispatch through a C++ MLX primitive. Test in A.1; the answer reshapes the rest of the plan.
2. **Is the M5 Max NA tile shape `(16, 32, 16)` optimal for MiniMax shapes (`in=3072, out=1536` for gate/up; `in=1536, out=3072` for down), or do we need a sweep?** P17 found the sweet spot on M3 Ultra differs from M4 by 4×. Reserve a half-day for an OPT-style sweep on M5 Max early in A.1.
3. **Does activation per-token INT8 quantization belong inside the Hadamard kernel, or is a free `mx.quantize_per_token`-style op acceptable?** Free op is simpler; integrated may be faster. Bench in A.6.
4. **Per-token vs per-row activation scaling.** Per-token is cider's default; per-row (per-channel) is more conservative. If the smoke test sees > 0.1 % activation clipping at INT8, switch to per-row before Phase A productionizes.
5. **L4 (Phase C only): ship the ANE router as part of the bundle (mlpackage embedded) or as a runtime-fetched artifact?** Embed. Bundles are self-contained per JANGTQ discipline.

## 11. Why this is the right shape

- **Honors the user's three asks.** Faster inference (Phase A kernel — prefill 3-4×, decode held flat-or-better). Quality preserved through L3 in Phase B. New format that survives cleanly alongside existing bundles (separate HF line, no consumer pain).
- **No false advertising on size.** L1 + L2 are kernel-layout layers; they happen to net to about zero bundle delta. Phase B does NOT shrink the bundle either — L3 improves quality at fixed bits. Real bit-rate reduction lives in Phase C and requires a finetune.
- **De-risks linearly.** Phase A kernel must work before B is justified. Phase A bundle format must work before B's per-layer codebooks add per-layer storage. Phase A's ANE-rejection rationale is what justifies Phase C's ANE-acceptance for the small-component carve-out.
- **Reuses two projects' worth of validation.** cider proved the integration shape on Apple Silicon (MLX custom primitive + Metal 4 cooperative_tensor); JANGTQ-PRESTACK proved we can extend the bundle layout without breaking consumers. Both are 2026 work, the latest possible reference points.
- **Doesn't pretend the ANE is the answer when it isn't.** ANE is reserved for the one place it actually wins (small router, small models for spec-decode drafts). The "NPU" inference path users want is the GPU Neural Accelerators — that's where the model lives.

---

## References

- `research/JANGTQ-REFERENCE.md` — current JANGTQ_K decode physics, optimization inventory P1–P18.
- `research/JANGTQ-PRESTACK-SPEC.md` — current bundle layout discipline.
- `research/MINIMAX-QUANT-NOISE-AUDIT.md` — per-layer activation σ data backing L2 and L3.
- Apple ML Research, "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" (machinelearning.apple.com, 2026-03).
- Mininglamp-AI/cider, GitHub (2026) — MLX W8A8/W4A8 primitive implementation.
- liuliu/example_matmul_metal4, GitHub — minimal Metal 4 cooperative_tensor example.
- ml-explore/mlx issue #2693 — MLX team's tracking issue for Metal 4 / M5 NA support.
- Orion: Characterizing and Programming Apple's Neural Engine, arXiv 2603.06728 — ANE constraints, FP16-only compute, 32 MB SRAM.
- ANEMLL — github.com/Anemll/Anemll, ANE LLM inference reference numbers.
