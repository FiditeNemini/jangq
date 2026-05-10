# JANGFP — JANG Floating-Point Microblock Quantization Format

> Custom 2/3/4-bit weight format + Metal kernel family for Apple Silicon, designed to
> succeed MXTQ as the primary JANG quantization path while running on M3 Ultra,
> M4 Max, and M5 Max with opportunistic per-chip uplift.

**Author:** Jinho Jang (eric@jangq.ai)
**Created:** 2026-04-27
**Status:** Design — brainstorming in progress, sections being approved incrementally.

---

## 0. Why this exists

MXTQ is the current JANG 2-bit/3-bit weight format. It uses Hadamard rotation +
Lloyd-Max codebook + per-row FP16 norms. Real measured wins: +1.2 dB SNR at 2-bit
vs RTN affine, deployed across 7 model families (DSV4, GLM-5.1, Kimi K2.6,
MiniMax M2.7, Mistral 4, Qwen3.5/3.6, and the dense Gemma 4 fallback).

Real measured ceilings (from
`research/MXTQ-KERNEL-OPTIMIZATION-ATTEMPTS-2026-04-27.md`):

- **V3 MIN structural ceiling: 25.9 tok/s on M3 Ultra**, vs DQ stock path at 36 tok/s.
- 12 ms/token gap traces to:
  - 4.3 ms of per-token Hadamard rotation (codec math requirement)
  - 6+ ms of `load_jangtq` runtime patches (P15 compile, P18 QKV fuse, P19 MLA fuse, fp32 upcasts)
- 4 distinct fused-rotation kernel attempts all reverted. Conclusion: rotation
  cost is **structural to MXTQ on the runtime side**, not a kernel tuning issue.
- 2-bit codebook noise floor: dimension-independent **9.3 dB SNR** across
  in_features 1536 → 7168 (`MINIMAX-QUANT-NOISE-AUDIT.md`). Smaller models are not
  more vulnerable; the floor is geometric.
- Per-row FP16 norm is too coarse for tensors with localized outliers. GLM-5.1
  routed `up_proj` measured **8.0 dB SNR / 65.7% relative error** at 2-bit — the
  per-row norm averaged across 4096 weights smears outliers and the codebook
  cannot recover.

Plus several non-quantitative pain points:

- **DSV4 FP8-native source pain**: silent BF16 fallback when FP8 dequant scale
  shapes mismatch (GLM-5.1 Issue #1: `(576, 6144)` weight with `(5, 48)` scale,
  576/5 = 115.2 → integer division produced 1-row-short tensor → silent
  bf16-from-raw-fp8-bytes garbage). All 79 layers had corrupted KV compression.
- **Qwen3.6-35B-A3B-JANGTQ4 Swift coherence bug** (2026-04-25): the bits=4
  packing path in `JANGTQKernels.swift` produces degenerate "2+2\n2+2\n..."
  loop output. The 4-bit code path is structurally undertested.
- **MoE shared expert quality cliff** at 2-bit (proven on GLM-5.1 — needed
  MXTQ3 for shared, can't drop to 2). Format today doesn't gracefully express
  per-tensor bit selection; it's 2/3 mixed via separate metadata fields.
- **Dense models are uncovered**: Gemma 4 27B has zero MXTQ benefit (no MoE);
  it lives on affine 4-bit, missing the kernel and quality wins of a custom format.

---

## 1. Goals (E3 — speed and quality both, Pareto checks at every fork)

1. **Structurally eliminate per-token Hadamard cost** by absorbing rotation
   offline at encode time. The runtime kernel sees no global rotation work; only
   a small online Walsh-Hadamard on intermediate dim before `down_proj` (the
   element-wise SwiGLU breaks the offline rotation otherwise).
2. **Beat MXTQ per-tensor SNR at the same bit-width** through finer scale
   granularity (FP8 microblock scale per 16 weights, vs FP16 per-row norm) and
   Lloyd-Max codebook on the rotated weight distribution.
3. **Cover every MLP-shaped tensor across every JANG-supported family**, with
   per-tensor bit selection in {2, 3, 4} driven by calibration SNR.
4. **One kernel family** handling 2/3/4 bits via shared microblock structure.
5. **Run on M3 Ultra, M4 Max, M5 Max** with the M3 Ultra GPU-only Metal path as
   the floor. SME (M4+) and GPU Neural Accelerators (M5+) are opportunistic
   uplifts, not requirements.
6. **Coexist with MXTQ forever** — no migration forced; existing 94 JANG
   bundles keep loading via `TurboQuantLinear`. New bundles opt-in via
   `weight_format: "jangfp"`.
7. **Loud-fail encoder.** No silent FP8 fallback. No silent shape mismatch. Hard
   conformance gates before bundle is emitted.

### Out of scope (v1)

- Sub-2-bit (1.58/1-bit ternary, vector-quant E8 lattice). Defer to v2 research.
- 5/6/7-bit. The 8-bit affine path (`mlx.nn.QuantizedLinear`) handles that lane.
- Attention QKV/O, MLA `wq_a/wq_b/wkv_a/wkv_b/o_proj`, MLA absorbed
  `embed_q/unembed_out`, embed, lm_head, router gate, `e_score_correction_bias`,
  `attn_sink`, RMSNorms, Pixtral/MoonViT/Qwen ViT, MTP heads.
  All stay 8-bit affine or fp16, exactly as today.
- Training (forward + backward). NVFP4-style stochastic-rounding training
  is a separate research project.
- ANE (Apple Neural Engine) offload. ANE is closed-ISA, fixed-graph, no
  streaming KV cache — mismatched against autoregressive decode.
- AMX direct dispatch. Private undocumented ISA on M1-M3; only reachable via
  Accelerate. Not part of the design space.
- Learned R1 rotation (SpinQuant). v1 uses random Hadamard. Learned R1 is
  +0.5 dB at 4× encoder cost — defer to v2.

---

## 2. Format & storage (Section 1 — APPROVED)

### 2.1 Microblock structure (shared across 2/3/4 bits)

- **16 elements per microblock.** SIMD-32 friendly (2 microblocks per warp
  fit in one SIMD reduce), clean uint32 packing math, matches NVFP4
  community convention.
- **Per-microblock scale: FP8 E4M3, 1 byte.** Key quality lever — 256× finer
  scale resolution than MXTQ's per-row FP16 norm, at half the dtype size.
  Targets the GLM-5.1 8.0-dB up_proj failure mode (outliers no longer smear
  across 4096 weights).
- **Per-tensor codebook: FP16, 4/8/16 levels.** Lloyd-Max optimal on the
  post-rotation weight distribution.
- **Per-tensor master scale: FP16 scalar.** Catches global tensor magnitude;
  microblock scales are relative to it.
- **Optional per-row norm: FP16.** Reserved slot; emitted only if calibration
  proves it helps a specific tensor class.

### 2.2 Pack math

| Bits | Indices/microblock | Bits/microblock | uint32/microblock | Total bpw incl FP8 scale |
|---:|---:|---:|---:|---:|
| 2 | 16 | 32 | 1 | 2.5 |
| 3 | 16 | 48 | 1.5 (3 uint32 per 2 microblocks) | 3.5 |
| 4 | 16 | 64 | 2 | 4.5 |

3-bit straddles uint32 boundaries. Kernel pairs microblocks 2-at-a-time
and reads 3 uint32 per pair — clean SIMD-32 dispatch (lanes 0-15 process
microblock 0, lanes 16-31 process microblock 1, all from the same
3-uint32 read).

### 2.3 On-disk companion tensors

Per quantized weight (replaces MXTQ's `.tq_packed` / `.tq_norms` / `.tq_bits`):

```
<base>.jfp_packed         uint32[..., out_features, packed_in_features_uint32]
<base>.jfp_scales         uint8 [..., out_features, n_microblocks]      # FP8 E4M3
<base>.jfp_codebook       float16[..., 16]                               # padded; unused entries zero
<base>.jfp_master_scale   float16 scalar (or [..., out_features] if per-row)
<base>.jfp_bits           uint8 scalar
```

Leading `...` is `(num_experts,)` for MoE switch tensors, empty for dense.

### 2.4 `jang_config.json` additions

```json
{
  "weight_format": "jangfp",
  "jangfp": {
    "format_version": "1.0",
    "microblock_size": 16,
    "scale_dtype": "fp8_e4m3",
    "rotation": {
      "r1_seed": 305419896,
      "r3_seeds": "per-tensor (see per_tensor map)"
    },
    "per_tensor": {
      "layers.0.mlp.gate_proj":      {"bits": 4, "online_hadamard": false},
      "layers.0.mlp.up_proj":        {"bits": 4, "online_hadamard": false},
      "layers.0.mlp.down_proj":      {"bits": 4, "online_hadamard": true,  "r3_seed": 12345},
      "layers.5.experts.gate_proj":  {"bits": 2, "online_hadamard": false},
      "layers.5.experts.down_proj":  {"bits": 2, "online_hadamard": true,  "r3_seed": 67890},
      "layers.5.shared_expert.up_proj": {"bits": 3, "online_hadamard": false}
    }
  }
}
```

`online_hadamard` flag on `down_proj` tensors is the runtime signal — kernel
applies a Walsh-Hadamard on the intermediate-dim activation before the matmul.
Gate/up have `online_hadamard: false` because their input is the
already-R1-rotated residual stream from the previous layer.

### 2.5 MXTQ coexistence

- Existing `.tq_packed` / `.tq_norms` / `.tq_bits` keep working unchanged.
- `weight_format: "mxtq"` → `TurboQuantLinear` (existing path, unchanged).
- `weight_format: "jangfp"` → new `JangFPLinear` module.
- No migration of existing 94 JANG bundles.
- New conversions opt in via `--format jangfp` CLI flag; eventual default.

---

## 3. Rotation & encoding (Section 2 — pending approval)

### 3.1 R1 — global residual-stream rotation (offline, zero runtime cost)

A single orthogonal `R1 ∈ ℝ^(hidden × hidden)` per model, baked into every
weight that touches the residual stream. After absorption, the residual
stream lives in a permanently-rotated basis. Activations are rotated;
quantized weights calibrated on rotated inputs. Nothing rotates at runtime.

| Weight | Rotation applied | Why |
|---|---|---|
| `embed_tokens.weight` | `W_embed @ R1` | Embedding output rotates into residual basis |
| Per-block `attention.wo` (or `o_proj` / `wo_b`) | `Wo @ R1` | Attention → residual; rotates on the way in |
| Per-block `mlp.down_proj` (and `experts.E.down_proj`, `shared_expert.down_proj`) | `Wd @ R1` (output side) | MLP → residual; rotates on the way in |
| `lm_head.weight` | `R1.T @ W_lmhead` | Logits unrotated; cancels R1 at the end |
| **RMSNorm gammas** | **untouched** | RMSNorm is element-wise; rotation propagates through unchanged |
| **Router gate, e_score_correction_bias** | `Wgate @ R1.T` (input side rotated) | Router sees rotated hidden; multiply input dim by R1ᵀ to cancel. Stays fp16. |
| **`attn_sink`, biases, non-gamma norms** | untouched | Element-wise, scalar, or rotation-invariant |

R1 is **shared across every layer** — one rotation, applied everywhere. Wg/Wu
input quantization becomes "free": they see rotated input from the previous
layer's output, calibrated on rotated input. Quantization quality up; runtime
cost zero.

**v1 R1 choice: random Hadamard with a fixed JANGFP-version seed.** Walsh-
Hadamard is sign-only (±1/√d), weight magnitudes don't blow up. QuaRot uses
this and it works. Learned R1 (SpinQuant) deferred to v2.

### 3.2 R3 — per-MLP intermediate Hadamard (online, on Wd input)

SwiGLU `silu(gate) * up` is not rotation-invariant — element-wise multiply
doesn't commute with R. Cannot rotate Wg/Wu output and expect Wd to compose
correctly.

Solution from QuaRot: apply Walsh-Hadamard `H ∈ ℝ^(intermediate × intermediate)`
to the SwiGLU output **at runtime**, before the Wd matmul. Wd is quantized in
the H-rotated basis offline.

```
gate     = x @ Wg_quant            # x is R1-rotated; Wg quantized in R1 basis
up       = x @ Wu_quant            # same
swig     = silu(gate) * up         # element-wise SwiGLU
swig_rot = walsh_hadamard(swig)    # ONLINE — only runtime rotation
y        = swig_rot @ Wd_quant     # Wd quantized in H-rotated basis; output is R1-rotated
```

Cost: `O(intermediate · log(intermediate))` per layer per token. At
intermediate=2048, ~22k FMAs per layer-token. Estimate ~1 ms total at 43
layers, vs MXTQ's 4.3 ms today. **~75% reduction in runtime rotation budget.**

R3 seed is per-tensor, in `jang_config.json`. Kernel reads seed, generates
Hadamard signs deterministically (PCG64 from existing JANGTQ work), applies
butterfly.

### 3.3 Encoder pipeline

```
1. Load source weights (HF safetensors, FP16/BF16/FP8)
   - On FP8: explicit dequant with ceiling-division shape math.
     FAIL LOUD on shape mismatch — no silent BF16 fallback. Ever.
   - try: dequant_fp8(); except Exception as e: raise RuntimeError(...)

2. Generate R1 (random Hadamard, JANGFP version seed)

3. Apply R1 globally to all R1-touched weights (Section 3.1 table).
   Validation: forward pass on calibration prompt before/after R1 must
   produce bit-identical logits in fp32 (within 1e-4). Hard gate.

4. Calibration pass (existing JANG calibration mix: code/agentic/academic,
   ~512 prompts).
   - Forward in fp32, post-R1.
   - Capture per-layer activation statistics (RMS per microblock per row).
   - Capture per-tensor input distribution for codebook fitting.

5. Per-MLP rotation (R3) for every down_proj (and shared/dense down_proj):
   a. Generate per-tensor R3 Walsh-Hadamard seed.
   b. Wd_rotated = walsh_hadamard_apply(Wd_pre_quant, axis=intermediate)
   c. Mark online_hadamard=true in tensor metadata.

6. Per-tensor bit selection:
   - For each MLP-shaped tensor, sweep bits ∈ {2, 3, 4}.
   - Pack with Lloyd-Max codebook on rotated weights; compute output SNR
     vs fp32 reference on calibration activations.
   - Pick lowest bit width exceeding SNR threshold:
     * Routed expert:                   ≥ 12 dB (MXTQ2 floor 9.3 + 1.2 dB margin)
     * Shared expert:                   ≥ 18 dB
     * Dense layer-0 / Gemma 4 dense:   ≥ 20 dB
   - Per-family overrideable (DSV4 routed may need 15 dB due to mHC).

7. Pack:
   - Per microblock of 16: max(|w|), set FP8 E4M3 scale.
   - Per-tensor Lloyd-Max codebook (4/8/16 levels) on rotated weights
     normalized by microblock scales.
   - Encode each weight as nearest codebook index.
   - Pack indices into uint32 per Section 2.2.

8. Emit JANG bundle:
   - safetensors with .jfp_packed, .jfp_scales, .jfp_codebook,
     .jfp_master_scale, .jfp_bits per quantized weight.
   - 8-bit affine path unchanged for attention/MLA/embed/lm_head/router.
   - jang_config.json with weight_format: "jangfp" + per-tensor metadata.

9. Conformance gate (blocks emit if any fails):
   a. Bit-identity check: fp32 forward on rotated-uncompressed weights vs
      original weights produces identical logits (within 1e-4).
   b. SNR check: every tensor exceeds tier threshold.
   c. Coherence check: load packed bundle, run "What is 2+2?" + 4 canonical
      JANG prompts. Output coherent (no degenerate loops, n-gram repeat
      heuristic + entropy threshold).
   d. NIAH check: 1k-token needle-in-haystack on hybrid-SSM models
      (Qwen3.5/3.6) — quant must not break long-context recall.
```

### 3.4 Per-family encoder branches

| Family | Encoder special case |
|---|---|
| **DSV4-Flash** | mHC residual mixing matrix B_l absorbed into R1 propagation per block. B_l has its own rotation behavior; encoder must handle Sinkhorn-doubly-stochastic interaction with R1. **Highest-risk family-specific work.** Hash-routed first 3 layers: routed experts quantize normally (routing by token ID, deterministic — no quality interaction with quant). |
| **Kimi K2.6 / GLM-5.1** | MLA absorbed `embed_q/unembed_out` stays 8-bit affine via QuantizedMultiLinear (out of scope, untouched). Layer-0 dense MLP (Kimi) goes through normal MLP path at calibration-picked bit width. |
| **Mistral 4** | Pixtral vision tower untouched (passthrough fp16). |
| **Qwen3.5/3.6 hybrid** | GatedDeltaNet (linear-attn) projections (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`, `conv1d`) — keep at 8-bit affine for v1 (attention-shaped, not MLP-shaped). 4-bit Swift coherence bug from 2026-04-25 was in `JANGTQKernels.swift` `vals_per_u32 = 32 / bits` packing — JANGFP's per-tensor codebook + microblock scale design eliminates that bug class. New format = clean slate. |
| **Gemma 4** | Pure dense MLPs at JANGFP. No MoE machinery. Calibrate per-tensor; expect most layers pick 4, some early/late may pick 8 (falls back to existing affine — JANGFP doesn't claim 8-bit). |
| **MiniMax M2.7** | Already at MXTQ2; JANGFP recalibrates routed experts at 2-bit. 40% expert prune stays — JANGFP is bit-width-agnostic to prune count. |

### 3.5 FP8-source robustness (DSV4, GLM-5.1, Mistral 4, Kimi K2.6)

The encoder dequants FP8 → FP32 once, applies R1, quantizes to JANGFP. The
DSV4/GLM "fp8 native we had alot of issues with it" pain was specifically:

- Silent BF16 fallback on FP8 dequant exception
- Scale-shape mismatch on non-divisible dimensions

JANGFP encoder requires:

- **Ceiling-division on FP8 scale shape:** `bh = (shape[0] + sh - 1) // sh; scale_full[:shape[0], :shape[1]]`
- **FP8 dequant exception → `RuntimeError`** with full context. Never silent fallback.
- **Per-tensor checksum logged:** `mean / std / min / max` of dequanted weights
  must match expected ranges (catches silent garbage from scale misuse).

---

## 4. Runtime & kernels (Section 3 — TBD)

To be designed.

---

## 5. Validation & coexistence (Section 4 — TBD)

To be designed.

---

## Decision log

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-04-27 | Portability across M3U / M4M / M5M is the floor, not a stretch. | User stated explicit constraint; reduces design space to "M3 Ultra GPU-only baseline + opportunistic uplift". |
| 2026-04-27 | Pareto-balanced (E3) goals — speed and quality both, evaluated at every fork. | User picked E3 over speed-first (E1) or quality-first (E2). |
| 2026-04-27 | G4 scope: 2/3/4-bit covering all MLP-shaped tensors. | Architecture survey showed Gemma 4 (dense) and Kimi K2.6 layer-0 (dense) need 4-bit coverage; G3 (2/3 only) and G1 (2 only) leave families unhelped. |
| 2026-04-27 | 16-element microblock, FP8 E4M3 scale. | NVFP4 convention; targets GLM-5.1 up_proj 8.0 dB failure mode (per-row FP16 norm too coarse); SIMD-32 friendly packing. |
| 2026-04-27 | Per-tensor Lloyd-Max codebook (not per-microblock). | Per-microblock codebook adds ~2 GB to a 200 GB model (32 bytes × 60M microblocks); per-tensor matches MXTQ's known-good approach. |
| 2026-04-27 | Coexistence with MXTQ forever — no forced migration. | 94 production JANG bundles; new format opt-in via `--format jangfp`. |
| 2026-04-27 | R1 = random Hadamard for v1; learned R1 deferred. | QuaRot precedent works; +0.5 dB SpinQuant uplift not worth 4× encoder cost in v1. |
| 2026-04-27 | R3 = online Walsh-Hadamard before Wd matmul (not eliminated 100%). | SwiGLU element-wise multiply breaks rotation invariance across gate/up boundary; QuaRot precedent. ~1 ms vs MXTQ's 4.3 ms — ~75% reduction. |
| 2026-04-27 | Hard conformance gate before bundle emit (bit-identity, SNR, coherence, NIAH). | Addresses past silent-corruption bugs (GLM-5.1 silent BF16 fallback, Qwen3.6 4-bit Swift loop). |
