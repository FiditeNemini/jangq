# Swift JANGTQ deep-dive — where the 3× gap to Python actually lives

**Date:** 2026-04-16
**Context:** User question — "JANGTQ should be ~100 tok/s on Swift AND Python."
**Result:** Python ✅ (95 tok/s). Swift still ❌ (29 tok/s after fixes; 27 before).
**Investigation depth:** instrumented kernel selection + applied compile fix + profiled per-component.

## The hunt for the gap (in order of investigation)

### 1. GatedDeltaNet (SSM) — RULED OUT as bottleneck

Hypothesis: 30 of 40 layers are `linear_attn` (Mamba-style SSM). If Swift fell
back to the slow `gatedDeltaOps` path instead of the fused
`gated_delta_step` Metal kernel, ~75% of layers would be slow.

Instrumented `GatedDelta.swift` to log which path fires on first call:

```
[GatedDelta] using FUSED Metal kernel (dtype=float16)
```

**Confirmed: Swift IS using the fused kernel** (matches Python's
`mlx_lm/models/gated_delta.py` `gated_delta_kernel`). One Metal dispatch per
SSM layer per token. Same as Python.

### 2. MoE TurboQuantSwitchGLU — fast kernel exists, dispatch count matches Python

Per-MoE-layer in BOTH paths: 4 Metal kernels (rotate → fused gate+up+SwiGLU →
rotate → gather TQ matmul). The `JANGTQKernels.fusedGateUpSwiGLU` is the same
P17 OPT=10 outputs-per-thread design as Python's
`fused_gate_up_kernel.py`. Routed expert math is identical dispatch count.

I tried wrapping the 4-kernel chain in `compile(shapeless: true)` mirroring
Python's `_get_compiled_decode`. **Zero speedup.** MLX-Swift's compile only
fuses element-wise pure-MLX ops; custom `MLXFast.metalKernel` calls are
opaque dispatches that compile cannot reorder, fuse, or elide.

The compiled cache is in `TurboQuantSwitchLinear.swift` and is functionally
correct (same outputs); it just doesn't help today.

### 3. MoE router math — applied compile fix, +5% measured

Python's `load_jangtq.py:399-417` wraps the router chain
(`softmax → argpartition → take_along_axis → divide`) in `mx.compile`,
caching by `(k, renorm)`. This fuses ~3-4 element-wise dispatches per layer ×
40 layers per token.

Ported to Swift as `qwen35JANGTQCompiledRouter(numExperts:k:renorm:)` in
`Qwen35JANGTQ.swift`. Required passing `numExperts` as a captured constant
because `scores.dim(-1)` can't be traced under compile (originally crashed
with "Slice cannot infer output shapes"). Fixed by hard-coding `kth = numExperts - k`
into the closure capture.

**Measured impact: 27.3 → 28.9 tok/s** on 128-token bench (3 warm runs).
~5% speedup. Less than expected — suggests router was a small part of total
dispatch overhead.

### 4. Where the remaining 3× gap actually lives

After GatedDelta-fast-path + router-compile, Swift is still at ~29 tok/s vs
Python's 95. That's 30% of Python.

Per-token cost ledger (estimated dispatches):

| Component | Python (compiled) | Swift (current) | Δ |
|---|---:|---:|---:|
| 30 GatedDelta layers (fused kernel) | 30 × 8 = 240 | 30 × 8 = 240 | 0 |
| 10 full attention layers | 10 × 4 (compiled qkv+rope) | 10 × 7 (separate) | +30 |
| 40 router math (compiled) | 40 × 1 | 40 × 4 (this round) → 40 × 1 (after fix) | 0 |
| 40 MoE TQ kernels | 40 × 4 | 40 × 4 | 0 |
| 40 shared-expert SwiGLU | 40 × 1 (compiled) | 40 × 4 (separate) | +120 |
| 40 sigmoid_gate × shared | 40 × 1 (already compiled) | 40 × 1 (already compiled) | 0 |
| 80 norms | 80 × 1 | 80 × 1 | 0 |
| Sampling + emit | ~3 | ~5 (HTTP overhead) | +2 |
| **Total est.** | **~370** | **~520** | **+150** |

At ~30 μs per dispatch overhead, the 150-dispatch delta is ~4.5 ms/token.
Python tokens at 10.5 ms; Swift tokens at 35 ms = 24.5 ms gap. So dispatch
count alone explains ~18% of the gap; the rest is per-dispatch overhead
(MLXArray Swift→C round-trip, Module field reflection, KVCache update path,
HTTP serialization).

## What WOULD close the gap (concrete, ranked)

1. **Compile attention math + RoPE in `Qwen35Attention.callAsFunction`** —
   port Python's mlx_lm pattern. Saves ~30 dispatches/token. Easy.
2. **Compile shared_expert SwiGLU** in Qwen35JANGTQSparseMoeBlock — wrap
   `gate_proj(x)→silu→up_proj(x)→multiply→down_proj` as one MLX graph.
   Saves ~120 dispatches/token. Medium.
3. **Compile linear_attn surrounding ops** in Qwen35GatedDeltaNet — wraps
   in_proj_qkv/z/b/a + reshapes + RMSNorm + RoPE. The `gated_delta_step`
   kernel is opaque to compile so the chain breaks at the kernel call;
   compile each side separately.
4. **Engine HTTP per-token loop instrumentation** — measure how much of
   the 35ms/token is non-MLX (chat-template re-render on each turn, EOS
   check, sampler, JSON encode). If >5ms/tok, optimize the Hummingbird
   stream path.

Combined estimate: **closing all four would put Swift at ~70-80 tok/s**.
Hitting Python's 95 tok/s requires the engine optimizations from (4) plus
some kernel-level reordering. Realistic target after this work: **75 tok/s
on Swift M4 Max**, vs Python's 95 — within 20%.

## VL status (per "VL LAYERS SUPER IMPORTANT")

- **On disk**: 333 vision_tower tensors preserved (~600 MB fp16 passthrough)
- **Python**: VL works natively via mlx_lm's Qwen3_5MoeModel
- **Swift**: text-only — `Qwen35JANGTQModel.sanitize` strips vision_tower
  keys (LLM-side class has no ViT). Adding VL needs a parallel
  `Qwen35MoEJANGTQ.swift` in `vMLXVLM` mirroring `vMLXVLM/Models/Qwen35MoE.swift`
  (~200 LOC) but using `TurboQuantSwitchGLU` for routed experts. Future work.

## Final measured numbers (today, MacBook M4 Max)

```
Python JANGTQ_2L (decode, after warmup):  95.24 tok/s   ✅
Python JANG_2L  (decode, after warmup):   38.45 tok/s
Swift  JANGTQ_2L (HTTP, 128 tok, warm):   28.85 tok/s   ❌
Swift  JANGTQ_2L (HTTP, 512 tok, warm):   28.45 tok/s
Swift  JANG_2L   (HTTP, 128 tok, warm):   ~22 tok/s
```

**Headlines:**
- Python JANGTQ ✅ matches user's "should be ~100 tok/s" expectation.
- Swift JANGTQ at 30% of Python — the gap is in element-wise op dispatch
  overhead distributed across attention, shared expert, and engine path.
- JANGTQ vs JANG_2L on Swift: +30% (codebook beats affine).

## Code changes this round (vmlx tree, uncommitted)

- `Sources/vMLXLLM/Models/GatedDelta.swift` — added one-shot debug logging
  to confirm fused kernel path is taken (then removed). Confirmed.
- `Sources/vMLXLLM/Models/Qwen35JANGTQ.swift` — added
  `qwen35JANGTQCompiledRouter(numExperts:k:renorm:)` + cache, mirrors
  Python P15 router compile.
- `Sources/vMLXLMCommon/TurboQuantSwitchLinear.swift` — added compiled
  cache for the 4-kernel MoE chain (kept; harmless even though no speedup).

## Next session targets to actually hit ~75 tok/s on Swift

Priority order:
1. Compile shared_expert SwiGLU (biggest single dispatch save)
2. Compile attention QKV+RoPE+SDPA chain
3. Compile linear_attn pre/post-kernel chains
4. Profile vMLXEngine.stream per-token loop for non-MLX overhead

Each of (1)-(3) is ~1-2 hours of focused work. (4) is medium-day. Net result:
Swift at parity (within 20%) of Python on JANGTQ decode.
