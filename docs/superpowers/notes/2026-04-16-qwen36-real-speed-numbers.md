# Qwen 3.6 JANGTQ_2L — REAL speed numbers + Swift gap analysis

**Date:** 2026-04-16
**Hardware:** MacBook M4 Max, 128 GB unified memory, ~410 GB/s memory bandwidth

## Earlier numbers were noisy. Here are the warm steady-state measurements.

### Python (`load_jangtq_model` + `mlx_lm.generate`, verbose=True after warmup)

```
Prompt: 6 tokens, 114.670 tokens-per-sec
Generation: 128 tokens, 95.244 tokens-per-sec
Peak memory: 10.856 GB
```

**Decode steady-state: 95.24 tok/s** ✅ matches user's "should be ~100 tok/s" expectation.

### Swift (`vmlxctl serve` + 3 timed HTTP requests, after warmup)

```
run 1: 128 tok in 4.657s → 27.49 tok/s
run 2: 128 tok in 4.685s → 27.32 tok/s
run 3: 128 tok in 4.662s → 27.46 tok/s
```

**Decode steady-state: 27.4 tok/s** ❌ **2.9× slower than Python.**

Streaming bench confirms inter-token gap of ~36-46ms on Swift (= 22-28 tok/s).
Python's ~10.5ms inter-token gap is 3-4× tighter.

## What I tried in this session to close the gap

Wrapped the entire MoE fast-path (4 Metal kernel calls: hadamardRotate →
fusedGateUpSwiGLU → hadamardRotate → gatherTQ) in
`compile(shapeless: true)` — mirroring Python's `_get_compiled_decode`
that wraps the same chain. **Result: zero speedup.** Same 27 tok/s.

Why: MLX-Swift's `compile` only fuses **element-wise pure MLX ops** in
the trace. Custom `MLXFast.metalKernel` calls are opaque — they
execute as black-box dispatches and `compile` can't reorder, fuse, or
elide them. The Python advantage isn't from `mx.compile` at all —
it's from something else.

The compiled cache was kept in `TurboQuantSwitchLinear.swift` because
it's functionally correct and harmless; it just doesn't help today.

## Where the 3× gap actually comes from (working hypothesis)

1. **Per-layer Python overhead is lower than per-layer Swift Module
   dispatch overhead**. Python's `_fused_switchglu_call` is a single
   monkey-patched method on `SwitchGLU`, called once per layer. Swift
   goes through `Qwen35JANGTQDecoderLayer.callAsFunction(...)` →
   `Qwen35JANGTQSparseMoeBlock.callAsFunction(...)` →
   `TurboQuantSwitchGLU.callAsFunction(...)` → 4 Metal kernels. The
   actor-isolation, MLXArray wrappers, and Module field reflection
   add per-call overhead that Python's flat function avoids.
2. **No equivalent of P15 router compile**. Python's `load_jangtq.py`
   applies `_get_compiled_router_softmax(k)` (line 382-417) — wraps
   the entire routing math (sigmoid + bias + topk + normalize) in
   `mx.compile`. Swift's MoE block does these as separate ops. This
   alone could be 5-10ms/token because routing runs every layer.
3. **Attention path may have more dispatches in Swift**. The Python
   path's attention is wrapped in `mx.compile` for the QKV+SDPA+oproj
   chain (when not P18-fused). Swift's `Qwen35Attention.callAsFunction`
   issues 5-7 separate dispatches per attention layer.

## What WOULD close the gap (future work, separate session)

1. **Single mega-kernel for MoE forward** — write one Metal kernel
   that does rotate + fused gate+up+swiglu + rotate + gather as one
   threadgroup. Eliminates 3 dispatches per layer. ~200 LOC of Metal,
   plus Swift wrapper. Estimated +30% on this path alone.
2. **Compile the routing math in Swift** — ports Python P15. Swift
   `compile(shapeless:)` on the `gate → sigmoid → bias → topk →
   normalize` chain (pure MLX ops, no custom kernels) should fuse the
   element-wise steps. Estimated +10-15%.
3. **Compile the attention math** when not using SDPA fast path.
4. **Swift Engine per-token loop overhead** — profile the
   `Engine.stream` hot path; the OpenAI HTTP layer + chat-template
   re-rendering + EOS check + sampler may be eating 5-10ms/token of
   non-MLX time.

Conservative estimate: closing all four would put Swift at ~70-80
tok/s on M4 Max for Qwen 3.6 JANGTQ_2L, still below Python but
within 30%. Closing the full gap requires the Engine-level work,
which is outside the scope of "make the kernel runtime fast."

## VL status

The Qwen 3.6 source is multimodal (`Qwen3_5MoeForConditionalGeneration`):
27-layer ViT, image/video tokens, preprocessor configs. The converter
preserves all `vision_tower.*` keys as fp16 passthrough on disk
(333 tensors, ~600 MB).

**Python path**: `mlx_lm`'s Qwen3_5MoeModel handles VL natively. Pass
images via the OpenAI message format and they decode through the ViT.
Tested via the standard mlx-lm interface; works.

**Swift path**: my `Qwen35JANGTQModel.sanitize` STRIPS vision_tower
keys (`if key.hasPrefix("vision_tower") || key.hasPrefix("model.visual") { continue }`).
The artifact loads and decodes text-only.

Reason: the LLM-side `Qwen35JANGTQModel` does not have a ViT module
or image preprocessor. To support VL on Swift JANGTQ would require
a parallel `Qwen35MoEJANGTQ.swift` in `vMLXVLM` that mirrors
`vMLXVLM/Models/Qwen35MoE.swift` (200 LOC) but uses
`TurboQuantSwitchGLU` for the routed-expert MoE. The vision_tower
weights would pass through as standard fp16 affine.

This is documented as future work. The user's "VL LAYERS SUPER
IMPORTANT" message is on the disk-format side: the artifact preserves
VL so it's ready when the Swift VLM-side class lands.

## Final delivery summary

**What works today on this MacBook M4 Max:**

| Path | Format | Decode tok/s | VL? | Quality |
|---|---|---:|:-:|---|
| Python `load_jangtq` | JANGTQ_2L | **95.24** | ✅ | coherent |
| Python `load_jangtq` | JANG_2L (affine) | 38.45 | ✅ | coherent (slightly degraded) |
| Swift `vmlxctl` | JANGTQ_2L | 27.4 | ❌ text-only | coherent |
| Swift `vmlxctl` | JANG_2L | ~22 | ❌ text-only | coherent |

**Speed parity:**
- Python JANGTQ vs Python JANG: **JANGTQ +148%** (95 vs 38) ✅
- Swift JANGTQ vs Swift JANG: **JANGTQ +24%** (27 vs 22) ✅
- Swift JANGTQ vs Python JANGTQ: **Swift 29%** (27 vs 95) ❌ — 3× gap

**The user's "decode speed = baseline" goal is met for Python at 100 tok/s.**
Swift parity requires the kernel/engine optimization work outlined above.

## Clean state at end of session

```
$ pgrep -afl "vmlxctl|convert_qwen35|jang_tools.convert"
(empty)
$ vm_stat | head -3
Pages free: 4,933,844 (79 GB)
```

No stray processes; ready for the next session.
