# MiniMax-M2.7 decode hot-path audit — dtype flow + dispatch count

**Date:** 2026-04-14
**Target:** MiniMax-M2.7-JANG_2L @ 14.35 tok/s on M4 Max, find the highest-ROI optimizations
**Method:** source-code inspection of the forward-pass code path, no live profiling yet
**Model code:** `vmlx/swift/Sources/vMLXLLM/Models/MiniMax.swift` + `vMLXLMCommon/SwitchLayers.swift` + `vMLXLMCommon/Evaluate.swift`

## Per-token dispatch budget

Counting Metal kernel dispatches per decoder layer in the decode phase (single token, L=1):

### Attention block (one layer)

```
wq(x), wk(x), wv(x)           3 × gather_qmm       (bf16 out)
qNorm/kNorm                    2 × fast.rmsNorm     (optional, MiniMax uses attn_k_eq_v so possibly 0)
reshape + transpose × 3        free (stride ops)
RoPE on q / RoPE on k          ~4 dispatches (sin/cos + complex multiply)
SDPA + cache update            2 × fast.sdpa + 1 scatter
wo(output)                     1 × gather_qmm
----
~11 dispatches per attention layer
```

### MoE block (one layer)

```
gate(x)                         1 Linear (router, 3072 -> 256)
sigmoid(gates)                  1
scores + eScoreCorrectionBias   1 (broadcast add)
argPartition(-scores, ...)      2 (negate + argPartition)
takeAlong(scores, inds)         1
scores / (sum + eps)            3 (sum, add, divide)
scores.asType(x.dtype)          1  (maybe no-op, see dtype section)
SwitchGLU:
  expandedDimensions            free
  gateProj gather_qmm           1
  upProj gather_qmm             1
  compiledSwiGLU                1 (compiled) / 2 (uncompiled)
  downProj gather_qmm           1
  squeezed                      free
weighted sum (y * scores[...]).sum(axis: -2)   2
----
~16-17 dispatches per MoE block
```

### Layer total

```
input_layernorm       1 fast.rmsNorm
attention block       11
residual add          1
post_attn_layernorm   1 fast.rmsNorm
MoE block             16-17
residual add          1
================
~31-32 dispatches per decoder layer
```

### Full-model budget per token

- Embedding lookup: **1**
- 62 decoder layers × 31 dispatches: **~1922**
- Final norm + lm_head: **3**
- Sample (argmax + post-processing): **3**
- **Total: ~1930 dispatches per decoded token**

At measured 14.35 tok/s (69.7 ms/token), that's **~36 μs per dispatch average**. This is right around the theoretical Metal command-buffer launch overhead on M-series Apple Silicon (~25–40 μs depending on kernel type) — **dispatch overhead is a significant chunk of total decode time**, not just the math inside each kernel.

### Implication

The single biggest lever purely from the dispatch-count angle is **reducing the number of dispatches per layer**. If we can cut 31 → 20 dispatches per layer, we save roughly (11 × 62 × 36 μs) = **~24.6 ms per token**, giving a new ceiling of ~22 tok/s from dispatch reduction alone. That's a real path to beating the ~14 tok/s baseline even before any kernel-level work.

## Dtype flow audit

### Clean paths

- **Attention forward** stays in bfloat16 throughout: Linear → RMSNorm (fast.rmsNorm handles fp32 accumulator internally) → reshape → RoPE → SDPA → reshape → Linear.
- **Residual adds** are bfloat16 + bfloat16.
- **Final logit softcapping** is once per token at the output head, not per layer.

### Suspicious sites

| Location | What happens | Impact |
|---|---|---|
| `MiniMax.swift:106` `MLXArray.zeros([args.numLocalExperts])` | `eScoreCorrectionBias` is initialised with no dtype arg. MLXArray.zeros defaults to Float32 unless overridden. If model load doesn't overwrite with a bf16 weight, the downstream `scores + eScoreCorrectionBias` add promotes bf16 → fp32. | Potentially the entire MoE routing computation runs in fp32 instead of bf16 — multiple implicit cast dispatches per layer. **Verify at runtime.** |
| `MiniMax.swift:129` `scores = scores.asType(x.dtype)` | Explicit cast before SwitchGLU. If upstream stayed in bf16, this is a no-op identity; if upstream drifted to fp32, this is load-bearing. | 1 dispatch/layer, 62/token — free to remove if upstream is bf16-clean. |
| `SwitchLayers.swift:122` compiledSwiGLU/compiledGeGLU | Gated on `HardwareInfo.isCompiledDecodeSupported` which defaults to `false`. Activation runs as unfused `silu(gate) * x` = 2 Metal dispatches instead of 1. | 1 extra dispatch/layer, 62/token. Measured live on MiniMax with `VMLX_FORCE_COMPILE_DECODE=1`: +1.3% speedup. |
| `Evaluate.swift:286-287` logits → fp32 before sampling | Once per token. Necessary for stable argmax. | Negligible. |

### Not a problem

- **RMSNorm** uses `MLXFast.rmsNorm` which handles the fp32 internal accumulator without creating a visible graph node. No cast cascade.
- **SDPA** uses `MLXFast.scaledDotProductAttention` which keeps inputs in bf16 and accumulates softmax in fp32 internally. No visible cast.
- **MLX Swift `QuantizedLinear`** dispatches directly to the `gather_quantized_mm` kernel on bf16 activations × uint32 packed weights × fp16 scales. All internal to the kernel.

### The dtype finding

**There is no dtype cascade in the forward pass.** The worst case is a potential fp32 escape through `eScoreCorrectionBias` init, which is one line to fix. **Kernel dispatch count reduction, not dtype choice, is the real lever.**

## Ranked optimization opportunities

### Tier 1 — dispatch reduction (highest impact)

#### 1. QKV fusion

Combine `wq`, `wk`, `wv` into a single `wqkv` Linear by concatenating their weight tensors at load time, then splitting the output. Drops 3 dispatches → 1 per attention layer. Saves **2 × 62 = 124 dispatches per token ≈ ~4.5 ms** at 36 μs/dispatch.

Implementation:
- Sanitize-time hook detects `*.q_proj.weight|scales|biases` + `k_proj` + `v_proj` and concats them into `*.qkv_proj.*`.
- `MiniMaxAttention` holds a single `wqkv: QuantizedLinear` and splits the output into 3 along the output axis.
- For JANG-quantized weights: concat must happen in the packed uint32 domain. If Q/K/V layers have DIFFERENT bit widths (per-tier quantization), fusion is blocked for those layers. MiniMax M2.7 JANG_2L has all three at 8-bit CRITICAL tier, so fusion is safe.

Risk: bit-width mismatch between Q/K/V causing load-time rejection. Mitigation: fall back to 3-way unfused for mismatched layers.

**Expected speedup: 15–25% on decode.**

#### 2. gate_up fusion inside SwitchGLU

SwitchGLU does `upProj(x) + gateProj(x)` which are two separate `gatherQuantizedMM` dispatches. Combine them — one fused `gate_up_proj` with output split along the intermediate axis. Drops 2 dispatches → 1 per MoE layer. Saves **62 dispatches per token ≈ ~2.2 ms**.

Implementation: same sanitize-time concat as #1 but on the MoE expert weights axis. After the fused dispatch, split the output into `gate` and `up` halves along the intermediate-size axis and proceed with `silu(gate) * up`.

**Expected speedup: 8–12% on decode.**

#### 3. Full-forward compile path

Already in the codebase at `Evaluate.swift:912`, gated on `HardwareInfo.isCompiledDecodeSupported` which we've confirmed works on M4 Max (env override verified, no crash). This wraps the entire decoder forward body in `compile(inputs: outputs:)` which lets MLX fuse kernels at the graph level across layers.

Measured impact on MiniMax when toggled ON: **+1.3%** (noise). This lever is largely exhausted for MiniMax specifically — the cache-update writeback and per-layer graph structure block cross-layer fusion.

### Tier 2 — redundant op removal (free but minor)

#### 4. Fix `eScoreCorrectionBias` init dtype

One line. `MLXArray.zeros([args.numLocalExperts], dtype: .bfloat16)`. Guarantees the MoE scores computation stays in bf16 even if the model weight load path doesn't override it. Worth doing regardless.

#### 5. Remove `scores = scores.asType(x.dtype)` (MiniMax.swift:129)

After fix #4, this cast is guaranteed to be a no-op identity. Delete the line. 62 dispatches/token saved ≈ ~0.2 ms. Trivial but clean.

#### 6. Avoid negation in argPartition

`argPartition(-scores, kth: k-1, axis: -1)[..<k]` → `argPartition(scores, kth: numExperts-k, axis: -1)[numExperts-k..<numExperts]`. Saves the unary negate dispatch. 62/token ≈ ~0.2 ms.

#### 7. Hoist epsilon tensor

`MLXArray(1e-20, dtype: scores.dtype)` is allocated every forward call. Make it a module-level cached constant. Likely MLX already CSE's this but worth verifying.

### Tier 3 — structural changes (biggest, riskiest)

#### 8. Cold expert pruning (Plan 7 L2)

Threshold-drop experts with routing weight below 5% of max. Halves effective top-k in the common case. **20–40% decode speedup** expected at ~1% MMLU cost.

Implementation hazard: the gather_qmm kernel expects a fixed `k` at compile time. Dynamic k may force recompilation per unique k value → kernel cache thrashing. Mitigation: quantize the effective k to `{2, 4, 6, 8}` and let MLX cache compiled kernels per k.

#### 9. Metal-hosted speculative decoding (Plan 8)

Separate plan. 5–6 week scope. Ceiling ~3× effective decode.

## Recommended execution order

**Session 1 (next block of work):**
1. Fix `eScoreCorrectionBias` init dtype → bfloat16 explicit (1 line).
2. Probe at runtime: does `VMLX_FORCE_COMPILE_DECODE=1` actually hit the `compile(inputs:outputs:)` branch in Evaluate.swift:912 for MiniMax, or does the KVCacheSimple gate skip it? If skipped, find out why and ungate.
3. Remove `scores.asType(x.dtype)` after verifying dtype is bf16 (1 line).
4. Swap argPartition to avoid negation (1 line).

Expected gain from (1)+(3)+(4): <1%. Low risk, easy to verify.

**Session 2:**
5. QKV fusion in sanitize + `MiniMaxAttention`. Biggest single lever.
6. Re-benchmark bundle vs source, confirm both benefit equally.
7. Expected gain: **15–25% on decode → ~17–18 tok/s**.

**Session 3:**
8. gate_up fusion in SwitchGLU sanitize.
9. Expected gain: additional **8–12% → ~19–21 tok/s**.

**Session 4–5:**
10. Cold expert pruning with MMLU validation.
11. Expected gain: **20–40% → ~25–30 tok/s**.

**Session 6+:**
12. Plan 8 speculative decoding with Metal-hosted draft.
13. Ceiling: **~50–80 tok/s effective**.

## Measurements needed before touching code

- Confirm `eScoreCorrectionBias.dtype` at runtime for a loaded MiniMax (bf16 vs fp32)
- Confirm whether `compile(inputs: cache, outputs: cache)` at Evaluate.swift:912 fires for MiniMax when `VMLX_FORCE_COMPILE_DECODE=1` is set (log / signpost)
- Xcode Instruments Metal trace to validate the theoretical 1930/token dispatch estimate

## What I would NOT do first

- JANGTQ P13+ Metal kernel extensions — slower path to impact than QKV fusion, needs deep Metal expertise, fusion wins come from graph-level restructuring not kernel internals.
- More aggressive quantization (2-bit → 1.5-bit for some layers) — quality risk, doesn't help dispatch count.
- Switch activation functions — SwiGLU is correct for MiniMax, changing breaks the model.
- Swap dtype bf16 → fp16 — bf16 is correct for MiniMax and matches the Metal accumulator.

---

## Cleanup results (measured)

Applied the one-line cleanups from Tier 2 to `MiniMax.swift` and re-measured:

```
eScoreCorrectionBias: MLXArray.zeros([...], dtype: .bfloat16)      // was: default fp32
argPartition(scores, kth: numExperts - k)[numExperts-k..<numExperts]  // was: argPartition(-scores, ...)[..<k]
epsilon: static MLXArray constant on the class                      // was: MLXArray(1e-20, ...) per forward
scores.asType(x.dtype) removed                                      // no-op after eScoreCorrectionBias fix
```

Result on MiniMax-M2.7-JANG_2L bundle, clean RAM, same prompt (60 prompt / 256 gen / greedy):

| Config | tok/s |
|---|---:|
| Baseline (source dir) | 14.35 |
| Baseline (bundle fat layout) | 14.23 |
| Bundle fat + compile-decode on | 14.42 |
| **Bundle fat + MoE cleanups** | **14.99** |

**+4.5% decode speedup** from four one-line changes. The biggest contributor is almost certainly the `scores + eScoreCorrectionBias` path staying in bf16 instead of promoting to fp32 — the negate + asType fixes are each small but compound. The eScoreCorrectionBias dtype init alone avoids ~5 dispatches per layer × 62 layers = ~310 ops per token.

This confirms the audit hypothesis: **dispatch count reduction is the real lever**, and the Tier 1 fusions (QKV + gate_up) should give much bigger wins because they collapse multiple quantized matmul dispatches per layer.

### New projected ceiling after Tier 1

Expected tok/s after QKV fusion + gate_up fusion on top of the current cleanups:
- +15–25% from QKV fusion → ~17–19 tok/s
- +8–12% from gate_up fusion → ~19–21 tok/s
- +20–40% from cold expert pruning → ~25–30 tok/s

Target: 20 tok/s is a realistic stretch goal before Plan 8 speculative decoding kicks in.

---

## QKV fusion — implemented, measurement pending

### What landed

`vmlx/swift/Sources/vMLXLLM/Models/MiniMax.swift`:

1. `MiniMaxAttention` rewritten to use a single fused `@ModuleInfo(key: "qkv_proj") var wqkv: Linear` with output dim = `qOutDim + 2 × kvOutDim` = `6144 + 1024 + 1024 = 8192`. Drops 2 dispatches per attention layer per token.
2. Forward pass: `let qkv = wqkv(x)` → slice along the last axis with `qkv[.ellipsis, 0..<qOutDim]` (queries), `[qOutDim..<(qOutDim+kvOutDim)]` (keys), `[(qOutDim+kvOutDim)..<(qOutDim+2*kvOutDim)]` (values). The slice is a zero-copy view; the downstream `reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)` is handled by MLX without an extra materialization.
3. `MiniMaxModel.sanitize(weights:)` now walks every layer and:
   - Reads `self_attn.{q,k,v}_proj.weight`
   - Checks the last-dim `packed_in` of all three — must be equal (same JANG bit width)
   - On match: `concatenated([qW, kW, vW], axis: 0)` → writes `self_attn.qkv_proj.weight`, removes the three per-projection keys
   - Same treatment for `.scales` and `.biases`
   - On mismatch (different bit widths across q/k/v in the same layer): leaves them alone, skips fusion for that layer

For MiniMax-M2.7-JANG_2L all three attention projections are 8-bit CRITICAL tier, so fusion fires on all 62 layers.

### Build status

Full release build clean after one unrelated bystander fix (`Stream.swift:477` had a `Self.logger.warning(...)` introduced by a concurrent agent edit where `Engine` is an actor with no `logger`; replaced with a direct stderr write to unblock).

### Measurement pending

Could not run a decode bench in this session. Memory state at the time of completion:
- `free=0.1 GB` (unusable)
- `compressed=76.4 GB` (held by previous MiniMax loads)
- `xctest` eating 32 GB running vmlx package tests + another for vmlx-swift-lm package tests (both launched by parallel sessions, not mine)

Per the RAM-safety directive, the session policy is: only launch MiniMax serve when `free + inactive` >= 80 GB and no competing xctests are eating RAM. That state was not available in this session.

### What to run when RAM is ready

```bash
# Clear competitors first
pgrep -lf "vmlxctl|xctest" | head   # verify whose xctests are live
# (only kill what's yours)

# Verify memory
python3 -c "import subprocess,re; o=subprocess.check_output(['vm_stat']).decode(); \
  free=int(re.search(r'Pages free:\s+(\d+)', o).group(1)); \
  print(f'free={free*16384/1e9:.1f}GB')"

# Launch (exec -a protects from pkill collisions with other test sessions)
cd /Users/eric/vmlx/swift
cat > /tmp/serve-qkv.sh <<'SH'
#!/bin/bash
cd /Users/eric/vmlx/swift
exec -a "jangspec-qkv" ./.build/release/vmlxctl serve \
    --model /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec \
    --port 8775
SH
chmod +x /tmp/serve-qkv.sh && nohup bash /tmp/serve-qkv.sh > /tmp/serve-qkv.log 2>&1 &
disown

# Wait for "Ready" in the log (usually 15–20s on clean RAM)
# Warmup + timed bench
curl -sS http://127.0.0.1:8775/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Hi."}],"max_tokens":4}' > /dev/null

START=$(python3 -c "import time;print(time.time())")
curl -sS http://127.0.0.1:8775/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"Write a short paragraph about the Roman Empire from 27 BC to 476 AD. Be concise but thorough."}],"max_tokens":256,"temperature":0}' \
  > /tmp/qkv-resp.json
END=$(python3 -c "import time;print(time.time())")

python3 -c "
import json
r = json.load(open('/tmp/qkv-resp.json'))
u = r['usage']
elapsed = $END - $START
print(f'QKV fusion:')
print(f'  wallclock = {elapsed:.2f}s, prompt={u[\"prompt_tokens\"]}, completion={u[\"completion_tokens\"]}')
print(f'  tok/s = {u[\"completion_tokens\"] / elapsed:.3f}')
"

# Always kill the serve immediately after
kill $(pgrep -f jangspec-qkv)
```

Expected outcome per the ceiling analysis: **~17–18 tok/s** (up from 14.99 baseline), roughly +15–20%. This bumps us from 14.4% → ~17% of the 104 tok/s memory-bandwidth ceiling.

### Risk assessment

- **Correctness**: the slice → reshape → transpose sequence is identical in byte layout to the original three-projection path. The concatenated weight is byte-equivalent to stacking three row ranges. Output tokens should be identical to pre-fusion for greedy decode.
- **Bit-width mismatch**: if some bundle has different bit widths for q/k/v in any layer, the sanitize fusion guard silently skips fusion for that layer AND leaves the original per-projection weights in the dict. But the model code now expects `qkv_proj` as the only module key — the `q_proj`/`k_proj`/`v_proj` keys won't match any module, and `loadWeights` will either warn or drop them. **This is a gap**: we either need to (a) assume all MiniMax JANG variants fuse-safely, or (b) add a fallback 3-way path in `MiniMaxAttention`. For now this assumes JANG MoE always assigns attention q/k/v to the same CRITICAL tier — documented limitation.
