# MiniMax-M2.7-JANG_2L on M4 Max — theoretical ceiling analysis

**Date:** 2026-04-14
**Hardware:** Apple M4 Max, 16-core CPU / 40-core GPU variant (546 GB/s LPDDR5X unified memory, ~34 TFLOPS fp16 GPU peak)
**SSD:** Internal ~5–7 GB/s sequential read

## Model spec (from config.json + jang_config.json)

| Property | Value |
|---|---|
| Total params | ~228 B |
| Active params/token | ~45 B (top-8 experts of 256, intermediate 1536 each) |
| hidden_size | 3072 |
| num_hidden_layers | 62 |
| num_attention_heads | 48 (query) |
| num_key_value_heads | 8 (GQA 6:1) |
| head_dim | 128 |
| vocab_size | 200,064 |
| num_local_experts | 256 |
| num_experts_per_tok | 8 |
| moe_intermediate_size | 1536 |
| JANG profile | `JANG_2L`, avg 2.1 bits, bit widths {2, 6, 8}, block_size 128 |
| Tier assignment | COMPRESS = 2-bit (MoE experts), IMPORTANT = 6-bit (embed/lm_head), CRITICAL = 8-bit (attention, routers) |

## Per-decoded-token memory read budget

Autoregressive decode at step i reads all weights the forward pass touches, once per step.

### Attention per layer (8-bit CRITICAL)

- `q_proj [3072 × 6144]` at 8-bit = 18.87 MB qweight + 0.59 MB scales/biases ≈ **19.5 MB**
- `k_proj [3072 × 1024]` at 8-bit ≈ **3.25 MB**
- `v_proj [3072 × 1024]` ≈ **3.25 MB**
- `o_proj [6144 × 3072]` at 8-bit ≈ **19.5 MB**
- RMSNorm weights: negligible

**Per attention layer: ~45.5 MB**
**× 62 layers: ~2.82 GB**

### Router per layer (8-bit CRITICAL)

- `gate [3072 × 256]` at 8-bit ≈ **0.8 MB**
- × 62 = **~50 MB**

### MoE experts (2-bit COMPRESS, top-8 of 256 per layer per token)

Per expert:
- `gate_proj [3072 × 1536]` at 2-bit = 1.18 MB + scales ≈ 1.25 MB
- `up_proj [3072 × 1536]` ≈ 1.25 MB
- `down_proj [1536 × 3072]` ≈ 1.25 MB
- Per expert: **~3.75 MB**

- Per layer (8 active experts): 8 × 3.75 = **30 MB**
- × 62 layers: **~1.86 GB**

### lm_head + embedding (6-bit IMPORTANT)

- Embedding gather: 1 row × 3072 × fp16 ≈ 6 KB per token — negligible
- `lm_head [3072 × 200064]` at 6-bit ≈ 461 MB + scales ≈ **~500 MB**

### KV cache (scales with context length)

Per cached position per layer: K + V at fp16 = 8 heads × 128 head_dim × 2 × 2 bytes = **~4 KB**
- × 62 layers × L cached positions = L × ~248 KB
- At L=256: ~64 MB per step (cumulative scan)
- At L=1024: ~250 MB per step

For a 256-token decode run, averaged over the run: **~30 MB per token**.

### Totals

| Component | Per-token read |
|---|---:|
| Attention weights | 2.82 GB |
| MoE active experts | 1.86 GB |
| Routers | 0.05 GB |
| `lm_head` | 0.50 GB |
| KV cache scan (L=256 avg) | 0.03 GB |
| Norms + biases | negligible |
| **Total** | **~5.26 GB per decoded token** |

## Theoretical ceiling — memory-bandwidth-bound

Decode is memory-bound on every modern hardware for dense autoregressive generation. Per-token read is the denominator, memory bandwidth is the numerator.

**M4 Max unified memory bandwidth: 546 GB/s**

### Pure decode (current path, no spec-dec)

```
ceiling_max  = 546 GB/s / 5.26 GB/token  = 103.8 tok/s   (100% utilization)
ceiling_80%  = 546 / 5.26 × 0.80        = 83.0 tok/s   (realistic — well-tuned matmul kernels)
ceiling_60%  = 546 / 5.26 × 0.60        = 62.3 tok/s   (typical 2-bit mixed-width kernel eff)
```

**Current measurement: 14.99 tok/s** = **14.4% of 100%-util ceiling** = **18% of 80%-util ceiling** = **24% of 60%-util ceiling**.

So with the current kernel efficiency, we have about **4–5× headroom** on the dispatch + fusion side before hitting memory bandwidth.

## Compute-bound ceiling (sanity check)

Per decoded token FLOPs:

- Attention projections per layer: 2 × 1 × (q + k + v + o) × 3072 ≈ 88 MFLOP
- Attention SDPA at L_ctx=256: 2 × 48 × 128 × 256 ≈ 3.1 MFLOP per layer
- MoE gate: 2 × 3072 × 256 ≈ 1.6 MFLOP
- MoE 8 experts × 3 projections (bf16 ops after dequant): 2 × 8 × (2 × 3072 × 1536 + 1536 × 3072) ≈ 227 MFLOP
- Per layer: ~320 MFLOP
- **Full model per token: 62 × 320 = ~19.8 GFLOP**

M4 Max GPU peak: ~34 TFLOPS fp16 / bf16

```
compute_bound_ceiling = 34 TFLOP/s / 19.8 GFLOP/token = 1717 tok/s
```

**Decode is memory-bound by ~17× over compute.** That matches the whole industry story: quantized matmul with small batch size is read-limited, not math-limited. Our dispatch-count audit from earlier is consistent — every Metal dispatch we save cuts wall-clock, not math.

## Speculative decoding ceiling

The speed-of-light argument changes dramatically with spec-dec because the target forward amortizes one weight-read over B proposed tokens. With acceptance rate α the effective tokens per target forward are:

```
effective_K = 1 + α × (B - 1)
```

Per accepted token, weight read drops to `5.26 GB / effective_K`.

### At block size B=8

| Accept rate α | effective_K | Per-token read | Ceiling (100% util) | Ceiling (60% util) |
|---:|---:|---:|---:|---:|
| 0.4 | 3.8 | 1.38 GB | 396 tok/s | 237 tok/s |
| 0.6 | 5.2 | 1.01 GB | 540 tok/s | 324 tok/s |
| 0.8 | 6.6 | 0.80 GB | 683 tok/s | 410 tok/s |

### At block size B=16

| Accept rate α | effective_K | Per-token read | Ceiling (100% util) | Ceiling (60% util) |
|---:|---:|---:|---:|---:|
| 0.4 | 7.0 | 0.75 GB | 728 tok/s | 437 tok/s |
| 0.6 | 10.0 | 0.53 GB | 1030 tok/s | 618 tok/s |
| 0.8 | 13.0 | 0.40 GB | 1365 tok/s | 819 tok/s |

Draft cost has to be subtracted. A well-tuned dense 1–2 B fp16 draft on mlx-swift Metal at 80–120 tok/s adds ~10–12 ms per B-token block. At B=8 that's ~3× the target forward cost at the ceiling — which narrows to something like **~200–400 tok/s effective** under realistic conditions (60% kernel util + draft overhead).

**Practical spec-dec target with B=8 α=0.6 Medusa draft: ~250–300 tok/s effective.**

## SSD streaming ceiling (for reference — NOT the MiniMax path)

M4 Max internal SSD: 5–7 GB/s sequential read, random-access per-page on APFS more like 3 GB/s with some latency overhead.

```
streaming_ceiling = 5.5 GB/s / 5.26 GB/token = 1.04 tok/s
```

**~1 tok/s if weights must be streamed from SSD every token.** Which is why jang-spec's SSD-streaming path (per-expert blobs via `MTLIOCommandQueue`) is only useful for models that **literally don't fit in RAM** — there, 1 tok/s is infinity better than "can't run at all." For MiniMax on a 128 GB machine the model fits in RAM and streaming is a dead path.

The earlier IO benchmark (`jang-spec-iobench`) measured 12.5 GB/s random-access `MTLIOCommandQueue` reads, which would give ~2.4 tok/s streaming ceiling. Still terrible compared to in-RAM, as expected.

## Where each optimization lands on the ceiling

Mapping the Plan 7 levers to the theoretical framing:

| Stage | tok/s | Util vs 546 GB/s | Gap to ceiling |
|---|---:|---:|---:|
| **Current (after MoE cleanups)** | **14.99** | 14.4% | 7× |
| + QKV fusion (–2 dispatches/layer) | ~18 | 17% | 5.8× |
| + gate_up fusion (–1 dispatch/layer) | ~20 | 19% | 5.2× |
| + cold expert prune (top-4 avg) | ~28 | 27% | 3.7× |
| + JANGTQ P13+ 2-bit kernel | ~45–60 | 43–58% | 1.7–2.3× |
| **Realistic pure-decode ceiling** | **~60–85** | **58–82%** | **bandwidth limit** |
| + Spec-dec B=8 α=0.6 Medusa draft | **~200–300** | bandwidth-amortized | **5–6× target decode rate** |
| + Spec-dec B=16 α=0.7 | **~400–500** | bandwidth-amortized | |

The spec-dec row is where the game actually changes — everything before it is at most a ~5× improvement; spec-dec stacks a 3–6× multiplier on top of whatever the pure decode rate is.

## Why the gap from 14.99 to ~85 tok/s is so large

Two factors:

1. **Kernel dispatch overhead dominates**: per the dispatch audit, ~1930 Metal dispatches per token × ~36 μs launch overhead ≈ 70 ms/token — basically 100% of current decode time is dispatch plumbing, not compute or memory. The QKV/gate_up fusions collapse dispatches. The JANGTQ P13+ work fuses expert dispatch across the top-k at the kernel level.

2. **2-bit mixed-bit-width gather_qmm kernel is inefficient**: MLX's `gather_qmm` supports variable bit widths per call, which costs dispatch-time branching. A dedicated 2-bit kernel would reach closer to the memory-bandwidth ceiling because it can fuse more tightly and skip the variable-bit dispatch path.

## How to read these numbers

- **14.99 tok/s** is the number we have.
- **~20 tok/s** is achievable in a week of fusion work (Tier 1 in Plan 7).
- **~30 tok/s** is achievable in 2–3 weeks including cold expert pruning.
- **~60–85 tok/s** is the pure-decode ceiling — requires deep kernel work (JANGTQ P13+).
- **~250–500 tok/s** is the spec-dec ceiling — requires Plan 8.
- **Anything above ~550 tok/s** requires H100-class hardware, not M4 Max.

**Target for the next session: QKV fusion, land at ~18–19 tok/s.** After that, gate_up fusion and cold expert pruning. Plan 8 spec-dec is the path past ~60 tok/s.
