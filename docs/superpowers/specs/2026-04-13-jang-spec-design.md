# jang-spec — SSD-Streamed MoE Speculative Decoding

- **Status:** design, not yet implemented
- **Date:** 2026-04-13
- **Author:** Jinho Jang (eric@jangq.ai)
- **Target runtime:** Swift + Metal, Apple Silicon (macOS 15+), unified memory
- **Ship vehicle:** `jang-runtime` during development, native to `vmlx` as the shipping home
- **Prior art:** [DFlash (z-lab/dflash)](https://github.com/z-lab/dflash) — parallel block drafting; we adopt the *idea* of block drafting but not its implementation, format, recipe, or code.

## 1. Goal

Run JANG MoE models that do not fit in unified memory (GLM-5.1, MiniMax M2.7, Cascade-2, Qwen 122B/397B-class) on a single Mac Studio by:

1. Keeping a small **draft model** resident in RAM.
2. Storing the **target** MoE model on SSD in a purpose-built streaming container, streaming experts on demand into unified memory via `MTLIOCommandQueue`.
3. Using speculative decoding so each SSD-streamed target verification pass advances output by many tokens, amortizing I/O.

Expert prefetching is driven by a **precomputed router prior** (coactivation and transition statistics captured during bundle build). The prior pins high-coactivation experts in the LRU cache and prefetches high-probability experts for the next layer based on the current layer's selection. This is a purely static, inference-free prefetch strategy.

A future v2 will add *router-aware drafting* — training the draft to predict target expert routing — to improve prefetch hit rate. That is explicitly deferred; v1 does not depend on it working.

## 2. Non-goals (v1)

- Multi-GPU, multi-host, distributed inference.
- Tree-based speculative decoding. v1 ships linear-block drafting only.
- Batching. v1 is single-stream; vmlx is a local app, not a server.
- Prefix caching across requests.
- Dense (non-MoE) JANG models. They fit in RAM; streaming buys nothing.
- Training the draft model from scratch inside this spec. A sibling spec covers the training recipe; v1 consumes a pre-trained draft.
- **Router-aware drafting.** Deferred to v2. See §13.

## 3. Why this is not a DFlash clone

| Dimension | DFlash | jang-spec |
|---|---|---|
| Drafting mechanism | Block diffusion with iterative refinement | Masked-prediction heads (Medusa-style), block-parallel in one pass |
| Target location | VRAM, fully resident | SSD, streamed per-expert |
| Platform | CUDA (vLLM, SGLang) | Swift + Metal unified memory |
| Container | Vanilla HuggingFace safetensors | Purpose-built `.jangspec` bundle with per-expert blobs and flat index |
| Calibration | N/A | Coactivation + transition priors to warm expert cache and drive prefetch |

DFlash is cited as prior art for "parallel block drafting as a paradigm." No DFlash code, checkpoints, training recipe, or benchmarks are reused.

The defensible contribution of v1 is the **streaming runtime itself**: `.jangspec` container format, unified-memory expert cache with prior-driven prefetch, `MTLIOCommandQueue`-based direct SSD reads, and Swift/Metal-native speculative decoding on Apple Silicon. No existing framework runs MoE targets from SSD this way.

## 4. Architecture overview

```
┌──────────────────────────────────────────────────────────────┐
│  Host RAM (unified memory)                                   │
│                                                              │
│   ┌───────────────────────┐   ┌──────────────────────────┐   │
│   │ Draft engine (1–4 B)  │   │ Target hot core          │   │
│   │ router-aware heads    │   │  attn q/k/v/o, router,   │   │
│   │ KV cache              │   │  shared experts, norms,  │   │
│   │                       │   │  embed / lm_head         │   │
│   └──────────┬────────────┘   └──────────▲───────────────┘   │
│              │                           │                  │
│              │ block of B tokens +       │ verification      │
│              │ predicted expert IDs      │ forward pass      │
│              ▼                           │                  │
│   ┌───────────────────────────────────────────────────────┐  │
│   │ Expert LRU cache (unified memory, N GB ceiling)       │  │
│   │   warmed from router_prior at startup                 │  │
│   └──────────▲───────────────────────────▲────────────────┘  │
│              │ MTLIOCommandQueue         │                   │
└──────────────┼───────────────────────────┼───────────────────┘
               │                           │
         ┌─────┴───────────────────────────┴─────┐
         │  SSD: .jangspec bundle                │
         │    experts-*.bin (per-expert blobs)   │
         │    experts.jsidx (mmap'd index)       │
         └───────────────────────────────────────┘
```

One step of the decoding loop:

1. Draft forward on current context, outputs `B` token proposals via Medusa-style masked-prediction heads in a single forward pass.
2. Target verification: parallel forward pass over the `B` proposed tokens. Per MoE layer, compute the router on the hot core, see which experts the target actually chose. The prior-driven prefetcher, seeded at startup and updated layer-to-layer by the transition matrix, aims to have those experts already resident. Any miss triggers a synchronous load. Expert matmul uses a gather-quantized kernel.
3. Accept the longest prefix where target and draft agree, roll back KV for rejected suffix, emit accepted tokens, loop.

Prefetch is *pipelined across layers within the verification pass*: while layer L's experts are computing, the transition matrix predicts layer L+1's likely experts and issues async reads for those not resident. This overlaps I/O with compute even without a router-aware draft.

## 5. The `.jangspec` bundle format

```
<name>.jangspec/
  jangspec.json                  manifest: version, tokenizer hash, target arch,
                                 hot-core tensor list, expert layout, prior version
  target/
    jang_config.json             copied from source JANG
    config.json                  HF config with quantization key
    hot_core.safetensors         MLX-native, pinned-resident tensors
    experts.jsidx                flat index of per-expert blobs (mmap'd)
    experts-00001.bin            raw quantized expert blobs, 4 KB-aligned
    experts-00002.bin
    ...
  draft/
    jang_config.json
    config.json
    model.safetensors            JANG-quantized draft, 1–4 B params
    medusa_heads.safetensors     B masked-prediction heads
  router_prior/
    coact.safetensors            expert coactivation stats (calibration)
    transition.safetensors       layer L -> L+1 expert transition probs
  tokenizer.json                 shared; mismatch is a build-time error
  tokenizer_config.json
```

### 5.1 Hot core

The hot core is computed at bundle-build time by scanning the source JANG's tier classification and pulling everything in the **CRITICAL** and **IMPORTANT** tiers that is not a streamed expert:

- `model.embed_tokens.*`
- `lm_head.*` (or tied)
- `layers.N.self_attn.{q,k,v,o}_proj.*`
- `layers.N.*router*`, `layers.N.mlp.gate.*` (MoE router gates)
- `layers.N.shared_expert.*` (any shared experts)
- `layers.N.*_layernorm.weight`, `model.norm.weight`

On GLM-5.1 1L / MiniMax M2.7-class models this is ~8–20 GB, comfortably resident alongside a 4 B draft.

### 5.2 Expert blobs

Source JANG already stores MoE experts as 3D stacked tensors, e.g. `switch_mlp.gate_proj.weight` of shape `[E, I, packed]` in `uint32`, with matching `scales` and `biases` of shape `[E, I, n_groups]` in `float16`.

The bundle builder slices each expert out of the 3D tensor at construction time, producing for each `(layer, expert_id)` tuple a contiguous byte range containing:

```
struct ExpertBlob {
    uint32_t magic;              // "JSPE"
    uint16_t bits;               // per-tensor bit widths
    uint16_t n_tensors;          // 3 for {gate, up, down}
    TensorHeader headers[3];     // offset, nbytes, dtype per sub-tensor
    uint8_t  pad_to_4k[...];
    uint8_t  payload[...];       // gate qweight, gate scales, gate biases,
                                 // up qweight, up scales, up biases,
                                 // down qweight, down scales, down biases
};
```

All blobs are 4 KB-aligned so `MTLIOCommandQueue` can issue direct reads.

### 5.3 Flat index

`experts.jsidx` is a packed binary table, one entry per expert:

```
struct ExpertIndexEntry {
    uint32_t layer_idx;
    uint32_t expert_id;
    uint16_t file_id;            // experts-NNNNN.bin
    uint16_t _pad;
    uint64_t offset;
    uint64_t nbytes;
};
```

The file header stores `(n_layers, n_experts_per_layer, total_entries, version)`. Lookup is `table[layer_idx * n_experts_per_layer + expert_id]`, zero parsing.

### 5.4 Router prior

During bundle build, a small calibration set (e.g. 1k WikiText + 1k code prompts) is run through the source JANG with router-logit logging. We emit:

- `coact.safetensors`: for each layer, an `(E, E)` coactivation matrix (how often expert i and j are chosen together in a token).
- `transition.safetensors`: for each layer pair `(L, L+1)`, an `(E, E)` transition matrix (probability expert at layer L+1 given expert at layer L).

These warm the expert LRU cache at startup and provide a cheap fallback if the draft's router head is uncertain.

## 6. Draft model

### 6.1 Architecture

- Backbone: small dense or small-MoE transformer, 1–4 B params, JANG-quantized.
- Shares tokenizer with target (hash-enforced at bundle build time).
- Heads (all run from the same final hidden state):

```
h = backbone(x)                           # [T, hidden]
token_logits = token_head(h)              # [T, vocab]

# B Medusa-style masked-prediction heads for block drafting
for k in 0..<B:
    pos_logits[k] = masked_head_k(h)      # [T, vocab]
```

### 6.2 Block drafting

We borrow the *idea* of parallel block proposals from DFlash without copying its diffusion approach: we train `B` Medusa-style masked-prediction heads. Each head predicts `x_{t+k}` from `h_t` with its own weights. At generation time, one draft forward produces a block of B proposed tokens. This is a published, well-understood technique and is implemented in our own code.

B is configurable; defaults:
- B=8 for latency-sensitive runs
- B=16 for throughput runs (matches DFlash's observed sweet spot)

### 6.3 Training recipe (owned by a sibling spec)

Plain distillation from the target JANG:

```
loss = α · CE(token_head(h), x_true)
     + β · Σ_k CE(masked_head_k(h), x_true[+k])
```

Training is out of scope for this spec; jang-spec runtime assumes a pre-trained draft exists. For the first target (GLM-5.1 JANG_1L) a minimal draft can be any small JANG model with a matching tokenizer, with Medusa heads trained in a few GPU-days.

## 7. Streaming target runtime

### 7.1 Expert cache

A fixed-size cache in unified memory holding decoded expert `MTLBuffer`s keyed by `(layer, expert_id)`. Size is configurable (default 40 GB on 192 GB Mac Studio; must leave room for hot core + draft + KV cache + per-token scratch).

Eviction: LRU with a pinned set seeded from the router prior's top-coactivating experts at startup, so the most common experts never evict.

### 7.2 Prefetch strategy

Prior-driven, pipelined within the verification pass:

```
// At startup: pin the top-N coactivating experts per layer, seeded from
//             coact.safetensors, until the cache budget is exhausted.

// During verification, one layer ahead of compute:
func onLayerStart(L) {
    if L+1 < nLayers {
        // transition[L][actual_topk(L)] -> distribution over layer L+1 experts
        let predicted = transition[L].predict(actual_topk_at_L, topk: targetTopK * 2)
        for (expert) in predicted where !cache.contains((L+1, expert)) {
            cache.enqueueLoad(L+1, expert)   // async MTLIOCommandBuffer
        }
    }
}
```

`MTLIOCommandQueue` (Metal 3) reads directly from file handles into Metal buffers backed by unified memory, no CPU staging copy. We do not wait; compute on layer L continues while layer L+1's experts stream in.

The prior is cheap and static — `transition[L]` is just a matrix lookup, and it's accurate enough to hit the most common next experts because token-level routing has strong layer-to-layer correlation in trained MoE models.

### 7.3 Verification forward pass

Parallel forward on the B proposed tokens, one KV write per token, one router evaluation per MoE layer:

```
for L in 0..<target.nLayers {
    x = attention(hot_core.attn[L], x, kvCache[L])   // hot path, no I/O
    if L is MoE layer {
        r = hot_core.router[L](x)                    // [B, E] hot
        actual_topk = topk(r, k)                     // [B, k]
        need = Set(actual_topk.flatten())
        miss = need.subtracting(cache.resident)
        if !miss.isEmpty {
            metrics.routerMiss += miss.count
            cache.loadSync(miss)                      // slow path, blocking
        }
        x = dispatchExperts(actual_topk, cache, x)   // gather-quantized matmul
        if hot_core.hasShared[L] {
            x = x + hot_core.shared[L](x)
        }
    } else {
        x = denseMLP(hot_core.mlp[L], x)
    }
}
logits = lm_head(x)
```

`dispatchExperts` uses a new Metal kernel `JANGExpertMatmul.metal` that is a gather-variant of the existing `quantized_matmul` — pass in the indices of selected experts plus a cache handle, compute the expert MLP per token, sum weighted outputs by gate scores. Written from scratch using the existing dequant primitives in `Metal/JANGDequant.metal` as building blocks.

### 7.4 Metrics emitted per step

- `expert_hit_rate`: fraction of required experts already resident
- `prefetch_hit_rate`: fraction hit because the transition prior prefetched them
- `pinned_hit_rate`: fraction hit because the coactivation prior pinned them
- `accept_rate`: fraction of B proposed tokens accepted by target
- `io_ms`, `compute_ms`, `io_compute_overlap_ms`
- `evictions`, `cache_bytes_resident`, `cache_pressure`

These drive user-facing tuning and motivate the v2 router-aware draft (§13) if prior-only hit rate turns out to be insufficient.

## 8. Swift engine layout

### 8.1 During development: `jang-runtime`

The existing Swift package at `/Users/eric/jang/jang-runtime` gains two new SwiftPM targets. The current dense v1 `JANG` target is left untouched and will be deprecated later.

```
jang-runtime/
  Package.swift                   # adds JANGCore, JANGSpec, updates JANGCLI
  Sources/
    JANGMetal/                    # shared, reused
    JANG/                         # DEPRECATED — dense v1, untouched
    JANGCore/                     # NEW — v2 JANG loader (MLX-native uint32 +
                                  #       scales/biases), MoE switch_mlp,
                                  #       router, shared experts. Used by
                                  #       both draft and target.
    JANGSpec/                     # NEW — jang-spec runtime
      JangSpecBundle.swift        # manifest + index parsing
      HotCoreLoader.swift         # mmap and pin hot_core.safetensors
      ExpertStreamer.swift        # MTLIOCommandQueue + LRU + prefetch
      RouterPrior.swift           # warm cache from prior
      DraftEngine.swift           # router-aware draft forward
      TargetEngine.swift          # streaming verification forward
      SpecDecoder.swift           # draft→prefetch→verify→accept loop
      Metrics.swift               # per-step metric collection
    JANGSpecCLI/                  # `jang-spec run <bundle> --prompt ...`
  Metal/
    JANGExpertMatmul.metal        # NEW — gather-quantized expert MLP
```

`JANGCore` exists as its own target rather than an expansion of `JANG` because the existing `JANG` target is dense+v1 only. Mixing MoE + v2 into it risks breaking the working dense path. `JANGCore` is the long-term home for all inference primitives; `JANG` becomes an alias over it once parity is reached.

### 8.2 Shipping home: `vmlx`

Once `JANGSpec` and `JANGCore` are stable in `jang-runtime`, they move into `vmlx` as first-class Swift targets alongside the Electron app:

```
vmlx/
  engine/
    vllm_mlx/                     # existing Python engine, unchanged
    jang-spec/                    # NEW — mirrored Swift sources
      Package.swift
      Sources/JANGCore/
      Sources/JANGSpec/
      Sources/jang-specd/         # HTTP/Unix-socket daemon
      Metal/
```

`jang-specd` exposes an OpenAI-compatible streaming API (`/v1/chat/completions` with SSE) matching the contract `vllm_mlx` already speaks. The Electron panel's model picker detects `.jangspec` directories and routes requests to `jang-specd`; `.jang` directories keep routing to `vllm_mlx`. No panel changes required beyond model detection.

Mirroring (not re-developing) means the canonical Swift sources live in `jang-runtime` during active development and get copied into `vmlx` at release time via a release script. Once the format is stable, canonical ownership moves to `vmlx` and `jang-runtime` becomes the dev consumer.

## 9. Bundle build tool

A Python tool in `jang-tools`:

```
jang spec build <source-jang-dir> \
    --draft <draft-jang-dir> \
    --draft-heads <heads.safetensors> \
    --calibration <calib.jsonl> \
    --out <bundle-dir>
```

Steps:

1. Load source JANG via `jang_tools.loader`.
2. Classify tensors into hot-core vs streamed experts by tier + name.
3. Write `hot_core.safetensors` (MLX-native, mmap-friendly).
4. For each `(layer, expert_id)`, slice gate/up/down from the 3D stacked tensors, pack into an `ExpertBlob`, 4 KB-align, append to an `experts-NNNNN.bin` (rolled at 4 GB per file for filesystem friendliness).
5. Emit `experts.jsidx`.
6. Copy draft into `draft/`, enforce tokenizer hash match.
7. Run calibration through the source target, record router top-k, compute coactivation and transition stats, emit `router_prior/`.
8. Write `jangspec.json` with all versions and hashes.
9. Verify: reload the bundle in pure Python and sanity-check one expert per layer.

Dense JANG models are rejected with a clear error.

## 10. Per-model rollout

Priority order for the first bundles:

1. **GLM-5.1 JANG_1L** — already working on Mac Studio; lowest-risk demo.
2. **MiniMax M2.7** — four profiles (2L/3L/4M/6M) share a single draft since they come from the same source.
3. **Cascade-2** (nemotron_h) — different architecture, validates that the design generalizes beyond Qwen-derived MoE.
4. **Qwen3.5-122B-A10B, Qwen3.5-397B-A17B** — stretch targets DFlash lists as "coming soon."

Dense JANG models (4B, 9B, 27B) are not candidates.

## 11. Open questions

- **`MTLIOCommandQueue` throughput in practice.** Apple quotes ~5 GB/s from NVMe on M-series Ultra. Must be measured on real Mac Studio with expert-sized reads (10–100 MB) before committing to the streaming premise. De-risked by Spike A in §14.
- **Prior-only prefetch hit rate.** Token-level routing has known layer-to-layer correlation, but how much of the needed experts the transition matrix actually catches is unmeasured on our target models. Low hit rate is survivable (synchronous loads degrade speed, not correctness) but directly sets achievable tok/s.
- **Draft size sweet spot.** 1 B may be too weak to drive good token acceptance; 4 B may be too slow per draft step. Start at 2 B, tune per target.
- **Acceptance-rate floor.** If the draft accepts <4 of 16 proposed tokens, SSD streaming can't amortize the per-verification I/O cost. Need a benchmark before committing to a draft size.

## 12. Success criteria for v1

- GLM-5.1 JANG_1L running on 192 GB Mac Studio at >= 8 tok/s sustained with coherent output.
- End-to-end expert cache hit rate (prior pins + transition prefetch) >= 80% on typical prompts.
- Accepted-tokens-per-verification >= 6 (i.e., >= 6/B acceptance rate).
- `jang-specd` drop-in replaces `vllm_mlx` in the vmlx Electron app for at least one target model, no panel code changes.
- Zero code, checkpoints, or training artifacts borrowed from DFlash; prior-art citation only.

## 13. Future work (v2)

**Router-aware drafting.** Add a router-prediction head to the draft model that emits `[L_target × E_target × topk]` expert-ID predictions per token, trained with a distillation loss against target router top-k labels. When mature, the draft's predictions union with the transition-prior prefetch and cover token-specific routing the prior cannot.

Gated by two things: (a) evidence from v1 that prior-only prefetch is hit-rate-limited, and (b) a feasibility study showing a small backbone can actually learn target routing better than the prior baseline.

**Other v2 candidates:** tree-based speculative decoding, multi-stream batching, cross-request expert cache, compressed expert blobs that skip the dequant step by running matmul directly on the packed format.

## 14. De-risk spikes (pre-implementation)

Before writing the implementation plan in full, run one spike to validate the streaming premise:

**Spike A — `MTLIOCommandQueue` benchmark.** 1 day. Swift binary `jang-spec-iobench` that:
- Creates a tmp directory with 512 files of 50 MB each (roughly one expert's worth for a MiniMax-class target).
- Measures random-access read throughput via `MTLIOCommandQueue` into unified-memory `MTLBuffer`s.
- Compares against plain `pread`.
- Reports: GB/s sustained, p50/p99 per-read latency, CPU copy cost.

Go/no-go: if sustained random-access throughput on Mac Studio SSD is < 2 GB/s or per-read latency > 5 ms, the spec needs revision (larger block sizes, fewer experts-per-verification, or giving up on streaming). If >= 3 GB/s with < 2 ms latency, proceed to full implementation.

The router-prediction feasibility spike is deferred along with the router-aware feature itself.
