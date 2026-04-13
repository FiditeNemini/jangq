# jang-spec — Router-Aware Speculative Decoding with SSD-Streamed MoE Targets

- **Status:** design, not yet implemented
- **Date:** 2026-04-13
- **Author:** Jinho Jang (eric@jangq.ai)
- **Target runtime:** Swift + Metal, Apple Silicon (macOS 15+), unified memory
- **Ship vehicle:** `jang-runtime` during development, native to `vmlx` as the shipping home
- **Prior art:** [DFlash (z-lab/dflash)](https://github.com/z-lab/dflash) — parallel block drafting; we adopt the *idea* of block drafting but not its implementation, format, recipe, or code.

## 1. Goal

Run JANG MoE models that do not fit in unified memory (GLM-5.1, MiniMax M2.7, Cascade-2, Qwen 122B/397B-class) on a single Mac Studio by:

1. Keeping a small **router-aware draft model** resident in RAM.
2. Storing the **target** MoE model on SSD in a purpose-built streaming container, streaming experts on demand into unified memory via `MTLIOCommandQueue`.
3. Using speculative decoding so each SSD-streamed target verification pass advances output by many tokens, amortizing I/O.

The draft predicts both tokens *and* the target's expert-routing decisions, so expert reads can be prefetched the moment drafting finishes — before verification starts. This makes streaming an MoE target practical instead of I/O-bound.

## 2. Non-goals (v1)

- Multi-GPU, multi-host, distributed inference.
- Tree-based speculative decoding. v1 ships linear-block drafting only.
- Batching. v1 is single-stream; vmlx is a local app, not a server.
- Prefix caching across requests.
- Dense (non-MoE) JANG models. They fit in RAM; streaming buys nothing.
- Training the draft model from scratch inside this spec. A sibling spec covers the training recipe; v1 consumes a pre-trained draft.

## 3. Why this is not a DFlash clone

| Dimension | DFlash | jang-spec |
|---|---|---|
| Drafting mechanism | Block diffusion with iterative refinement | Masked-prediction heads (Medusa-style), block-parallel in one pass |
| Draft output | Tokens only | **Tokens + per-layer target expert-ID predictions** |
| Target location | VRAM, fully resident | SSD, streamed per-expert |
| Platform | CUDA (vLLM, SGLang) | Swift + Metal unified memory |
| Container | Vanilla HuggingFace safetensors | Purpose-built `.jangspec` bundle with per-expert blobs and flat index |
| Calibration | N/A | Coactivation + transition priors to warm expert cache |

DFlash is cited as prior art for "parallel block drafting as a paradigm." No DFlash code, checkpoints, training recipe, or benchmarks are reused.

The defensible contribution is **router-aware drafting**: existing speculative decoders (DFlash, EAGLE, Medusa) predict tokens; none predict the target's routing. Routing prediction is only valuable when the target is streamed. The ML idea and the systems idea exist to serve each other.

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

1. Draft forward on current context, outputs `B` token proposals and a set `S` of target expert IDs it predicts will be needed.
2. Prefetch stage: submit async reads for experts in `S` not already resident via `MTLIOCommandQueue`. Return immediately; reads continue in background.
3. Target verification: parallel forward pass over the `B` proposed tokens. Per layer, compute router, see which experts the target actually chose; if any are missing from the cache, synchronous fallback load. Expert matmul uses a gather-quantized kernel.
4. Accept the longest prefix where target and draft agree, roll back KV for rejected suffix, emit accepted tokens, loop.

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
    draft_heads.safetensors      router-head weights (separate for versioning)
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

## 6. Router-aware draft model

### 6.1 Architecture

- Backbone: small dense or small-MoE transformer, 1–4 B params, JANG-quantized.
- Shares tokenizer with target (hash-enforced at bundle build time).
- Heads (both run from the same final hidden state):

```
h = backbone(x)                           # [T, hidden]
token_logits = token_head(h)              # [T, vocab]

# B parallel-prediction heads for Medusa-style block drafting
for k in 0..<B:
    pos_logits[k] = masked_head_k(h)      # [T, vocab]

# Router-aware head
router_pred = router_head(h)              # [T, L_target * topk] logits over
                                          # target expert ids; reshape to
                                          # [T, L_target, topk]
```

`L_target` = number of MoE layers in the target, `topk` = target top-k (typically 6–8). Memory cost of the router head is `hidden × L_target × E_target × topk` which is at most a few hundred MB even for MiniMax-scale targets.

### 6.2 Block drafting

Borrowing the *idea* of parallel block proposals from DFlash without copying its diffusion approach: we train `B` Medusa-style masked-prediction heads. Each head predicts `x_{t+k}` from `h_t` with its own weights. At generation time, one draft forward produces a block of B proposed tokens. This is a published, well-understood technique and is implemented in our own code.

B is configurable; defaults:
- B=8 for latency-sensitive runs
- B=16 for throughput runs (matches DFlash's observed sweet spot)

### 6.3 Training recipe (sketched here, owned by a sibling spec)

Distillation from the target JANG:

```
loss = α · CE(token_head(h), x_true)
     + β · Σ_k CE(masked_head_k(h), x_true[+k])
     + γ · Σ_L CE(router_pred[:,L,:], target_router_topk[:,L,:])
```

Target router top-k labels come from running the target JANG on the same batches and recording `topk(router_logits, k)` per MoE layer. No DFlash code, no DFlash data.

Training is out of scope for this spec; jang-spec runtime assumes a pre-trained draft exists.

## 7. Streaming target runtime

### 7.1 Expert cache

A fixed-size cache in unified memory holding decoded expert `MTLBuffer`s keyed by `(layer, expert_id)`. Size is configurable (default 40 GB on 192 GB Mac Studio; must leave room for hot core + draft + KV cache + per-token scratch).

Eviction: LRU with a pinned set seeded from the router prior's top-coactivating experts at startup, so the most common experts never evict.

### 7.2 Prefetch stage

The moment the draft forward returns:

```
pred_experts: Set<(Int, Int)> = union over b, L of router_pred[b,L,topk]
missing = pred_experts.subtracting(cache.resident)
for (layer, expert) in missing {
    cache.enqueueLoad(layer, expert)   // async MTLIOCommandBuffer
}
```

`MTLIOCommandQueue` (Metal 3) reads directly from file handles into Metal buffers backed by unified memory, no CPU staging copy. We do not wait; verification starts immediately.

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

- `router_hit_rate`: fraction of required experts already resident
- `router_prefetch_hit_rate`: fraction hit because the draft predicted them
- `router_prior_hit_rate`: fraction hit because they were pinned by the prior
- `accept_rate`: fraction of B proposed tokens accepted by target
- `io_ms`, `compute_ms`, `io_compute_overlap_ms`
- `evictions`, `cache_bytes_resident`, `cache_pressure`

These drive both user-facing tuning and, offline, retraining of the router head.

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

- **Draft size sweet spot.** 1 B probably too weak for routing prediction on 128-expert targets; 4 B may be too slow for drafting. Start at 2 B, tune empirically per target.
- **`MTLIOCommandQueue` throughput in practice.** Apple quotes ~5 GB/s from NVMe on M2/M3 Ultra. Need to measure on your Mac Studio's SSD with representative expert sizes.
- **Router head size.** `L_target × E_target × topk` fan-out on MiniMax is ~32 × 128 × 8 = 32k logits per token, doable but not tiny. Consider low-rank factorization if memory becomes a problem.
- **Acceptance-rate floor.** If the draft accepts <50% of tokens, SSD streaming can't keep up. We need a calibration benchmark before committing to a draft architecture.

## 12. Success criteria for v1

- GLM-5.1 JANG_1L running on 192 GB Mac Studio at >= 10 tok/s sustained with coherent output.
- End-to-end expert cache hit rate (prefetch + prior) >= 90% on typical prompts.
- Accepted-tokens-per-verification >= 6 (i.e., >= 6/B acceptance rate).
- `jang-specd` drop-in replaces `vllm_mlx` in the vmlx Electron app for at least one target model, no panel code changes.
- Zero code, checkpoints, or training artifacts borrowed from DFlash; prior-art citation only.
