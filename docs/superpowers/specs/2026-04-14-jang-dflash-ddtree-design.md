# JANG-DFlash + DDTree — Design Spec

**Date:** 2026-04-14
**Target hardware:** Apple M3 Ultra (primary), M4 Max (dev loop)
**Target model:** MiniMax-M2.7-JANG_2L (228B MoE, 62 layers, 256 experts, top-8 routing, bf16 activations)
**Goal:** Beat the 50 tok/s baseline on MiniMax. Stretch: 200–250 tok/s on M3 Ultra.

## 1. Claim

A JANG-native reimplementation of DFlash (Chen et al., arXiv 2602.06036) combined with EAGLE-2 dynamic-draft-tree verification (arXiv 2406.16858), running entirely on Apple Silicon via MLX-Swift. The keystone primitives are:

- **Block-diffusion drafter**: 5 decoder layers, 1-step masked-token denoising, emits B parallel candidate distributions in one forward pass
- **Target-hidden KV injection**: draft attention K/V prepended with projected target hidden states from 5 evenly-spaced target layers
- **DDTree verification**: top-k per slot × lattice beam search → prefix-trie → tree-attention mask → one target forward verifies all candidates

## 2. Expected numbers

Paper reports τ (mean accepted length) ≈ 6.5–7.9 on dense Qwen3 B=16. We assume a MoE penalty (drafter softmax is flatter for MoE targets) of ~20% on τ, landing at τ ≈ 5–6.

| Platform | Baseline | × τ_eff | / (1+draft_cost) | Effective tok/s |
|---|---:|---:|---:|---:|
| M4 Max (MacBook dev loop) | 14.99 | 5.5 | 1.10 | **~75** |
| M3 Ultra (JANGTQ P15 baseline) | 41–42 | 5.5 | 1.10 | **~205** |
| M3 Ultra stretch (τ=6.5) | 42 | 6.5 | 1.10 | **~248** |

Draft cost ≈ 10% of target forward: drafter is 5 layers vs MiniMax's 62, so drafter forward is ~8% of target even with the fusion MLP and KV injection overhead.

## 3. The math we are implementing

### 3.1 Forward corruption (from BD3-LM §B.1 / MaskGIT)

Per-token, independently, at block position ℓ:

```
q(x_t^ℓ | x_0^ℓ) = Cat(x_t^ℓ ; α_t · onehot(x_0^ℓ) + (1 − α_t) · onehot(MASK))
```

Absorbing-state transition: a clean token stays clean with probability α_t, otherwise gets replaced by `[MASK]`. DFlash uses a single noise level at training: α = 1/B (i.e., anchor position stays clean, all B−1 positions after are masked). No Gaussian, no embedding-space noise.

### 3.2 Training loss (DFlash Eq. 4)

```
L = Σ_{k=1..B-1} w_k · CE(p_θ(x_k | x_t^block, h_ctx), x_k^target)
w_k = exp(−(k − 1)/γ)
γ = 7 for B=16, γ=5 for B=10, γ=4 for B=8
```

Cross-entropy against the target model's chosen tokens (greedy argmax at temperature 0). Exponential-decay weights so errors at position 0 (which reject the entire block) dominate the gradient.

### 3.3 KV injection (DFlash §4.1)

```
For each drafter layer l ∈ 1..L_draft:
    Hl = DecoderLayer_l(Hl-1, K=[h_ctx_K, Hl-1_K], V=[h_ctx_V, Hl-1_V])
```

Where `h_ctx = MLP_fuse(concat(h_target[L1], h_target[L2], ..., h_target[L5]))`, with L1..L5 evenly spaced over MiniMax's 62 layers (L1=10, L2=22, L3=34, L4=46, L5=58). The fusion MLP has input dim `5 × 3072 = 15360` and output dim `d_draft = 1536`. h_ctx is persistent across drafter layers (not re-fused per layer) and appears in K/V of every attention layer, not in the input embeddings.

### 3.4 Inference (1-step denoising)

```
# Once per speculative round:
block_input = [bonus_tok, MASK, MASK, ..., MASK]    # length B=16
logits = drafter.forward(block_input, kv_inject=h_ctx)   # [B, V]
probs = softmax(logits[1:])                          # [B-1, V], ignore anchor
```

No iteration, no noise schedule at inference. The drafter is distilled to denoise the fixed `(B−1)/B` mask ratio in one shot.

### 3.5 DDTree (EAGLE-2 §4.1)

DFlash's native verification is flat: feed B tokens to target, one forward, accept longest match. DDTree adds per-slot top-k and tree verification:

```
topk_ids, topk_vals = probs.topk(k=4, dim=-1)        # [B-1, 4]
paths = beam_topm_lattice(topk_vals, topk_ids, m=60) # top-60 paths by Π joint prob
trie = PrefixTrie(paths)                              # dedup shared prefixes
flat, tree_mask = trie.flatten_with_tree_mask()       # [N], [N, N]
target_logits = target.forward_tree_masked(flat, tree_mask)
accepted = walk_accept_path(trie, target_logits, drafter_probs)
```

`beam_topm_lattice`: positions 1..B−1 are independent given h_ctx, so joint prob of a path is just the product of per-slot marginals. Beam search over the lattice: at each slot, extend surviving beams with top-k options, rerank by joint prob, keep top-m.

`flatten_with_tree_mask`: DFS over the trie, assign a flat index to each node, record `tree_mask[i, j] = 1 iff node j is on the root-to-i path`. This is the EAGLE-2 tree-attention layout.

`walk_accept_path`: standard speculative sampling — at each tree depth, pick the highest-joint-prob child of the current node, compare `p_target / p_draft > rand`, accept or fall back.

## 4. Architecture

### 4.1 Drafter model (MLX-Swift class `JangDFlashDrafter`)

- Base: Qwen3-style transformer
- Layers: **5** (paper's reference config; ablation shows 3 → 5 gives +0.26 τ)
- Hidden: **1536** (chosen to match MiniMax's h_ctx after fusion)
- Heads: 12 attention, 4 KV (3:1 GQA)
- FFN: 4096
- Vocab: 200064 (matches MiniMax)
- Params: ~180M
- Weights format: JANG-compatible safetensors, ~360 MB fp16, ~90 MB at JANG_4M
- **New primitive:** `forward_one_step(block, kv_inject)` — runs 1 layer of attention where K/V are the concatenation of injected `h_ctx` and the block's own K/V. Causal within the block, bidirectional against h_ctx.

### 4.2 Target hidden-tap

Modify `MiniMax.swift` to optionally return a `[5, T, 3072]` tensor of hidden states from layers 10, 22, 34, 46, 58. New forward signature:

```swift
func callAsFunction(
    _ inputs: MLXArray,
    cache: [KVCache]?,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
    tapLayers: [Int]?              // NEW — nil for normal decode
) -> (logits: MLXArray, tappedHiddens: MLXArray?)
```

Only populated when the spec-dec path asks. Zero cost when disabled.

### 4.3 Fusion MLP (`JangDFlashFusion`)

```
h_fused = Linear(5 * 3072, 1536)(concat(h_taps, dim=-1))  
h_ctx  = RMSNorm(1536)(SwiGLU(h_fused))
```

Trained alongside the drafter. Stored in the drafter checkpoint.

### 4.4 DDTree in Swift

`DDTreeBuilder` class in `vMLXLMCommon/DFlash/`:
- `build(probs: MLXArray, k: Int, m: Int) -> (flatTokens, treeMask, trieMap)`
- Beam search on the CPU (m=60 × B=16 × k=4 = 3840 operations, sub-millisecond even in pure Swift)
- Tree mask as MLXArray `[N, N]` boolean

### 4.5 Tree-attention primitive

MLX has no tree-attention built in. Build via SDPA additive bias:

```swift
let bias = MLXArray.full([N, N], -Float.infinity)
// set bias[i, j] = 0 wherever j is an ancestor of i in the trie
let output = MLXFast.scaledDotProductAttention(
    queries: q, keys: k, values: v,
    scale: 1.0 / sqrt(Float(headDim)),
    mask: .arrays(bias)
)
```

Trie ancestry mask is computed on CPU, pushed to GPU once per step. Verified identical to hand-written attention on small cases before wiring into the hot path.

### 4.6 Spec-dec loop integration

Extend `vMLXLMCommon/Evaluate.swift` `SpeculativeDecoder`:

```swift
struct JangDFlashConfig {
    var blockSize: Int = 16          // B
    var topK: Int = 4                // per-slot
    var numPaths: Int = 60           // m
    var tapLayerIDs: [Int] = [10, 22, 34, 46, 58]
}

class JangDFlashSpecDec {
    let target: any LanguageModel
    let drafter: JangDFlashDrafter
    let fusion: JangDFlashFusion
    let cfg: JangDFlashConfig
    
    func step(state: DecodeState) -> [Int] { /* returns 1..B accepted tokens */ }
}
```

## 5. Training recipe

### 5.1 Data

Mixed distillation corpus, ~50k sequences of length 1024:
- GSM8K train split (~7k)
- HumanEval test cases (~0.5k)
- MMLU-STEM train (~10k)
- OpenOrca filtered reasoning traces (~30k)
- MiniMax-specific: run MiniMax on 5k chain-of-thought prompts and store (hidden_states, target_tokens) pairs as training data

### 5.2 Offline dataset generation (Python, runs on M3 Ultra)

Script `jang_tools/dflash/distill_data.py`:

```python
for prompt in corpus:
    h_taps, tokens = minimax_jang.generate_with_taps(
        prompt, max_tokens=256, taps=[10, 22, 34, 46, 58])
    save(f"distill/{uuid}.safetensors", {"h_taps": h_taps, "tokens": tokens})
```

Expected dataset size: 50k × 256 × (5 × 3072 × 2 bytes) = ~780 GB. Store sharded on external SSD.

### 5.3 Training (5090 server, PyTorch)

```python
for step in range(1, 2001):
    batch = sample_batch(distill_data)
    h_ctx = fusion_mlp(batch.h_taps)  # [B, T, 1536]
    
    # Random-anchor masking (DFlash §4.2)
    anchors = torch.randint(0, T - B, (batch_size,))
    block_ids = batch.tokens[anchors : anchors + B]
    
    block_input = block_ids.clone()
    block_input[:, 1:] = MASK_ID  # keep anchor at position 0
    
    logits = drafter(block_input, kv_inject=h_ctx[anchors:anchors+B])
    
    loss = weighted_masked_ce(logits[:, 1:], block_ids[:, 1:], weights=w_k)
    loss.backward()
    optimizer.step()
```

Expected wall: ~4 hours on RTX 5090 for 2000 steps. Checkpoint format: safetensors, direct load into MLX via existing `JangLoader`.

### 5.4 Validation

- τ on held-out GSM8K prompts: target ≥ 5.5
- Acceptance rate at position 1: target ≥ 0.70
- GSM8K-5shot accuracy drift from MiniMax baseline: target ≤ 1 pp

## 6. MLX gaps and workarounds

| Needed | Status in MLX-Swift | Workaround |
|---|---|---|
| Tree attention mask | No primitive | Additive `-inf` bias in SDPA mask parameter |
| KV injection (persistent context K/V) | No primitive | Pre-build K/V tensors by running `W_k`, `W_v` on `h_ctx` once per round; concat with block K/V inside drafter forward |
| Hidden state tap from mid-layer | No hook | Add optional `tapLayers` parameter to `MiniMax.callAsFunction` |
| Masked-input forward | Works (MASK is just another token ID) | Use vocab extension: vocab[200064] = MASK, embedding layer gets 1 extra row init-zero |
| Tree-attention bias → `additive_bias` | SDPA fast path supports it in MLX ≥ 0.20 | Already present in our vendored mlx-swift |

## 7. Risks and mitigations

1. **JANG-2L bf16 hiddens ≠ FP16 target distribution** — distillation must use JANG-2L's own hiddens, not a FP16 reference. Mitigation: `distill_data.py` runs the quantized model, not the reference.

2. **MoE flatter softmax → lower τ** — paper reports 6.5–7.9 for dense; we assume 5–6 for MoE. Mitigation: (a) larger B (16 vs 8), (b) top-k=4 per slot instead of top-1 to recover branches the drafter was unsure about.

3. **Tree-attention overhead > gain** — if the trie has 60 nodes, target forward on 60 tokens is slower than on 16 tokens. Mitigation: benchmark m=16, 32, 60 and pick the crossover. Fall back to flat DFlash if tree verification hurts.

4. **Distillation data size** — 780 GB. Mitigation: stream from external SSD, batch-on-demand, use fp16 for hidden states (half the size).

5. **Drafter training compute** — 5090 at 2k steps × batch 16 × seq 1024 ≈ 4 GPU-hours. Fits in overnight budget.

6. **KV cache complexity during tree verification** — target's KV cache must handle "branching" (all m tree nodes share the same prefix, then diverge). Mitigation: v1 rebuilds the cache from scratch per speculative round (slightly slower but correct), v2 adds copy-on-write KV pages.

## 8. Phase split

**Phase 1 (this plan):** JANG-DFlash drafter + DDTree verification on in-RAM MiniMax, targeting ≥ 60 tok/s on MacBook / ≥ 200 tok/s on M3 Ultra.

**Phase 2 (separate plan, later):** Layer SSD-resident hot-core + cold-tail onto a 400B+ target (Qwen3.5-397B, DeepSeek-V3). Reuse the drafter as-is. Add the expert-prefetch worker. Target: 10–20 tok/s on models that currently run at ~1 tok/s or not at all.

**Phase 3 (later still):** ANE experiments for the draft model specifically — ANE has its own memory bus, so running the drafter on ANE while Metal runs target verify is "free compute" even if ANE's per-op speed is 3× slower than Metal.

## 9. Out of scope for Phase 1

- ANE integration
- SSD streaming (MiniMax fits in RAM)
- Tree-structured DFlash drafter (DFlash is depth-1 by construction; our tree is over per-slot top-k, not over draft depth)
- Sampling with temperature > 0 (only greedy, matches paper and bench target)
- Dynamic B (train at 16, infer at 16)
- Draft model JANG-quantization (ship as fp16, revisit later)

## 10. Acceptance criteria

Phase 1 is complete when:

1. A JANG-DFlash drafter checkpoint exists, trained against MiniMax-JANG_2L, on HuggingFace under `JANGQ-AI/MiniMax-M2.7-JANG_2L-DFlash-B16`
2. `vmlxctl serve -m minimax-dflash -p 8000` runs without crash, outputs coherent text
3. Measured pure decode tok/s on MacBook M4 Max ≥ 60 (baseline 14.99, target 4×)
4. Measured pure decode tok/s on M3 Ultra ≥ 200 (baseline ~42, target 4.8×)
5. GSM8K accuracy on 200 prompts drifts ≤ 1 pp from MiniMax-JANG_2L baseline
6. Tree verification overhead is measured and documented; flat-fallback path exists
7. All code lives in `/Users/eric/vmlx/swift/Sources/vMLXLMCommon/DFlash/` and `/Users/eric/jang/jang-tools/jang_tools/dflash/`
