// DeepseekV4JANGTQ.swift
//
// Swift runtime scaffold for DeepSeek-V4-Flash (and variants).
// Mirrors `jang-tools/jang_tools/dsv4_prune/mlx_model.py` (Python, coherent
// on JANG_2L / JANG4 / JANGTQ2 / JANGTQ4 bundles). All 13 runtime bug fixes
// from `research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md` must be preserved here.
//
// STATUS: scaffold only — math signatures + TODO markers. Wire into
// JANGTQModel/JANGTQAttentionBlock/JANGTQMoEBlock dispatch when implementing.
//
// References:
// - research/DSV-FAMILY-RUNTIME-GUIDE.md §28–32
// - research/DSV-EXHAUSTIVE-VARIABLES-GUIDE.md
// - jang-tools/jang_tools/dsv4_prune/mlx_model.py
// - /tmp/deepseek_v4_reference.py (PR #1192, authoritative)
// - encoding_dsv4.py (chat encoder) in bundle encoding/ dir
//
// Non-negotiable math (all verified bit-exact vs PR #1192):
//   HC collapse:  pre  = sigmoid(mixes*scale[0] + base[:H]) + eps   (NO sum-normalize)
//                 post = 2 * sigmoid(mixes*scale[1] + base[H:2H])    (NO eps)
//                 comb = softmax(... + scale[2]*mixes[2H:], lastAxis) + eps
//                        col-norm
//                        repeat (sinkhorn_iters - 1)× { row-norm; col-norm }
//   HC expand:    y[i,d] = post[i]*blockOut[d] + sum_j comb[i,j]*residual[j,d]
//                 (contract LAST axis of comb, equiv to `comb @ residual`)
//   YaRN RoPE:    high = min(ceil(correction_dim(betaSlow)), dim - 1)   (NOT dim/2-1)
//                 smooth = 1 - clip(ramp, 0, 1)
//                 freqs = invFreq/factor*(1-smooth) + invFreq*smooth
//   Per-layer rope theta:
//                 compressRatio > 0  →  compressRopeTheta (160000) + YaRN
//                 compressRatio == 0 →  ropeTheta (10000), NO YaRN
//   Gate logits:  matmul in fp32 (cast x and weight to fp32 before @)
//   Attention:    native SDPA with `sinks=` (attn_sink: per-head learned bias)
//                 mask must be explicit array (not "causal" string)
//                 window_size = sliding_window
//   MoE weight:   inds.astype(uint32) before gather_qmm
//   SwiGLU:       min(gate, limit) * clip(up, -limit, limit) then silu * up
//   Grouped O:    einsum bsgd,grd->bsgr then concat groups, then wo_b

import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - Config

/// DSV4 architecture + tokenizer + quant configuration.
/// Mirrors `ModelArgs` in `mlx_model.py`.
public struct DeepseekV4Config: Decodable {
    public let vocabSize: Int                    // 129280
    public let hiddenSize: Int                   // 4096
    public let numHiddenLayers: Int              // 43
    public let numAttentionHeads: Int            // 64
    public let numKeyValueHeads: Int             // 1
    public let headDim: Int                      // 512
    public let qkRopeHeadDim: Int                // 64
    public let qLoraRank: Int                    // 1024
    public let oLoraRank: Int                    // 1024
    public let oGroups: Int                      // 8
    public let nRoutedExperts: Int               // 256
    public let nSharedExperts: Int               // 1
    public let numExpertsPerTok: Int             // 6
    public let moeIntermediateSize: Int          // 2048
    public let numHashLayers: Int                // 3
    public let scoringFunc: String               // "sqrtsoftplus"
    public let normTopkProb: Bool                // true
    public let routedScalingFactor: Float        // 1.5
    public let swigluLimit: Float                // 10.0
    public let hcMult: Int                       // 4
    public let hcSinkhornIters: Int              // 20
    public let hcEps: Float                      // 1e-6
    public let ropeTheta: Float                  // 10000.0
    public let compressRopeTheta: Float          // 160000.0
    public let ropeScaling: RopeScaling
    public let maxPositionEmbeddings: Int        // 1048576
    public let slidingWindow: Int                // 128
    public let rmsNormEps: Float                 // 1e-6
    public let compressRatios: [Int]             // length 43+1 (or 44), values ∈ {0, 4, 128}
    public let indexNHeads: Int                  // 64
    public let indexHeadDim: Int                 // 128
    public let indexTopk: Int                    // 512

    public struct RopeScaling: Decodable {
        public let factor: Float                 // 16.0
        public let originalMaxPositionEmbeddings: Int  // 65536
        public let betaFast: Int                 // 32
        public let betaSlow: Int                 // 1
        public let type: String                  // "yarn" | "deepseek_yarn"
    }
}

// MARK: - RoPE

/// YaRN-aware RoPE. Computes cos/sin on-the-fly per call.
/// CRITICAL: `high = min(ceil(correction_dim(betaSlow)), dim - 1)`
/// NOT `dim // 2 - 1` — that was a subtle bug that caused 11.8% RMS drift.
public final class DeepseekV4RoPE {
    let dims: Int           // rope_head_dim (64)
    let base: Float         // rope_theta or compress_rope_theta
    let invFreq: [Float]    // length dims/2

    public init(dims: Int, base: Float, scaling: DeepseekV4Config.RopeScaling?, maxPosEmb: Int) {
        self.dims = dims
        self.base = base

        var inv = (0..<dims/2).map { i in
            1.0 / pow(base, Float(2 * i) / Float(dims))
        }

        if let s = scaling, s.type == "yarn" || s.type == "deepseek_yarn" {
            let correctionDim: (Float) -> Float = { n in
                Float(dims) * log(Float(s.originalMaxPositionEmbeddings) / (n * 2.0 * .pi)) / (2.0 * log(base))
            }
            let low = max(floor(correctionDim(Float(s.betaFast))), 0.0)
            var high = min(ceil(correctionDim(Float(s.betaSlow))), Float(dims - 1))  // NOT dims/2 - 1
            if low == high { high += 0.001 }

            for i in 0..<inv.count {
                let rampRaw = (Float(i) - low) / (high - low)
                let ramp = min(max(rampRaw, 0), 1)
                let smooth = 1 - ramp
                inv[i] = inv[i] / s.factor * (1 - smooth) + inv[i] * smooth
            }
        }
        self.invFreq = inv
    }

    /// Apply rope (or inverse-rope) to the LAST `dims` of `x`.
    /// x shape: (..., dims). Treats last axis as interleaved (x0, x1, x0, x1, ...).
    /// y_even = x_even * cos - x_odd * sin
    /// y_odd  = x_even * sin + x_odd * cos
    public func apply(_ x: MTLBuffer, shape: [Int], offset: Int, inverse: Bool) -> MTLBuffer {
        // TODO: Metal kernel. Take cos/sin tables computed from positions
        // (arange(offset, offset+L) outer invFreq). Write rotated output.
        fatalError("TODO: Metal-native rope apply")
    }
}

/// Partial rope: split x along last axis into (nope, pe), rope only the pe part.
public func applyPartialRope(_ x: MTLBuffer, xShape: [Int], rope: DeepseekV4RoPE,
                             offset: Int, inverse: Bool = false) -> MTLBuffer {
    // TODO: equivalent to:
    //   let nopeDim = xShape.last! - rope.dims
    //   let (nope, pe) = split(x, axis: -1, at: nopeDim)
    //   let peRot = rope.apply(pe, shape: pe.shape, offset: offset, inverse: inverse)
    //   return concat([nope, peRot], axis: -1)
    fatalError("TODO")
}

// MARK: - HyperConnection (mHC)

/// Manifold-Constrained Hyper-Connection. DSV4 unique.
/// Per block: `collapse(h)` → (x, post, comb); block processes x; `expand(block_out, residual, post, comb)` → h_new.
public final class HyperConnection {
    let config: DeepseekV4Config
    let fn: MTLBuffer          // shape ((2+hc_mult)*hc_mult, hc_mult*hidden_size) fp32
    let base: MTLBuffer        // shape ((2+hc_mult)*hc_mult,) fp32
    let scale: MTLBuffer       // shape (3,) fp32  [preScale, postScale, combScale]

    public init(config: DeepseekV4Config, fn: MTLBuffer, base: MTLBuffer, scale: MTLBuffer) {
        self.config = config
        self.fn = fn
        self.base = base
        self.scale = scale
    }

    /// Collapse (B, L, H, D) residual → (B, L, D) plus (post, comb).
    /// Math:
    ///   flat = x.reshape(B, L, H*D).float()
    ///   rsqrt = rsqrt(mean(flat², -1) + eps)
    ///   mixes = (flat @ fn.T) * rsqrt
    ///   (pre, post, comb) = hcSplitSinkhorn(mixes, scale, base, H, iters, eps)
    ///   collapsed = sum_h pre[h] * x[h]
    ///   return (collapsed, post, comb)
    public func collapse(_ h: MTLBuffer, shape: [Int]) -> (MTLBuffer, MTLBuffer, MTLBuffer) {
        fatalError("TODO: Metal HC collapse")
    }

    /// Expand: combine block_out + mixed residual → new residual stream.
    /// Math:
    ///   y[b,s,i,d] = post[i] * blockOut[d] + sum_j comb[i,j] * residual[j,d]
    /// CRITICAL: contract comb's LAST axis (not first). Equivalent to `comb @ residual`.
    public func expand(blockOut: MTLBuffer, residual: MTLBuffer, post: MTLBuffer, comb: MTLBuffer,
                       shape: [Int]) -> MTLBuffer {
        fatalError("TODO: Metal HC expand via einsum bsij,bsjd->bsid")
    }
}

/// Sinkhorn-normalize `comb`. Match PR #1192 exactly.
/// Output: (pre, post, comb)
///   pre  = sigmoid(mixes[..., :H] * scale[0] + base[:H]) + eps    NO sum-normalize
///   post = 2 * sigmoid(mixes[..., H:2H] * scale[1] + base[H:2H])   NO eps, factor of 2
///   comb = sinkhorn-doubly-stochastic (softmax init + col-norm + (iters-1) × rounds)
public func hcSplitSinkhorn(mixes: MTLBuffer, scale: MTLBuffer, base: MTLBuffer,
                            hcMult: Int, iters: Int = 20, eps: Float = 1e-6,
                            shape: [Int]) -> (MTLBuffer, MTLBuffer, MTLBuffer) {
    // TODO: Metal kernel
    // Key property: after `iters` iterations, comb is doubly-stochastic.
    // Use precise=true softmax (or fp32 computation).
    fatalError("TODO")
}

// MARK: - Attention (MLA + grouped O + sinks + compressor/indexer)

public final class DeepseekV4Attention {
    let config: DeepseekV4Config
    let layerId: Int
    let compressRatio: Int        // from config.compress_ratios[layer_id]

    // Weights
    let wqA: MTLBuffer            // (q_lora_rank, hidden)       8-bit affine typical
    let qNorm: MTLBuffer          // (q_lora_rank,)              fp16 passthrough
    let wqB: MTLBuffer            // (n_heads * head_dim, q_lora_rank) 8-bit affine
    let wkv: MTLBuffer            // (head_dim, hidden)          8-bit affine
    let kvNorm: MTLBuffer         // (head_dim,)                 fp16 passthrough
    let woA: MTLBuffer            // (o_groups * o_lora_rank, n_heads*head_dim/o_groups)
    let woB: MTLBuffer            // (hidden, o_groups * o_lora_rank)
    let attnSink: MTLBuffer       // (n_heads,) fp32 per-head learned sink logit

    // RoPE (per-layer — switches config based on compressRatio)
    let rope: DeepseekV4RoPE

    // Optional compressor/indexer (only when compressRatio > 0)
    let compressor: Compressor?
    let indexer: Indexer?

    public init(config: DeepseekV4Config, layerId: Int, weights: WeightBundle) {
        self.config = config
        self.layerId = layerId
        self.compressRatio = config.compressRatios[safe: layerId] ?? 0

        // Per-layer RoPE. DSV4 convention:
        //   compressRatio > 0 → compressRopeTheta=160000 + YaRN
        //   compressRatio == 0 → ropeTheta=10000, NO YaRN
        let base: Float = compressRatio > 0 ? config.compressRopeTheta : config.ropeTheta
        let scaling: DeepseekV4Config.RopeScaling? = compressRatio > 0 ? config.ropeScaling : nil
        self.rope = DeepseekV4RoPE(dims: config.qkRopeHeadDim, base: base, scaling: scaling,
                                    maxPosEmb: config.maxPositionEmbeddings)

        // Load all weight tensors from bundle
        self.wqA = weights["self_attn.wq_a.weight"]
        self.qNorm = weights["self_attn.q_norm.weight"]
        self.wqB = weights["self_attn.wq_b.weight"]
        self.wkv = weights["self_attn.wkv.weight"]
        self.kvNorm = weights["self_attn.kv_norm.weight"]
        self.woA = weights["self_attn.wo_a.weight"]
        self.woB = weights["self_attn.wo_b.weight"]
        self.attnSink = weights["self_attn.attn_sink"]

        if compressRatio > 0 {
            self.compressor = Compressor(config: config, ratio: compressRatio, weights: weights, prefix: "self_attn.compressor")
            self.indexer = compressRatio == 4 ? Indexer(config: config, ratio: compressRatio, weights: weights, prefix: "self_attn.indexer") : nil
        } else {
            self.compressor = nil
            self.indexer = nil
        }
    }

    /// Forward pass. Mirrors `DeepseekV4Attention.__call__` in mlx_model.py.
    /// input x: (B, L, hidden)
    /// mask: explicit array causal mask (B, 1, L, T) or equivalent — NOT "causal" string
    /// cache: DeepseekV4Cache (sliding-window local + compressor state) or plain KVCache
    public func forward(_ x: MTLBuffer, xShape: [Int], mask: MTLBuffer?, cache: AttentionCache?) -> MTLBuffer {
        // Steps (exact order from Python):
        //
        // 1. Q low-rank + per-head RMSNorm:
        //    qResidual = qNorm(wqA(x))
        //    q = wqB(qResidual).reshape(B, L, nHeads, headDim)
        //    q = q * rsqrt((q.f32 ** 2).mean(-1) + eps)  <- fp32 rsqrt
        //    q = q.cast(xDtype).transpose(0, 2, 1, 3)   <- (B, H, L, D)
        //
        // 2. KV (single head) + norm:
        //    kv = kvNorm(wkv(x)).reshape(B, L, 1, headDim).transpose(0, 2, 1, 3)
        //
        // 3. Partial rope on q and kv (last qk_rope_head_dim dims):
        //    q = applyPartialRope(q, rope, offset=cache.offset)
        //    kv = applyPartialRope(kv, rope, offset=cache.offset)
        //
        // 4. Cache update:
        //    if cache is not None:
        //        kv, _ = cache.updateAndFetch(kv, kv)
        //    fullKv = kv
        //
        // 5. (Optional) Compressor + Indexer for compress_ratio > 0:
        //    FAST-PATH: skip entirely when v4Cache is None AND L < compress_ratio
        //    (output would be empty, saves ~150 matmuls per token across 41 layers)
        //    Otherwise:
        //        pooled = compressor(x, rope, v4Cache, offset)
        //        if indexer and pooled.len > 0:
        //            topk = indexer(x, qResidual, rope, rope, v4Cache, offset)
        //            pooled = gather(pooled, topk)
        //        fullKv = concat([fullKv, pooled], axis=2)
        //        mask = pad(mask, ones) if longer than mask.width
        //
        // 6. Native SDPA with attention-sink:
        //    out = sdpa(q, fullKv, fullKv, scale=1/sqrt(headDim), mask=mask,
        //               sinks=attnSink.cast(qDtype))
        //
        // 7. Inverse RoPE on output:
        //    out = applyPartialRope(out, rope, offset, inverse: true)
        //
        // 8. Grouped O projection (out shape (B, H, L, D) → (B, L, hidden)):
        //    out = out.transpose(0, 2, 1, 3).reshape(B, L, nHeads*headDim)
        //    out = groupedOProjection(out)   <- einsum bsgd,grd->bsgr then reshape
        //    return woB(out)
        fatalError("TODO")
    }

    /// Grouped low-rank O projection.
    /// out.reshape(B, L, oGroups, groupFeat) where groupFeat = nHeads*headDim/oGroups
    /// For quantized wo_a: use mx.quantized_matmul per-group batched
    /// Else: einsum("bsgd,grd->bsgr", out, wo_a.reshape(oGroups, oLoraRank, groupFeat))
    /// Then: reshape (B, L, oGroups*oLoraRank)
    func groupedOProjection(_ out: MTLBuffer, outShape: [Int]) -> MTLBuffer {
        fatalError("TODO")
    }
}

// MARK: - Compressor + Indexer

public final class Compressor {
    let compressRatio: Int
    let headDim: Int
    let outDim: Int  // headDim * (2 if overlap else 1); overlap = ratio==4
    let wkv: MTLBuffer
    let wgate: MTLBuffer
    let ape: MTLBuffer          // (compressRatio, outDim) fp32
    let norm: MTLBuffer         // (headDim,) fp16

    public init(config: DeepseekV4Config, ratio: Int, weights: WeightBundle, prefix: String) {
        self.compressRatio = ratio
        self.headDim = config.headDim
        self.outDim = headDim * (ratio == 4 ? 2 : 1)
        self.wkv = weights["\(prefix).wkv.weight"]
        self.wgate = weights["\(prefix).wgate.weight"]
        self.ape = weights["\(prefix).ape"]
        self.norm = weights["\(prefix).norm.weight"]
    }

    /// Pool residual stream windows into compressed context.
    /// Returns (B, W, headDim) pooled, where W = usableLen / compressRatio.
    /// For short L < compressRatio AND cache=nil → returns (B, 0, headDim).
    public func forward(x: MTLBuffer, xShape: [Int], rope: DeepseekV4RoPE,
                        cache: DeepseekV4Cache?, startPos: Int, stateKey: String = "compressor") -> MTLBuffer {
        fatalError("TODO")
    }
}

public final class Indexer {
    let nHeads: Int
    let headDim: Int
    let indexTopk: Int
    let wqB: MTLBuffer
    let weightsProj: MTLBuffer
    let compressor: Compressor

    public init(config: DeepseekV4Config, ratio: Int, weights: WeightBundle, prefix: String) {
        self.nHeads = config.indexNHeads
        self.headDim = config.indexHeadDim
        self.indexTopk = config.indexTopk
        self.wqB = weights["\(prefix).wq_b.weight"]
        self.weightsProj = weights["\(prefix).weights_proj.weight"]
        self.compressor = Compressor(config: config, ratio: ratio, weights: weights, prefix: "\(prefix).compressor")
    }

    /// Produce topk indices into the pooled context.
    /// Returns nil if pooled is empty.
    public func forward(x: MTLBuffer, qResidual: MTLBuffer, rope: DeepseekV4RoPE,
                        positionRope: DeepseekV4RoPE, cache: DeepseekV4Cache?, startPos: Int) -> MTLBuffer? {
        fatalError("TODO")
    }
}

// MARK: - MoE

public final class MoEGate {
    let args: DeepseekV4Config
    let layerId: Int
    let isHashLayer: Bool                 // layerId < numHashLayers
    let weight: MTLBuffer                 // (nRoutedExperts, hiddenSize) fp16/fp32
    let tid2eid: MTLBuffer?               // (vocabSize, topk) int32 if hash
    let biasLogit: MTLBuffer?             // (nRoutedExperts,) fp32 if non-hash (e_score_correction_bias)

    public init(config: DeepseekV4Config, layerId: Int, weights: WeightBundle) {
        self.args = config
        self.layerId = layerId
        self.isHashLayer = layerId < config.numHashLayers
        self.weight = weights["mlp.gate.weight"]
        if isHashLayer {
            self.tid2eid = weights["mlp.gate.tid2eid"]
            self.biasLogit = nil
        } else {
            self.tid2eid = nil
            self.biasLogit = weights["mlp.gate.bias"]  // legacy name; reference uses e_score_correction_bias
        }
    }

    /// Returns (inds, scores).
    /// inds:   (B, L, topk) uint32  <- gather_qmm requires uint32
    /// scores: (B, L, topk) fp32 scaled by routedScalingFactor
    public func forward(x: MTLBuffer, xShape: [Int], inputIds: MTLBuffer) -> (MTLBuffer, MTLBuffer) {
        // CRITICAL: gate matmul must be fp32.
        //   gates = x.f32 @ weight.f32.T     // (B, L, nExperts) fp32
        //   scores = sqrt(log1p(exp(gates))) // sqrtsoftplus
        //
        // Hash path:
        //   inds = tid2eid[inputIds].cast(int32)
        //   weights = takeAlongAxis(scores, inds)
        //
        // Non-hash path:
        //   biased = scores + biasLogit
        //   inds = argpartition(-biased, kth=topk-1)[..., :topk].cast(int32)
        //   weights = takeAlongAxis(scores, inds)
        //
        // Both paths:
        //   weights = weights / sum(weights, axis=-1, keepdim=true)
        //   weights = weights * routedScalingFactor
        //   inds = inds.cast(uint32)
        //   return (inds, weights)
        fatalError("TODO")
    }
}

public final class DeepseekV4MoE {
    let gate: MoEGate
    let switchMLP: SwitchGLU              // existing JANGTQ Swift type, check shape
    let sharedExperts: DeepseekV4MLP

    public init(config: DeepseekV4Config, layerId: Int, weights: WeightBundle) {
        self.gate = MoEGate(config: config, layerId: layerId, weights: weights)
        // SwitchGLU activation = LimitedSwiGLU(limit=10.0); swap gate/up semantics inside
        self.switchMLP = SwitchGLU(inputDims: config.hiddenSize,
                                    hiddenDims: config.moeIntermediateSize,
                                    numExperts: config.nRoutedExperts,
                                    activation: LimitedSwiGLU(limit: config.swigluLimit),
                                    weights: weights,
                                    prefix: "mlp.switch_mlp")
        self.sharedExperts = DeepseekV4MLP(config: config, intermediateSize: config.moeIntermediateSize,
                                            weights: weights, prefix: "mlp.shared_experts")
    }

    /// Forward. Math:
    ///   (inds, scores) = gate(x, inputIds)
    ///   y = switchMLP(x, inds)                       // (B, L, topk, hidden)
    ///   y = sum_k scores[k] * y[k]                   // (B, L, hidden)
    ///   y = y + sharedExperts(x)
    public func forward(x: MTLBuffer, xShape: [Int], inputIds: MTLBuffer) -> MTLBuffer {
        fatalError("TODO")
    }
}

public final class DeepseekV4MLP {
    let gateProj: MTLBuffer
    let upProj: MTLBuffer
    let downProj: MTLBuffer
    let swigluLimit: Float

    public init(config: DeepseekV4Config, intermediateSize: Int, weights: WeightBundle, prefix: String) {
        self.gateProj = weights["\(prefix).gate_proj.weight"]
        self.upProj = weights["\(prefix).up_proj.weight"]
        self.downProj = weights["\(prefix).down_proj.weight"]
        self.swigluLimit = config.swigluLimit
    }

    /// y = down_proj( _limited_swiglu(gate_proj(x), up_proj(x), limit) )
    public func forward(x: MTLBuffer, xShape: [Int]) -> MTLBuffer {
        fatalError("TODO")
    }
}

/// SwiGLU with symmetric-up and max-only-gate clamp. silu(clamp(gate, ≤limit)) * clamp(up, ±limit).
public final class LimitedSwiGLU {
    let limit: Float
    public init(limit: Float) { self.limit = limit }

    /// NOTE: input order from MLX SwitchGLU is `(upOut, gateOut)`.
    /// Math equivalent to: silu(min(gateOut, limit)) * clip(upOut, -limit, limit)
    public func apply(_ upOut: MTLBuffer, _ gateOut: MTLBuffer) -> MTLBuffer {
        fatalError("TODO")
    }
}

// MARK: - Block + Model + Cache

public final class DeepseekV4Block {
    let selfAttn: DeepseekV4Attention
    let mlp: DeepseekV4MoE
    let inputLayernorm: MTLBuffer                // RMSNorm weight (hiddenSize,)
    let postAttentionLayernorm: MTLBuffer
    let hcAttn: HyperConnection
    let hcFfn: HyperConnection

    public init(config: DeepseekV4Config, layerId: Int, weights: WeightBundle) {
        self.selfAttn = DeepseekV4Attention(config: config, layerId: layerId, weights: weights)
        self.mlp = DeepseekV4MoE(config: config, layerId: layerId, weights: weights)
        self.inputLayernorm = weights["input_layernorm.weight"]
        self.postAttentionLayernorm = weights["post_attention_layernorm.weight"]
        self.hcAttn = HyperConnection(config: config,
                                       fn: weights["hc_attn_fn"],
                                       base: weights["hc_attn_base"],
                                       scale: weights["hc_attn_scale"])
        self.hcFfn = HyperConnection(config: config,
                                      fn: weights["hc_ffn_fn"],
                                      base: weights["hc_ffn_base"],
                                      scale: weights["hc_ffn_scale"])
    }

    /// Block forward:
    ///   residual = h                              // (B, L, hc_mult, hidden)
    ///   (x, post, comb) = hcAttn.collapse(h)      // (B, L, hidden)
    ///   x = rmsnorm(x, inputLayernorm)
    ///   x = selfAttn(x, mask, cache)
    ///   h = hcAttn.expand(x, residual, post, comb)
    ///
    ///   residual = h
    ///   (x, post, comb) = hcFfn.collapse(h)
    ///   x = rmsnorm(x, postAttentionLayernorm)
    ///   x = mlp(x, inputIds)
    ///   h = hcFfn.expand(x, residual, post, comb)
    ///   return h
    public func forward(h: MTLBuffer, hShape: [Int], mask: MTLBuffer?, cache: AttentionCache?,
                        inputIds: MTLBuffer) -> MTLBuffer {
        fatalError("TODO")
    }
}

public final class DeepseekV4Model {
    let config: DeepseekV4Config
    let embed: MTLBuffer                          // (vocabSize, hidden) quantized typical
    let layers: [DeepseekV4Block]
    let norm: MTLBuffer                           // final RMSNorm
    let hcHead: HyperHead

    public init(config: DeepseekV4Config, weights: WeightBundle) {
        self.config = config
        self.embed = weights["model.embed.weight"]
        self.layers = (0..<config.numHiddenLayers).map { i in
            DeepseekV4Block(config: config, layerId: i,
                            weights: weights.withPrefix("model.layers.\(i)."))
        }
        self.norm = weights["model.norm.weight"]
        self.hcHead = HyperHead(config: config,
                                 fn: weights["model.hc_head_fn"],
                                 base: weights["model.hc_head_base"],
                                 scale: weights["model.hc_head_scale"])
    }

    /// Full forward.
    ///   h = embed[inputIds]                      // (B, L, hidden)
    ///   h = tile(h.unsqueeze(-2), (1,1,hc_mult,1))  // (B, L, hc_mult, hidden)
    ///   mask = createAttentionMask(h[:,:,0,:], cache, windowSize=slidingWindow, returnArray=true)
    ///   for (layer, c) in zip(layers, cache) {
    ///       h = layer(h, mask, c, inputIds)
    ///   }
    ///   h = hcHead.reduce(h)                     // (B, L, hidden) — HyperHead folds 4 copies back
    ///   return rmsnorm(h, norm)
    public func forward(inputIds: MTLBuffer, cache: [AttentionCache?]?) -> MTLBuffer {
        fatalError("TODO")
    }
}

/// Final HyperConnection fold: (B, L, H, D) → (B, L, D) via a single learned sigmoid-weighted sum.
/// Math:
///   flat = x.reshape(B, L, H*D).float()
///   rsqrt = rsqrt(mean(flat², -1) + eps)
///   mixes = (flat @ fn.T) * rsqrt     <- fn shape (H, H*D)
///   pre = sigmoid(mixes * scale[0] + base) + eps
///   y = sum_h pre[h] * x[h]
public final class HyperHead {
    let config: DeepseekV4Config
    let fn: MTLBuffer                             // (hc_mult, hc_mult*hidden)
    let base: MTLBuffer                           // (hc_mult,)
    let scale: MTLBuffer                          // (1,)

    public init(config: DeepseekV4Config, fn: MTLBuffer, base: MTLBuffer, scale: MTLBuffer) {
        self.config = config
        self.fn = fn
        self.base = base
        self.scale = scale
    }
    public func reduce(_ x: MTLBuffer, xShape: [Int]) -> MTLBuffer { fatalError("TODO") }
}

// MARK: - Cache

public protocol AttentionCache: AnyObject {
    var offset: Int { get }
    func updateAndFetch(_ keys: MTLBuffer, _ values: MTLBuffer) -> (MTLBuffer, MTLBuffer)
}

/// DSV4 cache = sliding-window local KV + compressor state + indexer state.
/// For short prompts (< sliding_window), RotatingKVCache(max_size=128) is equivalent.
public final class DeepseekV4Cache: AttentionCache {
    let local: RotatingKVCache
    var compressorState: (bufferKV: MTLBuffer?, bufferGate: MTLBuffer?, pooled: MTLBuffer?)
    var indexerState: (bufferKV: MTLBuffer?, bufferGate: MTLBuffer?, pooled: MTLBuffer?)

    public init(slidingWindow: Int) {
        self.local = RotatingKVCache(maxSize: slidingWindow, keep: 0)
        self.compressorState = (nil, nil, nil)
        self.indexerState = (nil, nil, nil)
    }

    public var offset: Int { local.offset }
    public func updateAndFetch(_ keys: MTLBuffer, _ values: MTLBuffer) -> (MTLBuffer, MTLBuffer) {
        local.updateAndFetch(keys, values)
    }

    // TODO: accumulateWindows, updatePool — see reference PR #1192 DeepseekV4Cache
}

// MARK: - Placeholders (wire these into existing runtime types)

// RotatingKVCache — use existing JANGTQ cache or implement sliding-window variant
public final class RotatingKVCache: AttentionCache {
    public let maxSize: Int
    public let keep: Int
    public var offset: Int = 0
    public init(maxSize: Int, keep: Int) { self.maxSize = maxSize; self.keep = keep }
    public func updateAndFetch(_ keys: MTLBuffer, _ values: MTLBuffer) -> (MTLBuffer, MTLBuffer) { fatalError("TODO") }
}

// SwitchGLU — should be implemented elsewhere in Swift runtime (JANGCoreMetal?)
public final class SwitchGLU {
    public init(inputDims: Int, hiddenDims: Int, numExperts: Int, activation: LimitedSwiGLU,
                weights: WeightBundle, prefix: String) {}
    public func forward(_ x: MTLBuffer, _ inds: MTLBuffer) -> MTLBuffer { fatalError("TODO") }
}

// WeightBundle — loader abstraction (reuse JANGTQLoader types)
public struct WeightBundle {
    public subscript(name: String) -> MTLBuffer { fatalError("TODO") }
    public func withPrefix(_ prefix: String) -> WeightBundle { fatalError("TODO") }
}

// Safe subscript helper
private extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
