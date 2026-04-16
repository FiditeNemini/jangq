/*
 * JANGTQ Model — top-level forward pass for an entire JANGTQ MoE model.
 * Created by Jinho Jang (eric@jangq.ai)
 *
 * Wires together:
 *   - Embedding lookup (CPU dequant of one row)
 *   - 62 decoder layers (attention + MoE block)
 *   - Final RMSNorm
 *   - LM head (8-bit affine GEMV)
 *
 * Per token:
 *   logits = self.forward(tokenId, position)  → MTLBuffer of (vocab,) fp32
 *
 * Greedy sampling lives in `JANGTQSampler` below.
 */

import Foundation
import Metal
import JANGCoreMetal

public final class JANGTQModel {
    public let bundle: JANGTQModelBundle
    public let context: MetalContext
    public let kernels: JANGTQKernels
    public let affine8: JANGTQAffine8Matmul
    public let ops: JANGTQDecodeOps
    public let cache: JANGTQKVCache

    public let embedTokens: JANGTQAffineWeight
    public let lmHead: JANGTQAffineWeight  // tied to embed if not separate
    public let finalNorm: MTLBuffer
    public let layers: [JANGTQDecoderLayer]
    public let config: ModelConfig

    public init(
        bundle: JANGTQModelBundle,
        context: MetalContext,
        maxSeqLen: Int = 2048,
        moePrefix: String = "block_sparse_moe"
    ) throws {
        self.bundle = bundle
        self.context = context
        self.kernels = try JANGTQKernels(context: context)
        self.affine8 = try JANGTQAffine8Matmul(context: context)
        self.ops = try JANGTQDecodeOps(context: context)
        self.config = bundle.config.model

        guard let embed = bundle.affineWeights["model.embed_tokens"] else {
            throw JANGError.tensorNotFound("model.embed_tokens")
        }
        self.embedTokens = embed

        // tied or separate lm_head
        if let head = bundle.affineWeights["lm_head"] {
            self.lmHead = head
        } else {
            self.lmHead = embed
        }

        guard let fn = bundle.halfTensors["model.norm.weight"] else {
            throw JANGError.tensorNotFound("model.norm.weight")
        }
        self.finalNorm = fn

        // KV cache
        self.cache = try JANGTQKVCache(
            device: context.device,
            nLayers: config.numHiddenLayers,
            kvHeads: config.kvHeads,
            headDim: config.headDim,
            maxSeqLen: maxSeqLen
        )

        // Build layers
        var built: [JANGTQDecoderLayer] = []
        for i in 0..<config.numHiddenLayers {
            let layer = try JANGTQDecoderLayer(
                layerIndex: i, config: config, bundle: bundle,
                context: context, kernels: kernels, affine8: affine8,
                ops: ops, moePrefix: moePrefix, cache: cache
            )
            built.append(layer)
        }
        self.layers = built
    }

    /// Forward pass for a single token. Returns logits as a fp32 MTLBuffer.
    /// `position` is the absolute token position (= cache.currentLength on entry).
    /// Each layer's attention writes to the SAME position slot in the cache
    /// (because all 62 layers process the same token). The cache currentLength
    /// counter is advanced exactly ONCE per call, after all layers finish.
    public func forward(tokenId: Int, position: Int) throws -> MTLBuffer {
        precondition(position == cache.currentLength,
            "position \(position) must equal cache.currentLength \(cache.currentLength)")

        // 1. Embedding lookup
        var x = try jangtqDequantizeEmbedRow(
            embed: embedTokens, tokenId: tokenId, device: context.device
        )

        // 2. Decoder layers — all 62 layers see the same position
        for layer in layers {
            x = try layer.forward(x: x, position: position)
        }

        // 3. Final RMSNorm (GPU)
        let xNorm = try ops.rmsnorm.run(
            x: x, gamma: finalNorm, dim: config.hiddenSize, eps: config.normEps
        )

        // 4. LM head: (hidden,) → (vocab,)
        let logits = try affine8.run(
            qweightBuf: lmHead.qweight, scalesBuf: lmHead.scales, biasesBuf: lmHead.biases,
            xBuf: xNorm,
            inFeatures: lmHead.inFeatures, outFeatures: lmHead.outFeatures,
            groupSize: lmHead.groupSize
        )

        // 5. Advance KV cache by exactly one token (after all layers wrote their slot)
        cache.advanceOneToken()
        return logits
    }

    public func reset() { cache.reset() }
}

// MARK: - Decoder layer wrapper

public final class JANGTQDecoderLayer {
    public let layerIndex: Int
    public let attention: JANGTQAttentionBlock
    public let postAttnNorm: MTLBuffer
    public let engine: JANGTQDecoderEngine
    public let ops: JANGTQDecodeOps
    public let config: ModelConfig
    public let isMoE: Bool

    public init(
        layerIndex: Int,
        config: ModelConfig,
        bundle: JANGTQModelBundle,
        context: MetalContext,
        kernels: JANGTQKernels,
        affine8: JANGTQAffine8Matmul,
        ops: JANGTQDecodeOps,
        moePrefix: String,
        cache: JANGTQKVCache
    ) throws {
        self.layerIndex = layerIndex
        self.config = config
        self.ops = ops
        self.attention = try JANGTQAttentionBlock(
            layerIndex: layerIndex, config: config, bundle: bundle,
            affine8: affine8, ops: ops
        )
        let postNormKey = "model.layers.\(layerIndex).post_attention_layernorm.weight"
        guard let pn = bundle.halfTensors[postNormKey] else {
            throw JANGError.tensorNotFound(postNormKey)
        }
        self.postAttnNorm = pn

        let firstDense = config.firstKDenseReplace ?? 0
        self.isMoE = layerIndex >= firstDense

        self.engine = JANGTQDecoderEngine(
            bundle: bundle, context: context, kernels: kernels, affine8: affine8,
            ops: ops, cache: cache, moePrefix: moePrefix
        )
    }

    /// `r = x + attn(input_norm(x)); out = r + moe(post_attn_norm(r))`
    /// All ops on GPU (attention internally runs the input_layernorm via ops.rmsnorm).
    public func forward(x: MTLBuffer, position: Int) throws -> MTLBuffer {
        // Attention sub-block (its own input_layernorm is inside, on GPU)
        let attnOut = try attention.forward(
            x: x, cache: engine.cache, position: position
        )
        let r = try ops.residual.run(a: x, b: attnOut, dim: config.hiddenSize)

        // post_attention_layernorm (GPU)
        let normed = try ops.rmsnorm.run(
            x: r, gamma: postAttnNorm, dim: config.hiddenSize, eps: config.normEps
        )

        guard isMoE else {
            throw JANGError.invalidFormat(
                "layer \(layerIndex) is dense (first_k_dense_replace=\(config.firstKDenseReplace ?? 0)) " +
                "but JANGTQ Swift dense path not implemented"
            )
        }

        // MoE block (router on CPU, MLP kernels on GPU, combine on CPU for now)
        let k = config.numExpertsPerTok ?? 8
        let moeOut = try engine.runMoE(
            layer: layerIndex, normedX: normed, hidden: config.hiddenSize, k: k
        )
        return try ops.residual.run(a: r, b: moeOut, dim: config.hiddenSize)
    }
}

// MARK: - Greedy sampler

public struct JANGTQSampler {
    public init() {}

    /// Argmax over fp32 logits buffer.
    public func argmax(logits: MTLBuffer, vocabSize: Int) -> Int {
        let p = logits.contents().bindMemory(to: Float.self, capacity: vocabSize)
        var bestI = 0
        var bestV = p[0]
        for i in 1..<vocabSize {
            if p[i] > bestV { bestV = p[i]; bestI = i }
        }
        return bestI
    }
}
