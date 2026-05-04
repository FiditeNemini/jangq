// Parakeet.swift
// Native Swift/MLX port of the parakeet Conformer encoder for Nemotron audio.
//
// 24-layer Conformer with 1024 hidden, 8 attn heads, 4096 FF intermediate,
// 9-tap depthwise conv, 8× subsampling. Mirrors
// `jang_tools/nemotron_omni/parakeet.py`.
//
// Tensor naming on disk (load via JangLoader extension):
//   sound_encoder.encoder.subsampling.layers.{0,2,5}.{weight,bias}     Conv2d 3×3
//   sound_encoder.encoder.subsampling.layers.{3,6}.{weight,bias}       Conv2d 1×1
//   sound_encoder.encoder.subsampling.linear.{weight,bias}             Linear 4096→1024
//   sound_encoder.encoder.layers.{0..23}.norm_feed_forward1.{w,b}     LN 1024
//   sound_encoder.encoder.layers.{0..23}.feed_forward1.linear1.weight  4096×1024
//   sound_encoder.encoder.layers.{0..23}.feed_forward1.linear2.weight  1024×4096
//   sound_encoder.encoder.layers.{0..23}.norm_self_att.{w,b}          LN 1024
//   sound_encoder.encoder.layers.{0..23}.self_attn.{q,k,v,o}_proj.weight  1024×1024
//   sound_encoder.encoder.layers.{0..23}.self_attn.relative_k_proj.weight 1024×1024
//   sound_encoder.encoder.layers.{0..23}.self_attn.bias_{u,v}          (8, 128)
//   sound_encoder.encoder.layers.{0..23}.norm_conv.{w,b}              LN 1024
//   sound_encoder.encoder.layers.{0..23}.conv.pointwise_conv1.weight   2048×1024×1
//   sound_encoder.encoder.layers.{0..23}.conv.depthwise_conv.weight    1024×1×9
//   sound_encoder.encoder.layers.{0..23}.conv.norm.{weight,bias,running_mean,running_var}  BN 1024
//   sound_encoder.encoder.layers.{0..23}.conv.pointwise_conv2.weight   1024×1024×1
//   sound_encoder.encoder.layers.{0..23}.norm_feed_forward2.{w,b}     LN 1024
//   sound_encoder.encoder.layers.{0..23}.feed_forward2.linear1.weight  4096×1024
//   sound_encoder.encoder.layers.{0..23}.feed_forward2.linear2.weight  1024×4096
//   sound_encoder.encoder.layers.{0..23}.norm_out.{w,b}               LN 1024
//
// Status: skeleton — the relative-positional attention is approximated by
// content-bias only (matches the simplified Python port, which gives close
// but not bit-exact outputs). Full rel-pos attention is queued.

import Foundation
import MLX
import MLXNN

@available(macOS 14.0, *)
public class ParakeetEncoder: Module {
    @ModuleInfo(key: "subsampling") var subsampling: ParakeetSubsampling
    @ModuleInfo(key: "layers") var layers: [ConformerBlock]

    public init(
        hiddenSize: Int = 1024,
        numLayers: Int = 24,
        numHeads: Int = 8,
        ffHidden: Int = 4096,
        convKernel: Int = 9
    ) {
        self._subsampling.wrappedValue = ParakeetSubsampling(hidden: hiddenSize)
        self._layers.wrappedValue = (0..<numLayers).map { _ in
            ConformerBlock(
                dim: hiddenSize, numHeads: numHeads,
                ffHidden: ffHidden, convKernel: convKernel,
            )
        }
    }

    public func callAsFunction(_ mel: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var x = subsampling(mel)
        for layer in layers {
            x = layer(x, mask: mask)
        }
        return x
    }
}

@available(macOS 14.0, *)
public class ParakeetSubsampling: Module, UnaryLayer {
    // Conv2d layers. MLX Swift Conv2d expects channels-last input.
    @ModuleInfo(key: "layers_0") var layer0: Conv2d  // 1→256 k=3 s=2
    @ModuleInfo(key: "layers_2") var layer2: Conv2d  // 256→256 k=3 s=1
    @ModuleInfo(key: "layers_3") var layer3: Conv2d  // 256→256 k=1
    @ModuleInfo(key: "layers_5") var layer5: Conv2d  // 256→256 k=3 s=2
    @ModuleInfo(key: "layers_6") var layer6: Conv2d  // 256→256 k=1
    @ModuleInfo(key: "layers_8") var layer8: Conv2d  // 256→256 k=3 s=2
    @ModuleInfo(key: "linear") var linear: Linear   // 4096→1024

    public init(hidden: Int = 1024) {
        self._layer0.wrappedValue = Conv2d(inputChannels: 1, outputChannels: 256,
                                            kernelSize: .init(3), stride: .init(2), padding: .init(1))
        self._layer2.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256,
                                            kernelSize: .init(3), stride: .init(1), padding: .init(1))
        self._layer3.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256,
                                            kernelSize: .init(1), stride: .init(1), padding: .init(0))
        self._layer5.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256,
                                            kernelSize: .init(3), stride: .init(2), padding: .init(1))
        self._layer6.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256,
                                            kernelSize: .init(1), stride: .init(1), padding: .init(0))
        self._layer8.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256,
                                            kernelSize: .init(3), stride: .init(2), padding: .init(1))
        self._linear.wrappedValue = Linear(4096, hidden)
    }

    public func callAsFunction(_ mel: MLXArray) -> MLXArray {
        // mel: (B, T, n_mels=128) → reshape to (B, n_mels, T, 1) for Conv2d
        let (b, t, m) = (mel.shape[0], mel.shape[1], mel.shape[2])
        var x = mel.transposed(0, 2, 1).expandedDimensions(axis: -1)  // (B, 128, T, 1)
        x = relu(layer0(x))
        x = layer2(x)
        x = relu(layer3(x))
        x = layer5(x)
        x = relu(layer6(x))
        x = layer8(x)
        // Now x is (B, 16, T_sub, 256). Permute to (B, T_sub, 16, 256), flatten last two.
        let (b2, mSub, tSub, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        x = x.transposed(0, 2, 1, 3).reshaped([b2, tSub, mSub * c])
        return linear(x)
    }
}

@available(macOS 14.0, *)
public class ConformerBlock: Module {
    @ModuleInfo(key: "norm_feed_forward1") var nFF1: LayerNorm
    @ModuleInfo(key: "feed_forward1") var ff1: ConformerFeedForward
    @ModuleInfo(key: "norm_self_att") var nAttn: LayerNorm
    @ModuleInfo(key: "self_attn") var attn: RelativeMultiHeadAttention
    @ModuleInfo(key: "norm_conv") var nConv: LayerNorm
    @ModuleInfo(key: "conv") var conv: ConformerConvModule
    @ModuleInfo(key: "norm_feed_forward2") var nFF2: LayerNorm
    @ModuleInfo(key: "feed_forward2") var ff2: ConformerFeedForward
    @ModuleInfo(key: "norm_out") var nOut: LayerNorm

    public init(dim: Int, numHeads: Int, ffHidden: Int, convKernel: Int) {
        self._nFF1.wrappedValue = LayerNorm(dimensions: dim)
        self._ff1.wrappedValue = ConformerFeedForward(dim: dim, hidden: ffHidden)
        self._nAttn.wrappedValue = LayerNorm(dimensions: dim)
        self._attn.wrappedValue = RelativeMultiHeadAttention(dim: dim, numHeads: numHeads)
        self._nConv.wrappedValue = LayerNorm(dimensions: dim)
        self._conv.wrappedValue = ConformerConvModule(dim: dim, kernelSize: convKernel)
        self._nFF2.wrappedValue = LayerNorm(dimensions: dim)
        self._ff2.wrappedValue = ConformerFeedForward(dim: dim, hidden: ffHidden)
        self._nOut.wrappedValue = LayerNorm(dimensions: dim)
    }

    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x + 0.5 * ff1(nFF1(x))
        h = h + attn(nAttn(h), mask: mask)
        h = h + conv(nConv(h))
        h = h + 0.5 * ff2(nFF2(h))
        return nOut(h)
    }
}

@available(macOS 14.0, *)
public class ConformerFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    public init(dim: Int, hidden: Int) {
        self._linear1.wrappedValue = Linear(dim, hidden, bias: false)
        self._linear2.wrappedValue = Linear(hidden, dim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return linear2(silu(linear1(x)))
    }
}

/// Build sinusoidal relative-position embeddings, mirroring
/// `transformers.ParakeetEncoderRelPositionalEncoding`.
/// Returns: (1, 2*seqLen-1, hiddenSize)
@available(macOS 14.0, *)
public func buildRelPosEmbeddings(seqLen: Int, hiddenSize: Int, base: Float = 10000) -> MLXArray {
    let half = hiddenSize / 2
    let invFreqExp = MLXArray(stride(from: 0, to: hiddenSize, by: 2).map { Float($0) }) / Float(hiddenSize)
    let invFreq = MLX.pow(MLXArray(base), invFreqExp).recip()
    // position_ids: [seqLen-1, seqLen-2, ..., 0, -1, ..., -(seqLen-1)]
    let positionIds = MLXArray(stride(from: seqLen - 1, through: -seqLen + 1, by: -1).map { Float($0) })
    // freqs: (2*seqLen-1, half)
    let freqs = positionIds.expandedDimensions(axis: -1) * invFreq.expandedDimensions(axis: 0)
    let sin = MLX.sin(freqs)
    let cos = MLX.cos(freqs)
    // Interleave sin and cos along the last dim → (2*seqLen-1, hiddenSize)
    let stacked = MLX.stacked([sin, cos], axis: -1)
        .reshaped([2 * seqLen - 1, hiddenSize])
    return stacked.expandedDimensions(axis: 0)
}

/// Transformer-XL relative-position shift ("skewing" trick).
/// Input: (B, H, T, 2T-1) scores → output: (B, H, T, T)
@available(macOS 14.0, *)
public func relShift(_ scores: MLXArray, seqLen: Int) -> MLXArray {
    let B = scores.shape[0], H = scores.shape[1], T = scores.shape[2]
    // Pad with one zero column on the left → (B, H, T, 2T)
    let zeros = MLXArray.zeros([B, H, T, 1], dtype: scores.dtype)
    var padded = MLX.concatenated([zeros, scores], axis: -1)
    // Reshape and slice
    padded = padded.reshaped([B, H, 2 * T, T])
    padded = padded[0..., 0..., 1..., 0...]                // (B, H, 2T-1, T)
    padded = padded.reshaped([B, H, T, 2 * T - 1])
    return padded[0..., 0..., 0..., 0..<seqLen]
}

@available(macOS 14.0, *)
public class RelativeMultiHeadAttention: Module {
    let dim: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    @ModuleInfo(key: "relative_k_proj") var rk: Linear
    @ParameterInfo(key: "bias_u") var biasU: MLXArray
    @ParameterInfo(key: "bias_v") var biasV: MLXArray

    public init(dim: Int, numHeads: Int) {
        self.dim = dim
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._q.wrappedValue = Linear(dim, dim, bias: false)
        self._k.wrappedValue = Linear(dim, dim, bias: false)
        self._v.wrappedValue = Linear(dim, dim, bias: false)
        self._o.wrappedValue = Linear(dim, dim, bias: false)
        self._rk.wrappedValue = Linear(dim, dim, bias: false)
        self._biasU.wrappedValue = MLXArray.zeros([numHeads, headDim])
        self._biasV.wrappedValue = MLXArray.zeros([numHeads, headDim])
    }

    /// Full Transformer-XL relative-position attention.
    /// Validated against the Python port (parakeet.py `RelativeMultiHeadAttention`)
    /// which produces near-bit-exact transcription parity with PyTorch's
    /// `transformers.ParakeetEncoderAttention`.
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (b, t, _) = (x.shape[0], x.shape[1], x.shape[2])
        // Projections (B, H, T, Hd)
        let qO = q(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        let kO = k(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)
        let vO = v(x).reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3)

        // Sinusoidal rel-pos: (1, 2T-1, D) → relative_k_proj → (1, 2T-1, H, Hd)
        let posEmb = buildRelPosEmbeddings(seqLen: t, hiddenSize: dim)
        let relK = rk(posEmb).reshaped([1, 2 * t - 1, numHeads, headDim])

        // Term (b)+(d): (Q + bias_v) · R^T
        let qWithV = qO + biasV.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        let relKT = relK.transposed(0, 2, 3, 1)             // (1, H, Hd, 2T-1)
        var matrixBD = MLX.matmul(qWithV, relKT)             // (B, H, T, 2T-1)
        matrixBD = relShift(matrixBD, seqLen: t)
        matrixBD = matrixBD * scale

        if let m = mask {
            matrixBD = matrixBD + m
        }

        // Term (a)+(c): (Q + bias_u) · K^T (handled inside SDPA)
        let qWithU = qO + biasU.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
        var out = MLXFast.scaledDotProductAttention(
            queries: qWithU, keys: kO, values: vO, scale: scale, mask: matrixBD,
        )
        out = out.transposed(0, 2, 1, 3).reshaped([b, t, dim])
        return o(out)
    }
}

@available(macOS 14.0, *)
public class ConformerConvModule: Module, UnaryLayer {
    let dim: Int
    let kernelSize: Int

    @ModuleInfo(key: "pointwise_conv1") var pw1: Conv1d
    @ParameterInfo(key: "depthwise_conv_weight") var dwWeight: MLXArray
    @ModuleInfo(key: "norm") var norm: BatchNorm1d
    @ModuleInfo(key: "pointwise_conv2") var pw2: Conv1d

    public init(dim: Int, kernelSize: Int) {
        self.dim = dim
        self.kernelSize = kernelSize
        self._pw1.wrappedValue = Conv1d(inputChannels: dim, outputChannels: 2 * dim,
                                         kernelSize: 1, bias: false)
        self._dwWeight.wrappedValue = MLXArray.zeros([dim, 1, kernelSize])
        self._norm.wrappedValue = BatchNorm1d(dim: dim)
        self._pw2.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim,
                                         kernelSize: 1, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: (B, T, D) — Conv1d in MLX-Swift uses (B, T, C_in)
        var h = pw1(x)
        let split = MLX.split(h, indices: [dim], axis: -1)
        let (a, gate) = (split[0], split[1])
        h = a * sigmoid(gate)
        h = depthwise(h)
        h = norm(h)
        h = silu(h)
        return pw2(h)
    }

    private func depthwise(_ x: MLXArray) -> MLXArray {
        // Manual per-channel 1-D conv with kernel=9 and symmetric padding.
        let (b, t, d) = (x.shape[0], x.shape[1], x.shape[2])
        let pad = (kernelSize - 1) / 2
        let padded = MLX.padded(x, widths: [(0,0), (pad,pad), (0,0)])
        let w = dwWeight.reshaped([d, kernelSize])  // (D, K)
        var out = MLXArray.zeros([b, t, d])
        for i in 0..<kernelSize {
            out = out + padded[0..., i..<(i + t), 0...] * w[0..., i].reshaped([1, 1, d])
        }
        return out
    }
}

/// Inference-only BatchNorm1d using stored running stats.
@available(macOS 14.0, *)
public class BatchNorm1d: Module, UnaryLayer {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray  // γ
    @ParameterInfo(key: "bias") var bias: MLXArray      // β
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    public init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dim])
        self._bias.wrappedValue = MLXArray.zeros([dim])
        self._runningMean.wrappedValue = MLXArray.zeros([dim])
        self._runningVar.wrappedValue = MLXArray.ones([dim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return (x - runningMean) / MLX.sqrt(runningVar + eps) * weight + bias
    }
}
