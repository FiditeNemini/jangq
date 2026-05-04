// RADIOVision.swift
// Native Swift/MLX port of NVIDIA's RADIO ViT (radio_v2.5-h variant)
// for the Nemotron-3-Nano-Omni-30B-A3B vision tower.
//
// Mirrors `jang_tools/nemotron_omni/radio.py` (validated working, matches
// PyTorch within bf16 noise on a 512×512 input).
//
// Architecture:
//   • InputConditioner (optional): (x - mean) / std broadcast over (3, 1, 1)
//   • ViTPatchGenerator:
//       - Im2Patches:   (B, 3, H, W) → (B, num_patches, 3*P*P)
//       - embedder:     Linear(3*P*P → 1280)
//       - pos_embed:    bilinear interp from stored (1, 16384, 1280) max grid
//       - cls_token:    concat 10 cls/register tokens at front
//   • 32 × ViTBlock (pre-norm, standard timm):
//       LayerNorm → Attention(qkv, proj) → residual
//       LayerNorm → MLP(fc1, GELU, fc2)  → residual
//   • NO final norm (timm sets model.norm = nn.Identity)
//
// Tensor naming on disk (load via JangLoader extension):
//   vision_model.radio_model.input_conditioner.norm_{mean,std}        (3,1,1)
//   vision_model.radio_model.model.patch_generator.cls_token.token    (10, 1280)
//   vision_model.radio_model.model.patch_generator.embedder.weight    (1280, 768)
//   vision_model.radio_model.model.patch_generator.pos_embed          (1, 16384, 1280)
//   vision_model.radio_model.model.patch_generator.video_embedder.weight (1280, 1536)
//   vision_model.radio_model.model.blocks.{0..31}.{norm1,norm2}.{weight,bias}  (1280,)
//   vision_model.radio_model.model.blocks.{0..31}.attn.qkv.{weight,bias}        (3840, 1280) (3840,)
//   vision_model.radio_model.model.blocks.{0..31}.attn.proj.{weight,bias}       (1280, 1280) (1280,)
//   vision_model.radio_model.model.blocks.{0..31}.mlp.fc1.{weight,bias}         (5120, 1280) (5120,)
//   vision_model.radio_model.model.blocks.{0..31}.mlp.fc2.{weight,bias}         (1280, 5120) (1280,)
//
// Status: skeleton — fill in `bilinearResize2D` (Swift port of MLX
// `_bilinear_resize_2d`) and the patch-generator forward.

import Foundation
import MLX
import MLXNN

@available(macOS 14.0, *)
public class RADIOVisionModel: Module, UnaryLayer {
    let embedDim: Int
    let patchSize: Int
    let numClsTokens: Int

    @ModuleInfo(key: "patch_generator") var patchGenerator: ViTPatchGenerator
    @ModuleInfo(key: "blocks") var blocks: [ViTBlock]

    public init(
        embedDim: Int = 1280,
        numBlocks: Int = 32,
        numHeads: Int = 16,
        mlpRatio: Float = 4.0,
        patchSize: Int = 16,
        numClsTokens: Int = 10,
        maxGrid: Int = 128,
        videoTemporalPatch: Int = 2
    ) {
        self.embedDim = embedDim
        self.patchSize = patchSize
        self.numClsTokens = numClsTokens

        self._patchGenerator.wrappedValue = ViTPatchGenerator(
            patchSize: patchSize,
            embedDim: embedDim,
            numClsTokens: numClsTokens,
            maxGrid: maxGrid,
            videoTemporalPatch: videoTemporalPatch
        )
        self._blocks.wrappedValue = (0..<numBlocks).map { _ in
            ViTBlock(dim: embedDim, numHeads: numHeads, mlpRatio: mlpRatio)
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = patchGenerator(x, video: false)
        for block in blocks {
            h = block(h)
        }
        return h
    }
}

@available(macOS 14.0, *)
public class ViTPatchGenerator: Module, UnaryLayer {
    let patchSize: Int
    let embedDim: Int
    let numClsTokens: Int
    let maxGrid: Int

    @ModuleInfo(key: "embedder") var embedder: Linear
    @ModuleInfo(key: "video_embedder") var videoEmbedder: Linear
    @ParameterInfo(key: "cls_token") var clsToken: MLXArray
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray

    public init(
        patchSize: Int, embedDim: Int, numClsTokens: Int,
        maxGrid: Int, videoTemporalPatch: Int
    ) {
        self.patchSize = patchSize
        self.embedDim = embedDim
        self.numClsTokens = numClsTokens
        self.maxGrid = maxGrid
        self._embedder.wrappedValue = Linear(3 * patchSize * patchSize, embedDim, bias: false)
        self._videoEmbedder.wrappedValue = Linear(
            videoTemporalPatch * 3 * patchSize * patchSize, embedDim, bias: false,
        )
        self._clsToken.wrappedValue = MLXArray.zeros([numClsTokens, embedDim])
        self._posEmbed.wrappedValue = MLXArray.zeros([1, maxGrid * maxGrid, embedDim])
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        callAsFunction(x, video: false)
    }

    public func callAsFunction(_ x: MLXArray, video: Bool) -> MLXArray {
        // x: (B, 3, H, W) → patches → embed → +pos → +cls
        let (b, _, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let p = patchSize
        let py = h / p, px = w / p

        // Im2Patches: (B, 3, H, W) → (B, py*px, 3*p*p)
        // Reshape: (B, 3, py, p, px, p) → transpose (0,2,4,1,3,5) → (B, py, px, 3, p, p) → flatten
        var patches = x.reshaped([b, 3, py, p, px, p])
        patches = patches.transposed(0, 2, 4, 1, 3, 5)
        patches = patches.reshaped([b, py * px, 3 * p * p])

        // Embedder
        patches = video ? videoEmbedder(patches) : embedder(patches)

        // Position encoding (bilinear-interpolated from maxGrid → (py, px))
        let pos = getPosEmbed(inputH: h, inputW: w)
        patches = patches + pos

        // Concat cls tokens at front
        let cls = clsToken.expandedDimensions(axis: 0)
            .broadcasted(to: [b, numClsTokens, embedDim])
        return MLX.concatenated([cls, patches], axis: 1)
    }

    private func getPosEmbed(inputH: Int, inputW: Int) -> MLXArray {
        let gy = inputH / patchSize
        let gx = inputW / patchSize
        if gy == maxGrid && gx == maxGrid {
            return posEmbed
        }
        // pos_embed (1, max*max, D) → (1, max, max, D) → (1, D, max, max)
        var pe = posEmbed.reshaped([1, maxGrid, maxGrid, embedDim])
        pe = pe.transposed(0, 3, 1, 2)
        // Interpolate to max(gy, gx) square (eval-time CPE)
        let maxDim = max(gy, gx)
        pe = bilinearResize2D(pe, targetH: maxDim, targetW: maxDim)
        // Window-select to (gy, gx)
        pe = pe[0..., 0..., 0..<gy, 0..<gx]
        // Flatten back to (1, gy*gx, D)
        pe = pe.transposed(0, 2, 3, 1).reshaped([1, gy * gx, embedDim])
        return pe
    }

    /// Bilinear resize (1, C, H, W) → (1, C, targetH, targetW).
    /// align_corners=False (Megatron/vLLM convention, matches PyTorch eval).
    private func bilinearResize2D(_ x: MLXArray, targetH: Int, targetW: Int) -> MLXArray {
        // TODO: port the Python `_bilinear_resize_2d` math:
        //   src_y = (i + 0.5) * H / target - 0.5
        //   floor + ceil + clamp + linear blend across both axes
        // For now, use MLXFast or stub.
        fatalError("TODO: bilinear resize")
    }
}

@available(macOS 14.0, *)
public class ViTBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: ViTAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: ViTMLP

    public init(dim: Int, numHeads: Int, mlpRatio: Float) {
        self._norm1.wrappedValue = LayerNorm(dimensions: dim)
        self._attn.wrappedValue = ViTAttention(dim: dim, numHeads: numHeads)
        self._norm2.wrappedValue = LayerNorm(dimensions: dim)
        self._mlp.wrappedValue = ViTMLP(dim: dim, hiddenDim: Int(Float(dim) * mlpRatio))
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attn(norm1(x))
        h = h + mlp(norm2(h))
        return h
    }
}

@available(macOS 14.0, *)
public class ViTAttention: Module, UnaryLayer {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    public init(dim: Int, numHeads: Int, qkvBias: Bool = true) {
        self.numHeads = numHeads
        self.headDim = dim / numHeads
        self.scale = pow(Float(headDim), -0.5)
        self._qkv.wrappedValue = Linear(dim, 3 * dim, bias: qkvBias)
        self._proj.wrappedValue = Linear(dim, dim, bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n, c) = (x.shape[0], x.shape[1], x.shape[2])
        let qkvOut = qkv(x).reshaped([b, n, 3, numHeads, headDim])
            .transposed(2, 0, 3, 1, 4)
        let q = qkvOut[0], k = qkvOut[1], v = qkvOut[2]
        var out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil,
        )
        out = out.transposed(0, 2, 1, 3).reshaped([b, n, c])
        return proj(out)
    }
}

@available(macOS 14.0, *)
public class ViTMLP: Module, UnaryLayer {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    public init(dim: Int, hiddenDim: Int) {
        self._fc1.wrappedValue = Linear(dim, hiddenDim)
        self._fc2.wrappedValue = Linear(hiddenDim, dim)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return fc2(gelu(fc1(x)))
    }
}

/// Pixel-shuffle helper (ps_version='v2'): (B, H, W, C) → (B, H*r, W*r, C/r²)
@available(macOS 14.0, *)
public func pixelShuffle(_ x: MLXArray, scaleFactor: Float) -> MLXArray {
    let (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    let s = scaleFactor
    var out = x.reshaped([b, w, Int(Float(h) * s), Int(Float(c) / s)])
    out = out.transposed(0, 2, 1, 3)
    out = out.reshaped([b, Int(Float(h) * s), Int(Float(w) * s), Int(Float(c) / (s * s))])
    out = out.transposed(0, 2, 1, 3)
    return out
}
