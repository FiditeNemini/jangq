// Projectors.swift
// mlp1 (vision) + sound_projection (audio) projectors for Nemotron omni.
//
// Mirrors `jang_tools/nemotron_omni/projectors.py` (validated working).
//
// Tensor naming on disk:
//   mlp1.0.weight                        LayerNorm   (5120,)
//   mlp1.1.weight                        Linear      (20480, 5120)
//   mlp1.3.weight                        Linear      (2688, 20480)
//   sound_projection.norm.weight         RMSNorm     (1024,)
//   sound_projection.linear1.weight      Linear      (4096, 1024)
//   sound_projection.linear2.weight      Linear      (2688, 4096)

import Foundation
import MLX
import MLXNN

/// Vision MLP projector (post-pixel-shuffle vit features → LLM hidden).
/// Forward: LayerNorm → Linear → GELU → Linear
@available(macOS 14.0, *)
public class VisionMLPProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear   // mlp1.1
    @ModuleInfo(key: "fc2") var fc2: Linear   // mlp1.3 (mlp1.2 is GELU, no params)

    public init(inDim: Int, projectorDim: Int, llmDim: Int) {
        self._layerNorm.wrappedValue = LayerNorm(dimensions: inDim)
        self._fc1.wrappedValue = Linear(inDim, projectorDim, bias: false)
        self._fc2.wrappedValue = Linear(projectorDim, llmDim, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = layerNorm(x)
        h = fc1(h)
        h = gelu(h)
        h = fc2(h)
        return h
    }
}

/// Sound projector (parakeet 1024 → LLM hidden 2688).
/// Forward: RMSNorm → Linear → SquaredReLU → Linear
@available(macOS 14.0, *)
public class SoundProjector: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    public init(
        soundHidden: Int = 1024,
        projectionHidden: Int = 4096,
        llmHidden: Int = 2688,
        eps: Float = 1e-5
    ) {
        self._norm.wrappedValue = RMSNorm(dimensions: soundHidden, eps: eps)
        self._linear1.wrappedValue = Linear(soundHidden, projectionHidden, bias: false)
        self._linear2.wrappedValue = Linear(projectionHidden, llmHidden, bias: false)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm(x)
        h = linear1(h)
        // SquaredReLU: ReLU(x)^2
        h = relu(h)
        h = h * h
        return linear2(h)
    }
}

/// Maps on-disk `mlp1.{0,1,3}.*` keys to nn.Module attribute names.
@available(macOS 14.0, *)
public func remapMlp1Weights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let rename: [String: String] = [
        "mlp1.0.weight": "layer_norm.weight",
        "mlp1.0.bias":   "layer_norm.bias",
        "mlp1.1.weight": "fc1.weight",
        "mlp1.1.bias":   "fc1.bias",
        "mlp1.3.weight": "fc2.weight",
        "mlp1.3.bias":   "fc2.bias",
    ]
    var out = [String: MLXArray]()
    for (oldKey, newKey) in rename {
        if let v = weights[oldKey] {
            out[newKey] = v
        }
    }
    return out
}

/// Maps on-disk `sound_projection.*` keys.
@available(macOS 14.0, *)
public func remapSoundProjectionWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let rename: [String: String] = [
        "sound_projection.norm.weight":     "norm.weight",
        "sound_projection.linear1.weight":  "linear1.weight",
        "sound_projection.linear1.bias":    "linear1.bias",
        "sound_projection.linear2.weight":  "linear2.weight",
        "sound_projection.linear2.bias":    "linear2.bias",
    ]
    var out = [String: MLXArray]()
    for (oldKey, newKey) in rename {
        if let v = weights[oldKey] {
            out[newKey] = v
        }
    }
    return out
}
