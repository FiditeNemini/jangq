import Foundation

// Hy3 runtime skeleton for future vmlx-swift-lm work.
// This file is standalone documentation code; it does not load MLX weights.

enum Hy3MTPMode: String, Codable {
    case none
    case preservedDisabled = "preserved_disabled"
    case enabled
}

struct Hy3AttentionSpec: Codable {
    let hiddenSize: Int
    let numLayers: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let headDim: Int
    let qkNorm: Bool
    let ropeTheta: Double
    let maxPositionEmbeddings: Int

    var numKeyValueGroups: Int {
        numAttentionHeads / numKeyValueHeads
    }

    var kvBytesPerTokenFP16: Int {
        numLayers * 2 * numKeyValueHeads * headDim * 2
    }
}

struct Hy3MoESpec: Codable {
    let numExperts: Int
    let topK: Int
    let numSharedExperts: Int
    let expertHiddenDim: Int
    let firstDenseLayers: Int
    let routeNorm: Bool
    let routerScalingFactor: Double
    let useSigmoid: Bool
    let useExpertBias: Bool
}

struct Hy3RuntimeSpec: Codable {
    let modelType: String
    let attention: Hy3AttentionSpec
    let moe: Hy3MoESpec
    let mtpLayers: Int
    let mtpMode: Hy3MTPMode
    let parser: String
    let toolParser: String
}

struct Hy3ImplementationChecklist: Codable {
    let attention: [String]
    let moe: [String]
    let quantization: [String]
    let parser: [String]
    let mtp: [String]
}

let spec = Hy3RuntimeSpec(
    modelType: "hy_v3",
    attention: Hy3AttentionSpec(
        hiddenSize: 4096,
        numLayers: 80,
        numAttentionHeads: 64,
        numKeyValueHeads: 8,
        headDim: 128,
        qkNorm: true,
        ropeTheta: 11_158_840,
        maxPositionEmbeddings: 262_144
    ),
    moe: Hy3MoESpec(
        numExperts: 192,
        topK: 8,
        numSharedExperts: 1,
        expertHiddenDim: 1536,
        firstDenseLayers: 1,
        routeNorm: true,
        routerScalingFactor: 2.826,
        useSigmoid: true,
        useExpertBias: true
    ),
    mtpLayers: 1,
    mtpMode: .preservedDisabled,
    parser: "qwen3-compatible <think> tags plus reasoning_effort",
    toolParser: "hunyuan/tencent XML-like tool tags"
)

let checklist = Hy3ImplementationChecklist(
    attention: [
        "Use standard causal KV cache.",
        "Apply q_norm/k_norm per head before default RoPE.",
        "Use 64 Q heads, 8 KV heads, head_dim 128.",
        "Do not use MLA, SSM, CCA, sliding-window, or media cache salts."
    ],
    moe: [
        "Layer 0 is dense FFN.",
        "Layers 1...79 are sparse MoE.",
        "Router uses sigmoid probabilities.",
        "Expert bias affects top-k choice.",
        "Selected sigmoid weights are normalized when route_norm is true.",
        "Multiply selected weights by router_scaling_factor.",
        "Add always-active shared expert output."
    ],
    quantization: [
        "Routed expert gate/up/down use JANGTQ2 TurboQuant kernels.",
        "Attention/shared/dense/embed/lm_head/MTP matmuls use affine 8-bit first bundle.",
        "Norms/router/expert_bias stay fp16 passthrough."
    ],
    parser: [
        "Support reasoning_effort values no_think, low, high.",
        "Do not leak closed <think></think> into visible content.",
        "Parse <tool_calls>/<tool_call>/<tool_sep>/<arg_key>/<arg_value> tags."
    ],
    mtp: [
        "First runtime mode is preserved_disabled.",
        "When enabled later, keep draft state separate from accepted KV.",
        "Only commit a drafted token after target-model verification."
    ]
)

let output: [String: any Encodable] = [
    "spec": spec,
    "checklist": checklist,
    "kvGBAt4K": Double(spec.attention.kvBytesPerTokenFP16 * 4096) / 1_000_000_000.0
]

struct AnyEncodable: Encodable {
    let value: any Encodable
    func encode(to encoder: Encoder) throws {
        try value.encode(to: encoder)
    }
}

let erased = output.mapValues { AnyEncodable(value: $0) }
let data = try JSONEncoder().encode(erased)
print(String(data: data, encoding: .utf8)!)

