import Foundation

struct Hy3RuntimeContract: Codable {
    struct Attention: Codable {
        let type: String
        let cacheTopology: String
        let qHeads: Int
        let kvHeads: Int
        let headDim: Int
        let qkNorm: Bool
        let ropeTheta: Double
        let maxPositionEmbeddings: Int
    }

    struct MoE: Codable {
        let type: String
        let routedExperts: Int
        let topK: Int
        let sharedExperts: Int
        let firstDenseLayers: Int
        let routerScalingFactor: Double
        let routeNorm: Bool
        let rule: String
    }

    let modelType: String
    let architecture: String
    let modality: String
    let layers: Int
    let mtpLayers: Int
    let attention: Attention
    let moe: MoE
    let cacheRule: String
    let quantizationRule: String
    let firstJANGTQProfile: String
    let memoryRule: String
}

let contract = Hy3RuntimeContract(
    modelType: "hy_v3",
    architecture: "HYV3ForCausalLM",
    modality: "text",
    layers: 80,
    mtpLayers: 1,
    attention: .init(
        type: "dense_gqa",
        cacheTopology: "standard_kv",
        qHeads: 64,
        kvHeads: 8,
        headDim: 128,
        qkNorm: true,
        ropeTheta: 11_158_840,
        maxPositionEmbeddings: 262_144
    ),
    moe: .init(
        type: "sigmoid_bias_topk_with_shared_expert",
        routedExperts: 192,
        topK: 8,
        sharedExperts: 1,
        firstDenseLayers: 1,
        routerScalingFactor: 2.826,
        routeNorm: true,
        rule: "sigmoid router logits; add expert correction bias for top-k choice; normalize selected sigmoid weights; apply router scaling; add always-active shared expert"
    ),
    cacheRule: "standard causal KV cache; no VL media salt; MTP cache must be isolated from normal decode unless speculative path is implemented",
    quantizationRule: "First 128 GB target is JANGTQ2: routed gate/up/down 2-bit, attention/shared/dense/MTP 8-bit, router bias and norms passthrough",
    firstJANGTQProfile: "JANGTQ2",
    memoryRule: "JANGTQ2 is the 128 GB release candidate; JANGTQ_K is quality-first and likely tight on 128 GB unless measured load proof says otherwise"
)

let data = try JSONEncoder().encode(contract)
print(String(data: data, encoding: .utf8)!)
