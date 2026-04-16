/*
 * JANG Model Configuration
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Parses config.json and jang_config.json to determine model
 * architecture, dimensions, and quantization parameters.
 */

import Foundation

/// HuggingFace model configuration (from config.json).
///
/// Two layouts are handled transparently:
///
///   1. Top-level (MiniMax M2.7, GLM 5.1 etc.) — every field lives at the
///      root of config.json.
///   2. Nested under `text_config` (Qwen3-Next, Qwen3.5, Qwen3.6, VL models) —
///      the outer config wraps a text-only sub-config plus vision_config.
///
/// `init(from:)` reads from `text_config` first when present, falls back
/// to root-level, so downstream code can stay oblivious.
public struct ModelConfig: Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int?
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int?
    public let ropeTheta: Double?
    public let rmsNormEps: Double?
    public let modelType: String?
    public let architectures: [String]?
    public let tieWordEmbeddings: Bool?

    // MoE fields (present only for MoE architectures).
    // MiniMax/GLM use `num_local_experts`; Qwen3-Next/3.5/3.6 use `num_experts`.
    public let numLocalExperts: Int?
    public let numExpertsPerTok: Int?
    public let moeIntermediateSize: Int?
    public let firstKDenseReplace: Int?
    public let kvLoraRank: Int?
    public let qLoraRank: Int?

    // Qwen3-Next / Qwen3.5 / Qwen3.6 hybrid attention + shared expert fields.
    public let sharedExpertIntermediateSize: Int?
    public let fullAttentionInterval: Int?
    public let layerTypes: [String]?
    public let linearNumValueHeads: Int?
    public let linearNumKeyHeads: Int?
    public let linearValueHeadDim: Int?
    public let linearKeyHeadDim: Int?
    public let linearConvKernelDim: Int?
    public let attnOutputGate: Bool?
    public let partialRotaryFactor: Double?
    public let normTopkProb: Bool?
    public let headDimOverride: Int?   // `head_dim` explicit (Qwen3.6=256)

    /// Whether the model is a Mixture-of-Experts.
    public var isMoE: Bool { (numLocalExperts ?? 0) > 0 && (numExpertsPerTok ?? 0) > 0 }

    /// Whether the model uses Multi-head Latent Attention (DeepSeek/GLM family).
    public var isMLA: Bool { (kvLoraRank ?? 0) > 0 }

    /// Hybrid linear + full attention (Qwen3-Next/3.5/3.6).
    public var isHybridAttention: Bool {
        (layerTypes?.contains("linear_attention") ?? false)
            || (fullAttentionInterval ?? 0) > 0
    }

    /// Has a DeepSeek-style always-active shared expert alongside routed MoE.
    public var hasSharedExpert: Bool { (sharedExpertIntermediateSize ?? 0) > 0 }

    /// Attention output has a per-head sigmoid gate (Qwen3-Next q_proj outputs 2×).
    public var hasAttnOutputGate: Bool { attnOutputGate ?? false }

    /// True iff routing is softmax+topk (Qwen-family) vs sigmoid+e_score_bias (MiniMax/GLM).
    public var isSoftmaxRouter: Bool {
        // Heuristic: Qwen3-Next/3.5/3.6 carry `norm_topk_prob` (True/False).
        // MiniMax/GLM don't set this field; they use e_score_correction_bias.
        normTopkProb != nil
    }

    /// Number of KV heads (defaults to num_attention_heads if not specified = MHA).
    public var kvHeads: Int { numKeyValueHeads ?? numAttentionHeads }

    /// Head dimension — respects explicit `head_dim` config field (Qwen3.6=256).
    public var headDim: Int {
        if let hd = headDimOverride, hd > 0 { return hd }
        return hiddenSize / numAttentionHeads
    }

    /// RoPE base frequency.
    public var ropeBase: Float { Float(ropeTheta ?? 10000.0) }

    /// RMSNorm epsilon.
    public var normEps: Float { Float(rmsNormEps ?? 1e-5) }

    /// Whether embeddings and lm_head share weights.
    public var tiedEmbeddings: Bool { tieWordEmbeddings ?? false }

    // MARK: - Decoding

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case modelType = "model_type"
        case architectures
        case tieWordEmbeddings = "tie_word_embeddings"
        case numLocalExperts = "num_local_experts"
        case numExperts = "num_experts"    // Qwen3-Next synonym
        case numExpertsPerTok = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case fullAttentionInterval = "full_attention_interval"
        case layerTypes = "layer_types"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case attnOutputGate = "attn_output_gate"
        case partialRotaryFactor = "partial_rotary_factor"
        case normTopkProb = "norm_topk_prob"
        case headDim = "head_dim"
    }
}

extension ModelConfig: Decodable {
    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: CodingKeys.self)
        // If text_config is present, that's our primary source (Qwen3.5/3.6 etc).
        let nested: KeyedDecodingContainer<CodingKeys>? =
            try? root.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig)

        /// Pulls from `text_config` first, else from root.
        func get<T: Decodable>(_ key: CodingKeys) -> T? {
            if let n = nested, let v = try? n.decodeIfPresent(T.self, forKey: key) {
                return v
            }
            return try? root.decodeIfPresent(T.self, forKey: key)
        }
        func getReq<T: Decodable>(_ key: CodingKeys, default d: T) -> T {
            get(key) ?? d
        }

        // Required dims — if neither layout has them, fall back to a safe default
        // (0/empty) so decode doesn't crash; downstream guards will catch invalid configs.
        self.hiddenSize = getReq(.hiddenSize, default: 0)
        self.intermediateSize = getReq(.intermediateSize, default: 0)
        self.numHiddenLayers = getReq(.numHiddenLayers, default: 0)
        self.numAttentionHeads = getReq(.numAttentionHeads, default: 0)
        self.numKeyValueHeads = get(.numKeyValueHeads)
        self.vocabSize = getReq(.vocabSize, default: 0)
        self.maxPositionEmbeddings = get(.maxPositionEmbeddings)
        self.ropeTheta = get(.ropeTheta)
        self.rmsNormEps = get(.rmsNormEps)
        // Architecture identity is a ROOT-level concept (VLM wrapper names the
        // family; text_config names the sub-module, e.g., qwen3_5_moe_text).
        self.modelType = (try? root.decodeIfPresent(String.self, forKey: .modelType)) ?? get(.modelType)
        self.architectures = try? root.decodeIfPresent([String].self, forKey: .architectures)
        self.tieWordEmbeddings = (try? root.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings))
            ?? get(.tieWordEmbeddings)

        // MoE: num_local_experts (MiniMax/GLM) OR num_experts (Qwen3-Next/3.5/3.6)
        let nLocal: Int? = get(.numLocalExperts)
        let nGeneric: Int? = get(.numExperts)
        self.numLocalExperts = nLocal ?? nGeneric
        self.numExpertsPerTok = get(.numExpertsPerTok)
        self.moeIntermediateSize = get(.moeIntermediateSize)
        self.firstKDenseReplace = get(.firstKDenseReplace)
        self.kvLoraRank = get(.kvLoraRank)
        self.qLoraRank = get(.qLoraRank)

        // Qwen3-Next family extras
        self.sharedExpertIntermediateSize = get(.sharedExpertIntermediateSize)
        self.fullAttentionInterval = get(.fullAttentionInterval)
        self.layerTypes = get(.layerTypes)
        self.linearNumValueHeads = get(.linearNumValueHeads)
        self.linearNumKeyHeads = get(.linearNumKeyHeads)
        self.linearValueHeadDim = get(.linearValueHeadDim)
        self.linearKeyHeadDim = get(.linearKeyHeadDim)
        self.linearConvKernelDim = get(.linearConvKernelDim)
        self.attnOutputGate = get(.attnOutputGate)
        self.partialRotaryFactor = get(.partialRotaryFactor)
        self.normTopkProb = get(.normTopkProb)
        self.headDimOverride = get(.headDim)
    }
}

/// Distinguishes the two JANG weight formats:
///
///   - `.mxq`  : legacy / standard JANG (affine N-bit per group). Dense models.
///               Recognized via either `format == "jang"` (v1) or
///               `weight_format == "mxq"` / `version == 2` (v2 safetensors).
///
///   - `.mxtq` : JANGTQ — Turbo Quant, codebook + Hadamard, sub-2-bit MoE.
///               Recognized via `weight_format == "mxtq"`. Routed-expert
///               weights are stored as 2-bit codebook indices + per-row norms;
///               attention / shared-expert / embed / lm_head stay affine
///               at the bit-width listed in `mxtq_bits`.
public enum JANGFormat: String, Sendable {
    case mxq
    case mxtq
}

/// JANG quantization configuration (from jang_config.json).
///
/// Format-aware: the fields populated depend on `format`. For `.mxq` we
/// fill `targetBits / actualBits / blockSize / bitWidthsUsed`. For `.mxtq`
/// we fill `mxtqSeed / mxtqBits` (the per-component bit map).
public struct JANGQuantConfig: Sendable {
    public let format: JANGFormat
    public let formatVersion: String
    public let sourceModelName: String

    // .mxq fields (zero/empty for .mxtq)
    public let targetBits: Float
    public let actualBits: Float
    public let blockSize: Int
    public let bitWidthsUsed: [Int]
    public let totalWeightBytes: Int

    // .mxtq fields (zero/empty for .mxq)
    public let mxtqSeed: Int
    public let mxtqBits: [String: Int]   // {"attention": 8, "routed_expert": 2, ...}
    public let mxtqGroupSize: Int
    public let mxtqBitsDefault: Int

    public init(from dict: [String: Any]) throws {
        // Detect format. Three accepted shapes:
        //   v1 (legacy):  {"format": "jang", "format_version": "1.0", ...}
        //   v2 mxq:       {"version": 2, "weight_format": "mxq", ...}
        //   v2 mxtq:      {"version": 2, "weight_format": "mxtq", "profile": "JANGTQ_2L",
        //                  "mxtq_seed": 42, "mxtq_bits": {...}}
        let detectedFormat: JANGFormat
        if let weightFormat = dict["weight_format"] as? String {
            switch weightFormat {
            case "mxtq":
                detectedFormat = .mxtq
            case "mxq", "jang", "jjqf":
                detectedFormat = .mxq
            default:
                throw JANGError.invalidFormat(
                    "unknown weight_format '\(weightFormat)' — expected 'mxq' or 'mxtq'"
                )
            }
        } else if let legacyFormat = dict["format"] as? String, legacyFormat == "jang" {
            detectedFormat = .mxq
        } else {
            throw JANGError.invalidFormat(
                "jang_config.json must have either 'weight_format' or 'format' field"
            )
        }
        self.format = detectedFormat

        // version: integer for v2, string "1.0" for v1
        if let v = dict["version"] as? Int {
            self.formatVersion = String(v)
        } else if let v = dict["format_version"] as? String {
            self.formatVersion = v
        } else {
            self.formatVersion = "1"
        }

        let source = dict["source_model"] as? [String: Any] ?? [:]
        self.sourceModelName = source["name"] as? String ?? "unknown"

        let quant = dict["quantization"] as? [String: Any] ?? [:]

        switch detectedFormat {
        case .mxq:
            self.targetBits = (quant["target_bits"] as? NSNumber)?.floatValue ?? 2.5
            self.actualBits = (quant["actual_bits"] as? NSNumber)?.floatValue ?? self.targetBits
            self.blockSize = (quant["block_size"] as? Int) ?? 64
            self.bitWidthsUsed = (quant["bit_widths_used"] as? [Int]) ?? [2, 3, 4]
            let runtime = dict["runtime"] as? [String: Any] ?? [:]
            self.totalWeightBytes = runtime["total_weight_bytes"] as? Int ?? 0
            self.mxtqSeed = 0
            self.mxtqBits = [:]
            self.mxtqGroupSize = 0
            self.mxtqBitsDefault = 0

        case .mxtq:
            self.targetBits = 0
            self.actualBits = 0
            self.blockSize = 0
            self.bitWidthsUsed = []
            self.totalWeightBytes = 0
            self.mxtqSeed = (dict["mxtq_seed"] as? Int) ?? 42
            self.mxtqBits = (dict["mxtq_bits"] as? [String: Int]) ?? [:]
            self.mxtqGroupSize = (quant["group_size"] as? Int) ?? 64
            self.mxtqBitsDefault = (quant["bits_default"] as? Int) ?? 2
        }
    }

    /// Bits for a logical component (attention / routed_expert / shared_expert / embed_tokens / lm_head).
    /// Returns the default if not specified in the per-component map.
    public func bits(for component: String) -> Int {
        return mxtqBits[component] ?? mxtqBitsDefault
    }
}

/// Combined model + quantization config.
public struct JANGModelConfig: Sendable {
    public let model: ModelConfig
    public let quant: JANGQuantConfig
    public let modelPath: URL

    public static func load(from path: URL) throws -> JANGModelConfig {
        // Load config.json
        let configURL = path.appendingPathComponent("config.json")
        let configData = try Data(contentsOf: configURL)
        let decoder = JSONDecoder()
        let model = try decoder.decode(ModelConfig.self, from: configData)

        // Load jang_config.json
        let mxqConfigURL = path.appendingPathComponent("jang_config.json")
        let mxqData = try Data(contentsOf: mxqConfigURL)
        guard let mxqDict = try JSONSerialization.jsonObject(with: mxqData) as? [String: Any] else {
            throw JANGError.invalidFormat("jang_config.json is not a valid JSON object")
        }
        let quant = try JANGQuantConfig(from: mxqDict)

        return JANGModelConfig(model: model, quant: quant, modelPath: path)
    }
}
