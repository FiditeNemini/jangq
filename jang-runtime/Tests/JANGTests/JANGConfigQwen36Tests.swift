import XCTest
@testable import JANG

/// Verifies `ModelConfig` decodes the Qwen3.5/3.6 text_config-nested layout
/// AND the MiniMax-style top-level layout without regression.
final class JANGConfigQwen36Tests: XCTestCase {

    // Minimal Qwen3.6-shaped config. Reflects real qwen3_5_moe layout:
    // everything the decoder cares about lives under text_config.
    private static let qwen36ConfigJSON = #"""
    {
      "model_type": "qwen3_5_moe",
      "architectures": ["Qwen3_5MoeForConditionalGeneration"],
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "intermediate_size": 512,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": false,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_num_value_heads": 32,
        "linear_num_key_heads": 16,
        "linear_value_head_dim": 128,
        "linear_key_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "attn_output_gate": true,
        "partial_rotary_factor": 0.25,
        "norm_topk_prob": true,
        "rope_parameters": {"rope_theta": 10000000}
      }
    }
    """#

    // Real MiniMax-style top-level config (subset).
    private static let minimaxConfigJSON = #"""
    {
      "model_type": "minimax_m2",
      "architectures": ["MiniMaxM2ForCausalLM"],
      "hidden_size": 3072,
      "intermediate_size": 8192,
      "num_hidden_layers": 62,
      "num_attention_heads": 48,
      "num_key_value_heads": 8,
      "vocab_size": 200064,
      "rms_norm_eps": 1e-05,
      "rope_theta": 5000000,
      "tie_word_embeddings": false,
      "num_local_experts": 256,
      "num_experts_per_tok": 8,
      "moe_intermediate_size": 1536
    }
    """#

    func testDecodeQwen36ConfigFromTextConfig() throws {
        let data = Self.qwen36ConfigJSON.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(ModelConfig.self, from: data)

        XCTAssertEqual(cfg.modelType, "qwen3_5_moe")
        XCTAssertEqual(cfg.hiddenSize, 2048)
        XCTAssertEqual(cfg.numHiddenLayers, 40)
        XCTAssertEqual(cfg.numAttentionHeads, 16)
        XCTAssertEqual(cfg.numKeyValueHeads, 2)
        XCTAssertEqual(cfg.headDim, 256, "explicit head_dim must override hidden/num_heads")
        XCTAssertEqual(cfg.vocabSize, 248320)
        XCTAssertEqual(cfg.numLocalExperts, 256, "num_experts must feed numLocalExperts")
        XCTAssertEqual(cfg.numExpertsPerTok, 8)
        XCTAssertEqual(cfg.moeIntermediateSize, 512)
        XCTAssertEqual(cfg.sharedExpertIntermediateSize, 512)
        XCTAssertEqual(cfg.fullAttentionInterval, 4)
        XCTAssertEqual(cfg.layerTypes?.count, 4)
        XCTAssertEqual(cfg.linearNumValueHeads, 32)
        XCTAssertEqual(cfg.linearNumKeyHeads, 16)
        XCTAssertEqual(cfg.linearConvKernelDim, 4)
        XCTAssertEqual(cfg.attnOutputGate, true)
        XCTAssertEqual(cfg.partialRotaryFactor, 0.25)
        XCTAssertEqual(cfg.normTopkProb, true)

        XCTAssertTrue(cfg.isMoE)
        XCTAssertTrue(cfg.isHybridAttention)
        XCTAssertTrue(cfg.hasSharedExpert)
        XCTAssertTrue(cfg.hasAttnOutputGate)
        XCTAssertTrue(cfg.isSoftmaxRouter)
        XCTAssertFalse(cfg.isMLA)
    }

    func testDecodeMiniMaxConfigTopLevel() throws {
        let data = Self.minimaxConfigJSON.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(ModelConfig.self, from: data)

        XCTAssertEqual(cfg.modelType, "minimax_m2")
        XCTAssertEqual(cfg.hiddenSize, 3072)
        XCTAssertEqual(cfg.numHiddenLayers, 62)
        XCTAssertEqual(cfg.numAttentionHeads, 48)
        XCTAssertEqual(cfg.numLocalExperts, 256)
        XCTAssertEqual(cfg.numExpertsPerTok, 8)

        XCTAssertTrue(cfg.isMoE)
        XCTAssertFalse(cfg.isHybridAttention)
        XCTAssertFalse(cfg.hasSharedExpert)
        XCTAssertFalse(cfg.hasAttnOutputGate)
        XCTAssertFalse(cfg.isSoftmaxRouter, "minimax uses sigmoid+bias router (no norm_topk_prob)")
        XCTAssertFalse(cfg.isMLA)
    }

    /// Extra: load the real Qwen3.6 config from disk if available (CI may skip).
    /// Override with JANG_TEST_QWEN36_CONFIG env var.
    func testDecodeRealQwen36ConfigIfAvailable() throws {
        let envOverride = ProcessInfo.processInfo.environment["JANG_TEST_QWEN36_CONFIG"]
        let home = FileManager.default.homeDirectoryForCurrentUser
        let p = envOverride.map { URL(fileURLWithPath: $0) }
            ?? home.appendingPathComponent(".cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/7da1103448ba36029c34ce1a9a741dfe93ee0c50/config.json")
        guard FileManager.default.fileExists(atPath: p.path) else {
            throw XCTSkip("Qwen3.6 source not downloaded locally")
        }
        let data = try Data(contentsOf: p)
        let cfg = try JSONDecoder().decode(ModelConfig.self, from: data)
        XCTAssertEqual(cfg.modelType, "qwen3_5_moe")
        XCTAssertEqual(cfg.numHiddenLayers, 40)
        XCTAssertEqual(cfg.numLocalExperts, 256)
        XCTAssertTrue(cfg.isHybridAttention)
        XCTAssertTrue(cfg.hasSharedExpert)
    }
}
