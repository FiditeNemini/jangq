// NemotronHOmni.swift
// Native multimodal wrapper for Nemotron-3-Nano-Omni-30B-A3B-Reasoning.
//
// Combines:
//   • LLM (NemotronH.swift, already in vmlx-swift-lm)
//   • RADIO ViT vision tower (RADIOVision.swift, this dir)
//   • Parakeet Conformer encoder (Parakeet.swift, this dir)
//   • mlp1 vision projector + sound_projection (Projectors.swift)
//   • Image / video / audio preprocessors (CIImage + AVAudioFile + Accelerate)
//
// Stage-3 native Swift multimodal. Drop-in replacement for the Python
// `OmniSession` once parity is validated.
//
// Status: skeleton — fill in `extractImageEmbeds`, `extractAudioEmbeds`,
// and weight-loading remap functions per the matching Python module
// (jang_tools/nemotron_omni/model.py). The LLM portion already works via
// vmlx-swift-lm/Libraries/MLXLLM/Models/NemotronH.swift.

import Foundation
import MLX
import MLXNN
import MLXLLM
import MLXLMCommon

@available(macOS 14.0, *)
public class NemotronHOmni {
    public let bundlePath: URL
    public let imageSize: Int
    public let downsampleRatio: Float

    private let imgContextTokenId: Int
    private let videoContextTokenId: Int
    private let soundContextTokenId: Int

    public let llm: NemotronHModel        // existing in vmlx-swift-lm
    public let visionModel: RADIOVisionModel
    public let mlp1: VisionMLPProjector
    public let soundEncoder: ParakeetEncoder
    public let soundProjection: SoundProjector
    public let tokenizer: any Tokenizer    // from MLXLMCommon

    private var cache: [KVCache?]?         // persistent multi-turn cache
    private let eosIds: Set<Int> = [11]

    public init(bundlePath: URL) throws {
        self.bundlePath = bundlePath

        // Read config_omni.json for token IDs + dims
        let omniConfigURL = bundlePath.appendingPathComponent("config_omni.json")
        let cfgData = try Data(contentsOf: omniConfigURL)
        let cfg = try JSONSerialization.jsonObject(with: cfgData) as! [String: Any]
        self.imageSize = cfg["force_image_size"] as? Int ?? 512
        self.downsampleRatio = cfg["downsample_ratio"] as? Float ?? 0.5
        self.imgContextTokenId = cfg["img_context_token_id"] as! Int
        self.videoContextTokenId = cfg["video_context_token_id"] as? Int ?? imgContextTokenId
        self.soundContextTokenId = cfg["sound_context_token_id"] as! Int

        // Load LLM via existing JangLoader path (auto-detects mlx vs jangtq)
        // self.llm = try JangLoader.loadNemotronH(at: bundlePath)
        // self.tokenizer = try await loadTokenizer(at: bundlePath)
        fatalError("TODO: wire up LLM + tokenizer via JangLoader")
    }

    // MARK: - Multimodal embedding extraction

    /// PIL-equivalent images → image embeds
    public func extractImageEmbeds(images: [CIImage]) async throws -> MLXArray {
        // 1. NVLM 1-D tile preprocessing → (N_tiles, 3, 512, 512) MLXArray
        // 2. visionModel(pixelValues) → (N_tiles, 10 + 1024, 1280)
        // 3. Strip first 10 cls tokens → (N_tiles, 1024, 1280)
        // 4. Reshape (N_tiles, 32, 32, 1280)
        // 5. pixelShuffle(scale=0.5) → (N_tiles, 16, 16, 5120)
        // 6. Flatten → (N_tiles, 256, 5120)
        // 7. mlp1(...) → (N_tiles, 256, llmHidden=2688)
        fatalError("TODO")
    }

    /// 16 kHz mono audio → audio embeds projected to LLM hidden
    public func extractAudioEmbeds(audioData: [Float]) async throws -> MLXArray {
        // 1. Mel STFT (n_fft=512, hop=160, win=400, n_mels=128, slaney norm)
        //    via Accelerate.vDSP — 16 kHz mono input
        // 2. soundEncoder(mel) → (1, F_sub, 1024)
        // 3. soundProjection(...) → (1, F_sub, llmHidden)
        fatalError("TODO")
    }

    // MARK: - Multi-turn chat

    public func reset() {
        self.cache = nil
    }

    private func ensureCache() {
        if cache == nil {
            cache = llm.newCache(parameters: nil)
        }
    }

    public func turn(
        text: String,
        images: [CIImage]? = nil,
        audio: [Float]? = nil,
        maxTokens: Int = 256,
        temperature: Float = 0.6,
        topP: Float = 0.95,
        enableThinking: Bool = true
    ) async throws -> String {
        ensureCache()

        // 1. Encode multimodal inputs
        var imageEmbeds: MLXArray?
        var audioEmbeds: MLXArray?
        var nImageTokens = 0
        var nAudioTokens = 0
        if let imgs = images, !imgs.isEmpty {
            let ie = try await extractImageEmbeds(images: imgs)
            imageEmbeds = ie
            nImageTokens = ie.shape[0] * ie.shape[1]
        }
        if let aud = audio {
            let ae = try await extractAudioEmbeds(audioData: aud)
            audioEmbeds = ae
            nAudioTokens = ae.shape[0] * ae.shape[1]
        }

        // 2. Build prompt with multimodal placeholders
        var media = ""
        if nImageTokens > 0 {
            media += "<img>" + String(repeating: "<image>", count: nImageTokens) + "</img>\n"
        }
        if nAudioTokens > 0 {
            media += "<sound>" + String(repeating: "<so_embedding>", count: nAudioTokens) + "</sound>\n"
        }
        let userMessage = media + text
        // tokenizer.applyChatTemplate(...)
        // Embed text → MLX, inject multimodal embeds at <image>/<so_embedding> positions

        // 3. Decode using llm.callAsFunction(inputsEmbeds:cache:) — analogous
        // to Python `_inline_decode`. The existing NemotronHModel forward
        // takes input_ids; we need to add an `inputsEmbeds` parameter.

        fatalError("TODO: implement embed-injection prefill + sample loop")
    }
}
