// JANGKit.Model — unified high-level API for loading + generating with JANG models.
//
// Under the hood this composes loadModel + JANGInferenceEngine + JANGTokenizer
// + JANGSampler. JANGTQ support via this facade is deferred; for JANGTQ models
// use JANGTQGenerator directly from the JANG module.
//
// API gaps (Phase P2 work):
//   - applyChatTemplate: JANGTokenizer only has encodeChatPrompt(system:user:),
//     which implements the Qwen im_start/im_end template. The Model.chat() method
//     uses that directly and documents the limitation. A generic Jinja template
//     executor would be needed to support other families (LLaMA, Mistral, etc.).
//   - JANGSampler.sample() requires an explicit vocabSize parameter; we read this
//     from the loaded model config.

import Foundation
import Metal
import JANG

extension JANGKit {

    // MARK: - GenerationResult

    /// Result of a generate() call — the generated text plus timing info.
    public struct GenerationResult: Sendable, Equatable {
        public let text: String
        public let tokens: Int
        public let elapsedSeconds: Double
        public let tokensPerSecond: Double
        public let finishReason: FinishReason

        public enum FinishReason: String, Sendable, Equatable {
            case stop       // EOS token produced
            case maxTokens  // hit the maxTokens cap
            case cancelled  // reserved for future streaming cancellation
            case error      // mid-generation exception (text contains partial output)
        }
    }

    // MARK: - SamplingConfig

    /// Sampling parameters for generate().
    public struct SamplingConfig: Sendable {
        public var temperature: Double
        public var topP: Double
        public var topK: Int
        public var maxTokens: Int

        public init(temperature: Double = 0.0,
                    topP: Double = 1.0,
                    topK: Int = 0,
                    maxTokens: Int = 200) {
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.maxTokens = maxTokens
        }

        /// Convenience: converts to the lower-level SamplingParams.
        fileprivate var asSamplingParams: SamplingParams {
            var p = SamplingParams()
            p.temperature = Float(temperature)
            p.topP = Float(topP)
            p.topK = topK
            p.repetitionPenalty = 1.0
            return p
        }
    }

    // MARK: - ModelError

    /// Errors surfaced by the high-level JANGKit API.
    public enum ModelError: Error, LocalizedError {
        case metalDeviceUnavailable
        case modelLoadFailed(String)
        case tokenizerLoadFailed(String)
        case generationFailed(String)
        /// JANGTQ models are not yet supported by the JANGKit facade.
        /// Use JANGTQGenerator directly from the JANG module for JANGTQ inference.
        case jangtqNotYetSupported

        public var errorDescription: String? {
            switch self {
            case .metalDeviceUnavailable:
                return "Metal device not available — JANG requires Apple Silicon."
            case .modelLoadFailed(let m):
                return "Model load failed: \(m)"
            case .tokenizerLoadFailed(let m):
                return "Tokenizer load failed: \(m)"
            case .generationFailed(let m):
                return "Generation failed: \(m)"
            case .jangtqNotYetSupported:
                return "JANGTQ models are not yet supported by JANGKit. "
                    + "Use JANGTQGenerator directly from the JANG module."
            }
        }
    }

    // MARK: - Model

    /// A loaded JANG model ready for single-shot generation.
    ///
    /// Create via `Model.load(at:)`. Thread safety: this type is an actor; all
    /// generation state (KV cache position inside JANGInferenceEngine) is serialized.
    ///
    /// JANGTQ models (`weight_format == "mxtq"` in jang_config.json) are not yet
    /// supported through this facade — `load(at:)` will throw `.jangtqNotYetSupported`.
    /// Use `JANGTQGenerator` directly from the `JANG` module for those models.
    public actor Model {
        private let mxqModel: MXQModel
        private let engine: JANGInferenceEngine
        private let tokenizer: JANGTokenizer
        private let vocabSize: Int
        /// The directory from which this model was loaded.
        public nonisolated let modelURL: URL

        private init(mxqModel: MXQModel,
                     engine: JANGInferenceEngine,
                     tokenizer: JANGTokenizer,
                     modelURL: URL) {
            self.mxqModel = mxqModel
            self.engine = engine
            self.tokenizer = tokenizer
            self.vocabSize = mxqModel.config.model.vocabSize
            self.modelURL = modelURL
        }

        // MARK: load

        /// Load a JANG model from a directory.
        ///
        /// Auto-detects format via jang_config.json. Throws `.jangtqNotYetSupported` for
        /// JANGTQ (`weight_format == "mxtq"`) models.
        ///
        /// - Parameter url: Path to the model directory (must contain jang_config.json,
        ///   config.json, tokenizer.json, and one or more .safetensors shards).
        public static func load(at url: URL) async throws -> Model {
            // 1. Check jang_config.json exists
            let configURL = url.appendingPathComponent("jang_config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                throw ModelError.modelLoadFailed(
                    "jang_config.json not found at \(configURL.path)"
                )
            }

            // 2. Detect JANGTQ — bail early with a clear error
            if let data = try? Data(contentsOf: configURL),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                // v2 shape: top-level "weight_format" == "mxtq"
                if (obj["weight_format"] as? String)?.lowercased() == "mxtq" {
                    throw ModelError.jangtqNotYetSupported
                }
                // Legacy / alternate shape: nested quantization.method == "jangtq"
                if let quant = obj["quantization"] as? [String: Any],
                   (quant["method"] as? String)?.lowercased() == "jangtq" {
                    throw ModelError.jangtqNotYetSupported
                }
            }

            // 3. Create Metal device (JANGMetalDevice owns its own MTLDevice)
            let metalDevice: JANGMetalDevice
            do {
                metalDevice = try JANGMetalDevice()
            } catch {
                throw ModelError.metalDeviceUnavailable
            }

            // 4. Load model weights
            let mxqModel: MXQModel
            do {
                mxqModel = try loadModel(url: url, device: metalDevice.device)
            } catch {
                throw ModelError.modelLoadFailed("\(error)")
            }

            // 5. Create inference engine
            let engine: JANGInferenceEngine
            do {
                engine = try JANGInferenceEngine(model: mxqModel, metalDevice: metalDevice)
            } catch {
                throw ModelError.modelLoadFailed("inference engine init: \(error)")
            }

            // 6. Load tokenizer — JANGTokenizer takes the path to tokenizer.json
            let tokenizerFileURL = url.appendingPathComponent("tokenizer.json")
            let tokenizer: JANGTokenizer
            do {
                tokenizer = try JANGTokenizer(tokenizerPath: tokenizerFileURL)
            } catch {
                throw ModelError.tokenizerLoadFailed("\(error)")
            }

            return Model(
                mxqModel: mxqModel,
                engine: engine,
                tokenizer: tokenizer,
                modelURL: url
            )
        }

        // MARK: generate

        /// Generate text from a raw prompt string.
        ///
        /// This is one-shot (not streaming) — the full response is returned as a single
        /// `GenerationResult`. For streaming use the lower-level `JANGInferenceEngine`
        /// directly.
        ///
        /// - Parameters:
        ///   - prompt: The raw prompt string (already formatted with any special tokens
        ///     you need). For chat formatting use `chat(messages:config:)` instead.
        ///   - config: Sampling parameters. Defaults to greedy (temperature=0).
        public func generate(
            prompt: String,
            config: SamplingConfig = SamplingConfig()
        ) async throws -> GenerationResult {
            let t0 = Date()

            // Tokenize prompt
            let promptIds = tokenizer.encode(prompt)

            // Reset KV cache for a fresh sequence
            engine.reset()

            // Prefill: forward every prompt token to build the KV cache
            for tokenId in promptIds {
                _ = try engine.forward(tokenId: tokenId)
            }

            // Decode loop
            let sampler = JANGSampler()
            let samplingParams = config.asSamplingParams
            var generatedIds: [Int] = []
            var reason: GenerationResult.FinishReason = .maxTokens
            let eosId = tokenizer.eosTokenId

            for step in 0..<config.maxTokens {
                // For step 0 we need the logits produced by the last prefill token
                // which engine.forward already returned — so on step 0 we use the
                // previous token (last prompt token) to generate the next one.
                // On step > 0 we use the last generated token.
                let inputTokenId: Int
                if step == 0 {
                    // Run one forward pass for the first decode step
                    inputTokenId = promptIds.last ?? eosId
                } else {
                    inputTokenId = generatedIds.last ?? eosId
                }

                let logits = try engine.forward(tokenId: inputTokenId)
                let nextId = sampler.sample(
                    logits: logits,
                    vocabSize: vocabSize,
                    params: samplingParams
                )

                if nextId == eosId {
                    reason = .stop
                    break
                }
                generatedIds.append(nextId)
            }

            let text = tokenizer.decode(generatedIds)
            let elapsed = Date().timeIntervalSince(t0)
            let tps = elapsed > 0 ? Double(generatedIds.count) / elapsed : 0

            return GenerationResult(
                text: text,
                tokens: generatedIds.count,
                elapsedSeconds: elapsed,
                tokensPerSecond: tps,
                finishReason: reason
            )
        }

        // MARK: chat

        /// Apply the Qwen im_start/im_end chat template and generate.
        ///
        /// **Limitation (Phase P2 work):** `JANGTokenizer` implements the Qwen
        /// `<|im_start|>` / `<|im_end|>` template via `encodeChatPrompt(system:user:)`.
        /// It does not support arbitrary Jinja templates (LLaMA, Mistral, etc.).
        /// For other model families, format your prompt manually and call `generate`
        /// directly, or use the Python `mlx_lm` path which has full Jinja support.
        ///
        /// - Parameters:
        ///   - system: Optional system prompt. Defaults to "You are a helpful assistant."
        ///   - user: The user message content.
        ///   - config: Sampling parameters.
        public func chat(
            system: String? = nil,
            user: String,
            config: SamplingConfig = SamplingConfig()
        ) async throws -> GenerationResult {
            let t0 = Date()

            // encodeChatPrompt returns [Int] directly (no throws, no generic messages array)
            let promptIds = tokenizer.encodeChatPrompt(system: system, user: user)

            engine.reset()

            // Prefill
            for tokenId in promptIds {
                _ = try engine.forward(tokenId: tokenId)
            }

            // Decode
            let sampler = JANGSampler()
            let samplingParams = config.asSamplingParams
            var generatedIds: [Int] = []
            var reason: GenerationResult.FinishReason = .maxTokens
            let eosId = tokenizer.eosTokenId

            for step in 0..<config.maxTokens {
                let inputTokenId: Int
                if step == 0 {
                    inputTokenId = promptIds.last ?? eosId
                } else {
                    inputTokenId = generatedIds.last ?? eosId
                }

                let logits = try engine.forward(tokenId: inputTokenId)
                let nextId = sampler.sample(
                    logits: logits,
                    vocabSize: vocabSize,
                    params: samplingParams
                )

                if nextId == eosId {
                    reason = .stop
                    break
                }
                generatedIds.append(nextId)
            }

            let text = tokenizer.decode(generatedIds)
            let elapsed = Date().timeIntervalSince(t0)
            let tps = elapsed > 0 ? Double(generatedIds.count) / elapsed : 0

            return GenerationResult(
                text: text,
                tokens: generatedIds.count,
                elapsedSeconds: elapsed,
                tokensPerSecond: tps,
                finishReason: reason
            )
        }
    }
}
