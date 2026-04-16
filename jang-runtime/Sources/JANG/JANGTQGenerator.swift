/*
 * JANGTQ Generator — autoregressive decode loop over JANGTQModel.
 * Created by Jinho Jang (eric@jangq.ai)
 *
 * Wraps a JANGTQModel + JANGTQTokenizer + JANGTQSampler into a single
 * `generate(prompt:, maxTokens:)` API. Mirrors the shape of mlx_lm's
 * `generate()` so callers don't have to think about token IDs, the cache,
 * or stop conditions.
 *
 * Usage:
 *   let tok = try JANGTQTokenizer(modelDir: modelDir)
 *   let model = try JANGTQModel(bundle: bundle, context: ctx)
 *   let gen = JANGTQGenerator(model: model, tokenizer: tok)
 *   let result = try gen.generate(
 *       messages: [JANGTQChatMessage(role: "user", content: "Hello!")],
 *       maxTokens: 200
 *   )
 *   print(result.text)
 */

import Foundation
import Metal

public struct JANGTQGenerationResult {
    public let text: String              // Decoded answer (post-`<think>` strip)
    public let rawText: String           // Raw decode including `<think>...</think>`
    public let promptTokens: Int
    public let outputTokens: Int
    public let stopReason: StopReason
    public let elapsedSec: Double
    public var tokensPerSec: Double {
        elapsedSec > 0 ? Double(outputTokens) / elapsedSec : 0
    }

    public enum StopReason: String {
        case stopToken
        case maxTokens
        case error
    }
}

public final class JANGTQGenerator {
    public let model: JANGTQModel
    public let tokenizer: JANGTQTokenizer
    public let sampler: JANGTQSampler

    public init(model: JANGTQModel, tokenizer: JANGTQTokenizer, sampler: JANGTQSampler = JANGTQSampler()) {
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
    }

    /// Generate a response to a single user message.
    public func generate(
        userMessage: String,
        system: String? = nil,
        maxTokens: Int = 256,
        verbose: Bool = false
    ) throws -> JANGTQGenerationResult {
        return try generate(
            messages: [JANGTQChatMessage(role: "user", content: userMessage)],
            system: system,
            maxTokens: maxTokens,
            verbose: verbose
        )
    }

    /// Generate a response for a list of chat messages.
    /// Resets the model's KV cache before generation.
    public func generate(
        messages: [JANGTQChatMessage],
        system: String? = nil,
        maxTokens: Int = 256,
        verbose: Bool = false
    ) throws -> JANGTQGenerationResult {
        let promptIds = tokenizer.applyChatTemplate(
            messages: messages, system: system
        )

        if verbose {
            print("Prompt tokens: \(promptIds.count)")
        }

        model.reset()
        let t0 = Date()

        // === Prefill: feed each prompt token one at a time ===
        // (Batched prefill is a future optimization; one-at-a-time matches
        // the engine's single-token forward and works fine for short prompts.)
        var lastLogits: MTLBuffer? = nil
        for (i, tok) in promptIds.enumerated() {
            lastLogits = try model.forward(tokenId: tok, position: i)
        }

        // === Decode loop ===
        var generatedIds: [Int] = []
        var stopReason: JANGTQGenerationResult.StopReason = .maxTokens

        // Sampling loop
        for step in 0..<maxTokens {
            guard let logits = lastLogits else { break }
            let nextId = sampler.argmax(logits: logits, vocabSize: model.config.vocabSize)
            if tokenizer.stopTokenIds.contains(nextId) {
                stopReason = .stopToken
                break
            }
            generatedIds.append(nextId)
            if verbose {
                let s = tokenizer.decodeToken(nextId)
                if !s.isEmpty { print(s, terminator: ""); fflush(stdout) }
            }
            if step + 1 == maxTokens { break }
            // Forward the new token at the next position
            let pos = promptIds.count + step
            lastLogits = try model.forward(tokenId: nextId, position: pos)
        }

        let elapsed = Date().timeIntervalSince(t0)
        let raw = tokenizer.decode(generatedIds)
        let stripped = tokenizer.stripThinking(raw)

        return JANGTQGenerationResult(
            text: stripped,
            rawText: raw,
            promptTokens: promptIds.count,
            outputTokens: generatedIds.count,
            stopReason: stopReason,
            elapsedSec: elapsed
        )
    }
}
