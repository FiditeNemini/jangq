import Foundation
import Observation

@Observable
@MainActor
final class TestInferenceViewModel {
    var messages: [ChatMessage] = []
    var promptText: String = ""
    var systemPrompt: String = "You are a helpful assistant."
    var temperature: Double = 0.0
    var maxTokens: Int = 150
    var pendingImagePath: URL?
    var pendingVideoPath: URL?
    var isGenerating: Bool = false
    var lastError: String?
    var lastTokensPerSec: Double = 0
    var lastPeakRssMb: Double = 0
    /// M121: opt-in toggle for reasoning models (GLM-5.1 / Qwen3.6 / MiniMax
    /// M2.7). When true, passes --no-thinking to jang_tools inference so the
    /// chat template doesn't wrap the prompt with <think>…</think>. Default
    /// false preserves existing behavior for users running reasoning
    /// benchmarks who expect the full thinking block.
    var skipThinking: Bool = false

    private var runner: InferenceRunner

    init(modelPath: URL) {
        self.runner = InferenceRunner(modelPath: modelPath)
    }

    func send() async {
        guard !isGenerating else { return }
        let trimmed = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        let userMsg = ChatMessage(role: .user, text: trimmed, imagePath: pendingImagePath?.path)
        messages.append(userMsg)
        promptText = ""
        isGenerating = true
        lastError = nil

        // Build prompt — for chat models we'd apply the template here; for v1
        // we pass prompt directly. The Python side already handles chat template
        // when tokenizer has one (via the model's apply_chat_template).
        let prompt = trimmed
        let imagePath = pendingImagePath
        let videoPath = pendingVideoPath

        do {
            let result = try await runner.generate(
                prompt: prompt,
                maxTokens: maxTokens,
                temperature: temperature,
                imagePath: imagePath,
                videoPath: videoPath,
                noThinking: skipThinking
            )
            let msg = ChatMessage(
                role: .assistant,
                text: result.text,
                tokensPerSec: result.tokensPerSec,
                elapsedS: result.elapsedS
            )
            messages.append(msg)
            lastTokensPerSec = result.tokensPerSec
            lastPeakRssMb = result.peakRssMb
            pendingImagePath = nil
            pendingVideoPath = nil
        } catch let e as InferenceError {
            // Don't surface user-initiated cancellation as an error banner —
            // the user pressed Cancel deliberately; showing a red "generation
            // cancelled by user" message would feel like a failure.
            if !e.wasCancelled {
                lastError = e.message
            }
        } catch {
            lastError = "\(error)"
        }
        isGenerating = false
    }

    func cancel() async {
        await runner.cancel()
        isGenerating = false
    }

    func clear() {
        messages.removeAll()
        lastError = nil
        lastTokensPerSec = 0
        lastPeakRssMb = 0
    }

    func exportTranscript(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let payload: [String: Any] = [
            "system_prompt": systemPrompt,
            "temperature": temperature,
            "max_tokens": maxTokens,
            "messages": (try? encoder.encode(messages)).flatMap { try? JSONSerialization.jsonObject(with: $0) } ?? [],
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])
        try data.write(to: url)
    }
}
