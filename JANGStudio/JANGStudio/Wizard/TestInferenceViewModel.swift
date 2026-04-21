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
    /// M225 (iter 150): last observed load time in seconds. Populated
    /// from `InferenceResult.loadTimeS` on every successful generate.
    /// Used by the working-status label to calibrate user expectations:
    /// on subsequent sends, show the actual previous load time ("loading
    /// model… (last run took 18s)") so the user knows what to expect.
    var lastLoadTimeS: Double? = nil
    /// M225 (iter 150): wall-clock start of the current generate() call.
    /// Drives the elapsed-seconds counter in the working-status label so
    /// a stranger sees "Loading model… (14s elapsed)" rather than a
    /// silent spinner. Without this, users watching a 30GB MoE load
    /// can't tell if the app is stuck or still making progress.
    var generateStartedAt: Date? = nil
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
        generateStartedAt = Date()   // M225 (iter 150): start of wall-clock for "Loading… (Ns elapsed)" label.
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
            // M225 (iter 150): record actual load time so the NEXT send's
            // working-status label can cite a real number ("last run took
            // 18s to load") rather than a generic "could take 30s".
            lastLoadTimeS = result.loadTimeS
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
        generateStartedAt = nil
    }

    /// M225 (iter 150): honest working-status label that distinguishes
    /// model LOAD (15-30s for large MoE) from GENERATION (a few seconds).
    /// Pre-M225 the UI always said "Generating..." which misled users
    /// into thinking the model was slow at generation when in fact
    /// loading dominated. Each `Send` spawns a fresh InferenceRunner
    /// subprocess that reloads the model from disk — so every send has
    /// this cost. When we know the previous load time, cite it ("last
    /// run took 18s") so the user calibrates expectations.
    func workingStatusLabel() -> String {
        let elapsed: Int = {
            guard let start = generateStartedAt else { return 0 }
            return Int(Date().timeIntervalSince(start))
        }()
        let elapsedPart = elapsed > 0 ? " (\(elapsed)s elapsed)" : ""
        // First Send of the session — no prior load time to cite.
        if lastLoadTimeS == nil {
            return "Loading model + generating\(elapsedPart)… large MoE models can take 30s+ on first run."
        }
        // Subsequent sends — cite the previous load time to calibrate.
        let prev = Int(lastLoadTimeS!.rounded())
        return "Loading model + generating\(elapsedPart)… previous run loaded in \(prev)s. Each Send reloads the model."
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
