// JANGStudio/JANGStudio/Runner/InferenceRunner.swift
import Foundation

struct InferenceResult: Codable, Equatable {
    let text: String
    let tokens: Int
    let tokensPerSec: Double
    let elapsedS: Double
    let loadTimeS: Double?
    let peakRssMb: Double
    let model: String

    enum CodingKeys: String, CodingKey {
        case text, tokens, model
        case tokensPerSec = "tokens_per_sec"
        case elapsedS = "elapsed_s"
        case loadTimeS = "load_time_s"
        case peakRssMb = "peak_rss_mb"
    }
}

struct InferenceError: Error, Equatable {
    let message: String
    let code: Int32
}

/// Wraps `python -m jang_tools inference` as a one-shot subprocess.
/// Not streaming in v1 — returns the full response at once. Each call spawns
/// a fresh Python process that loads the model, generates, exits.
/// TODO for v1.1: add persistent-process streaming via stdin/stdout pipe.
actor InferenceRunner {
    nonisolated let modelPath: URL
    private var currentProcess: Process?

    init(modelPath: URL) {
        self.modelPath = modelPath
    }

    func generate(prompt: String,
                  maxTokens: Int = 100,
                  temperature: Double = 0.0,
                  imagePath: URL? = nil,
                  videoPath: URL? = nil) async throws -> InferenceResult {
        var args = [
            "-m", "jang_tools", "inference",
            "--model", modelPath.path,
            "--prompt", prompt,
            "--max-tokens", String(maxTokens),
            "--temperature", String(temperature),
            "--json",
        ]
        if let image = imagePath {
            args += ["--image", image.path]
        }
        if let video = videoPath {
            args += ["--video", video.path]
        }

        let proc = Process()
        proc.executableURL = BundleResolver.pythonExecutable
        proc.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        proc.environment = env

        let out = Pipe()
        let err = Pipe()
        proc.standardOutput = out
        proc.standardError = err
        self.currentProcess = proc

        try proc.run()
        proc.waitUntilExit()

        let stdout = out.fileHandleForReading.readDataToEndOfFile()
        let stderr = err.fileHandleForReading.readDataToEndOfFile()
        let errorText = String(data: stderr, encoding: .utf8) ?? ""

        guard proc.terminationStatus == 0 else {
            throw InferenceError(message: errorText.isEmpty
                                 ? "inference exited with code \(proc.terminationStatus)"
                                 : errorText.trimmingCharacters(in: .whitespacesAndNewlines),
                                 code: proc.terminationStatus)
        }

        // jang_tools inference emits a SINGLE JSON line at the end. Strip anything before.
        let raw = String(data: stdout, encoding: .utf8) ?? ""
        guard let jsonLine = raw.split(whereSeparator: \.isNewline).last(where: { $0.hasPrefix("{") }) else {
            throw InferenceError(message: "inference output did not include a JSON line: \(raw.suffix(200))", code: -1)
        }
        do {
            let data = Data(jsonLine.utf8)
            // Check for error shape first
            if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let errorStr = obj["error"] as? String {
                throw InferenceError(message: errorStr, code: -1)
            }
            return try JSONDecoder().decode(InferenceResult.self, from: data)
        } catch let e as InferenceError {
            throw e
        } catch {
            throw InferenceError(message: "inference JSON decode failed: \(error)", code: -1)
        }
    }

    func cancel() {
        if let p = currentProcess, p.isRunning {
            p.terminate()
        }
    }
}
