import Foundation

struct PublishResult: Codable, Equatable {
    let dryRun: Bool
    let repo: String
    let url: String?
    let filesCount: Int?
    let totalSizeBytes: Int?

    enum CodingKeys: String, CodingKey {
        case dryRun = "dry_run"
        case repo, url
        case filesCount = "files_count"
        case totalSizeBytes = "total_size_bytes"
    }
}

enum PublishServiceError: Error, LocalizedError {
    case missingToken
    case cliError(code: Int32, stderr: String)
    case decodeError(String)

    var errorDescription: String? {
        switch self {
        case .missingToken:
            return "HuggingFace token missing — set HF_HUB_TOKEN env var or paste a token in Settings."
        case .cliError(let c, let s):
            return "jang-tools publish exited \(c): \(s.trimmingCharacters(in: .whitespacesAndNewlines))"
        case .decodeError(let s): return s
        }
    }
}

@MainActor
enum PublishService {
    /// Dry-run the publish to preview file count + total size before committing.
    static func dryRun(modelPath: URL, repo: String, isPrivate: Bool, token: String) async throws -> PublishResult {
        try await _invoke(
            modelPath: modelPath, repo: repo, isPrivate: isPrivate, token: token, isDryRun: true)
    }

    /// Actually push to HF. Blocks until upload completes.
    static func publish(modelPath: URL, repo: String, isPrivate: Bool, token: String) async throws -> PublishResult {
        try await _invoke(
            modelPath: modelPath, repo: repo, isPrivate: isPrivate, token: token, isDryRun: false)
    }

    private static func _invoke(modelPath: URL, repo: String, isPrivate: Bool, token: String, isDryRun: Bool) async throws -> PublishResult {
        guard !token.isEmpty else { throw PublishServiceError.missingToken }

        var args: [String] = [
            "-m", "jang_tools", "publish",
            "--model", modelPath.path,
            "--repo", repo,
            "--token", token,
            "--json",
        ]
        if isPrivate { args.append("--private") }
        if isDryRun { args.append("--dry-run") }

        let data = try await invoke(args: args)
        do {
            return try JSONDecoder().decode(PublishResult.self, from: data)
        } catch {
            throw PublishServiceError.decodeError("\(error)")
        }
    }

    private nonisolated static func invoke(args: [String]) async throws -> Data {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global().async {
                do {
                    let proc = Process()
                    proc.executableURL = BundleResolver.pythonExecutable
                    proc.arguments = args
                    let out = Pipe(); let err = Pipe()
                    proc.standardOutput = out
                    proc.standardError = err
                    try proc.run()
                    proc.waitUntilExit()
                    if proc.terminationStatus != 0 {
                        let stderr = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                        cont.resume(throwing: PublishServiceError.cliError(code: proc.terminationStatus, stderr: stderr))
                        return
                    }
                    cont.resume(returning: out.fileHandleForReading.readDataToEndOfFile())
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}
