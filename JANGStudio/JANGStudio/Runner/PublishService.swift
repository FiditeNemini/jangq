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

        // SECURITY: The token is passed via an environment variable (HF_HUB_TOKEN)
        // rather than argv. A 200 GB publish can take 30+ minutes; a command-line
        // token is visible to any user running `ps aux` (or macOS Activity Monitor)
        // for the whole window. Env vars are only visible to the process itself
        // and to root. Related audit item: M41.
        var args: [String] = [
            "-m", "jang_tools", "publish",
            "--model", modelPath.path,
            "--repo", repo,
            "--json",
        ]
        if isPrivate { args.append("--private") }
        if isDryRun { args.append("--dry-run") }

        let data = try await invoke(args: args, token: token)
        do {
            return try JSONDecoder().decode(PublishResult.self, from: data)
        } catch {
            throw PublishServiceError.decodeError("\(error)")
        }
    }

    private nonisolated static func invoke(args: [String], token: String) async throws -> Data {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global().async {
                do {
                    let proc = Process()
                    proc.executableURL = BundleResolver.pythonExecutable
                    proc.arguments = args
                    var env = ProcessInfo.processInfo.environment
                    env["HF_HUB_TOKEN"] = token
                    env["PYTHONUNBUFFERED"] = "1"
                    proc.environment = env
                    let out = Pipe(); let err = Pipe()
                    proc.standardOutput = out
                    proc.standardError = err
                    try proc.run()
                    proc.waitUntilExit()
                    if proc.terminationStatus != 0 {
                        let stderrData = err.fileHandleForReading.readDataToEndOfFile()
                        let stderrRaw = String(data: stderrData, encoding: .utf8) ?? ""
                        // Extra belt-and-suspenders: if the Python side ever leaks
                        // the token into an error message, scrub it before surfacing.
                        let stderr = stderrRaw.replacingOccurrences(of: token, with: "<redacted>")
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
