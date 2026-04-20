import Foundation

struct PublishResult: Codable, Equatable {
    let dryRun: Bool
    let repo: String
    let url: String?
    /// URL of the specific commit created by this upload, e.g.
    /// `https://huggingface.co/foo/bar/commit/abc123`. Present only on successful
    /// non-dry-run publishes. Decoded in iter 7 to close M44 — the commit link
    /// was being emitted by Python but silently dropped on the Swift side, so
    /// the UI only showed the repo URL and the user had no confirmation their
    /// upload landed until they manually navigated there.
    let commitUrl: String?
    let filesCount: Int?
    let totalSizeBytes: Int?

    enum CodingKeys: String, CodingKey {
        case dryRun = "dry_run"
        case repo, url
        case commitUrl = "commit_url"
        case filesCount = "files_count"
        case totalSizeBytes = "total_size_bytes"
    }
}

/// Validate an HF-style `org/name` repo identifier before we dispatch a 30-minute
/// upload. HF requires the slash, disallows spaces, and enforces a char class.
/// Returns a user-facing error string if invalid, nil if valid.
///
/// Rule summary (from huggingface_hub validation):
/// - Must match `<org>/<name>`
/// - org and name each start with alphanumeric
/// - subsequent chars: alphanumeric, `_`, `.`, `-`
/// - each segment max 96 chars
/// - no spaces, no double-slash
@MainActor
enum HFRepoValidator {
    static func validationError(_ repo: String) -> String? {
        let trimmed = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            return "Repo id is empty."
        }
        if trimmed.contains(" ") {
            return "Repo id cannot contain spaces."
        }
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        if parts.count != 2 {
            return "Repo id must be in the format 'org/model-name' (one forward slash)."
        }
        let org = String(parts[0])
        let name = String(parts[1])
        if org.isEmpty || name.isEmpty {
            return "Repo id must have both org and name: 'org/model-name'."
        }
        let segmentRegex = try? NSRegularExpression(
            pattern: "^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$", options: [])
        for (label, segment) in [("org", org), ("name", name)] {
            let range = NSRange(segment.startIndex..<segment.endIndex, in: segment)
            if segmentRegex?.firstMatch(in: segment, options: [], range: range) == nil {
                return "Invalid \(label) segment '\(segment)': start with a letter/digit, then letters/digits/._- up to 96 chars."
            }
        }
        return nil
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
                    // M62 env-passthrough for PYTHONPATH / thread count /
                    // throttle. Publish is a one-shot Python invoke just like
                    // PythonRunner, so it benefits from the same user settings.
                    for (k, v) in BundleResolver.childProcessEnvAdditions(inherited: env) {
                        env[k] = v
                    }
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
