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
            // M164 (iter 87): huggingface_hub.validate_repo_id also forbids
            // consecutive `..` / `--` and trailing `.` / `-`. Pre-M164 these
            // passed the client validator but HF rejected them at upload
            // time — defeating M48's whole purpose (fail-fast before the
            // 30-minute publish dispatch starts). Most common real-world
            // triggers: auto-complete dropping a trailing `.`, or a stray
            // trailing `-` from model-name templating.
            if segment.hasSuffix(".") || segment.hasSuffix("-") {
                return "Invalid \(label) segment '\(segment)': cannot end with '.' or '-'."
            }
            if segment.contains("..") || segment.contains("--") {
                return "Invalid \(label) segment '\(segment)': cannot contain consecutive '..' or '--'."
            }
        }
        return nil
    }
}

/// Sendable holder for the publish subprocess's `Process` reference so the
/// `AsyncThrowingStream.onTermination` callback (Sendable closure) can SIGTERM
/// the child when the consumer cancels. Final + internally locked so writes
/// from the producer task and reads from the onTermination callback are safe.
/// Introduced iter 30 to close M96.
final class ProcessHandle: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var _wasCancelled: Bool = false

    /// True if `cancel()` was called at any point. Used by _streamPublish's
    /// exit branch to distinguish user-cancel (clean finish) from a real
    /// subprocess failure (throw).
    var wasCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return _wasCancelled
    }

    func set(process: Process) {
        lock.lock(); defer { lock.unlock() }
        // If cancel() already fired before run() landed (race), terminate
        // immediately. Otherwise just store the reference.
        if _wasCancelled {
            if process.isRunning { process.terminate() }
        }
        self.process = process
    }

    func cancel() {
        lock.lock()
        _wasCancelled = true
        let proc = process
        lock.unlock()
        // SIGTERM the subprocess. PythonRunner pattern: escalate to SIGKILL
        // after 3 seconds if the process ignores SIGTERM.
        if let proc, proc.isRunning {
            proc.terminate()
            Task.detached {
                try? await Task.sleep(for: .seconds(3))
                if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
            }
        }
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

    /// Streaming publish with per-file progress.
    ///
    /// Yields `ProgressEvent`s from the Python side's JSONL protocol (same
    /// schema as convert's 5-phase stream; see iter 23's Python half of M43).
    /// The stream completes when the subprocess exits. On success the final
    /// PublishResult is available via `finalResult(...)` — it's ALSO emitted
    /// as stdout JSON by the Python side and parsed at stream end.
    ///
    /// Usage: UIs iterate the stream for progress events (phase / tick /
    /// info / warn) and read the returned `result` once the stream completes.
    /// Non-streaming callers should continue to use `publish(...)` which is
    /// faster (bulk upload_folder, no per-file commits).
    static func publishWithProgress(modelPath: URL, repo: String, isPrivate: Bool,
                                    token: String) -> AsyncThrowingStream<ProgressEvent, Error> {
        AsyncThrowingStream { continuation in
            // M96 (iter 30): hold a box for the child Process so
            // continuation.onTermination can terminate it when the consuming
            // Task cancels. Without this, cancelling the stream (or dismissing
            // the Publish sheet) left the Python subprocess running and
            // uploading to HuggingFace in the background — stranding the
            // half-published repo with partial files. `ProcessHandle` is a
            // simple Sendable class so we can reference it across threads.
            let handle = ProcessHandle()
            continuation.onTermination = { _ in
                handle.cancel()
            }
            Task.detached {
                await Self._streamPublish(modelPath: modelPath, repo: repo,
                                          isPrivate: isPrivate, token: token,
                                          continuation: continuation,
                                          handle: handle)
            }
        }
    }

    private nonisolated static func _streamPublish(
        modelPath: URL, repo: String, isPrivate: Bool, token: String,
        continuation: AsyncThrowingStream<ProgressEvent, Error>.Continuation,
        handle: ProcessHandle
    ) async {
        guard !token.isEmpty else {
            continuation.finish(throwing: PublishServiceError.missingToken)
            return
        }

        var args: [String] = [
            "-m", "jang_tools", "publish",
            "--model", modelPath.path,
            "--repo", repo,
            "--json",
            "--progress", "json",   // iter 23: Python emits JSONL to stderr
        ]
        if isPrivate { args.append("--private") }

        let proc = Process()
        proc.executableURL = BundleResolver.pythonExecutable
        proc.arguments = args
        var env = ProcessInfo.processInfo.environment
        env["HF_HUB_TOKEN"] = token
        env["PYTHONUNBUFFERED"] = "1"
        for (k, v) in BundleResolver.childProcessEnvAdditions(inherited: env) {
            env[k] = v
        }
        proc.environment = env

        let errPipe = Pipe()
        // M159 (iter 82): publishWithProgress yields ProgressEvents from
        // stderr only; stdout is never consumed by this streaming path (the
        // non-streaming `publish()` is the one that reads the final JSON
        // result, via a separate subprocess invocation through
        // PythonCLIInvoker). The old code wired `outPipe = Pipe()` to
        // stdout but never read it — a latent deadlock if the Python side
        // ever emitted >64 KB on stdout (even an unexpected `print()` in a
        // dependency like huggingface_hub would do it). Swapped to
        // `FileHandle.nullDevice` so the kernel discards stdout without
        // ever filling a buffer we'd have to drain.
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = errPipe

        // Drain stderr for progress events (JSONL schema matches convert).
        let stderrTask = Task.detached {
            let parser = JSONLProgressParser()
            var lastErrTail = ""
            for try await line in errPipe.fileHandleForReading.bytes.lines {
                lastErrTail = String(line.suffix(256))
                if let ev = parser.parse(line: line) {
                    continuation.yield(ev)
                }
            }
            return lastErrTail
        }

        do {
            try proc.run()
        } catch {
            continuation.finish(throwing: error)
            return
        }

        // Register the running process so the onTermination handler can
        // reach it if the consumer cancels.
        handle.set(process: proc)

        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            proc.terminationHandler = { _ in done.resume() }
        }

        let lastErrTail = (try? await stderrTask.value) ?? ""

        if proc.terminationStatus == 0 {
            continuation.finish()
        } else if handle.wasCancelled {
            // User-initiated cancel — finish cleanly, not as an error. UI
            // treats a cancelled publish as "user dismissed", not "upload
            // failed". Any files already written to the HF repo are
            // documented as M97 (partial-repo cleanup follow-up).
            continuation.finish()
        } else {
            // Scrub token before surfacing stderr to UI (iter 6 M41 layer-2).
            let scrubbed = lastErrTail.replacingOccurrences(of: token, with: "<redacted>")
            continuation.finish(
                throwing: PublishServiceError.cliError(code: proc.terminationStatus, stderr: scrubbed))
        }
    }

    private nonisolated static func invoke(args: [String], token: String) async throws -> Data {
        // M156 (iter 79): migrated to shared PythonCLIInvoker (iter-76
        // M153 / iter-78 M155 pattern). 7th copy of the cross-layer cancel
        // pattern was hiding here — iter-76 scoped to 5 services; iter-78
        // caught SourceDetector; this iter catches dryRun's variant which
        // also needed env-var threading for HF_HUB_TOKEN.
        //
        // The errorFactory closure captures `token` so it can redact it
        // from the stderr before wrapping the typed error — keeps the
        // security-critical sanitization at the call site where the
        // token is in scope.
        var env = ProcessInfo.processInfo.environment
        env["HF_HUB_TOKEN"] = token
        env["PYTHONUNBUFFERED"] = "1"
        for (k, v) in BundleResolver.childProcessEnvAdditions(inherited: env) {
            env[k] = v
        }
        return try await PythonCLIInvoker.invoke(args: args, env: env) { code, stderr in
            let scrubbed = stderr.replacingOccurrences(of: token, with: "<redacted>")
            return PublishServiceError.cliError(code: code, stderr: scrubbed)
        }
    }
}
