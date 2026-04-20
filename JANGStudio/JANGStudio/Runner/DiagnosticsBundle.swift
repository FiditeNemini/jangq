// JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift
import Foundation

@MainActor
enum DiagnosticsBundle {
    /// Regex patterns whose matches get redacted from logs/events before the
    /// bundle is written to disk. Each entry is `(description, regex)`. The
    /// match's substring is replaced with `<redacted>`.
    /// Introduced iter 14 (M22e / M16) — a failed convert or publish could
    /// land an HF token in stderr when the huggingface_hub client raises
    /// HfHubHTTPError. iter 6 scrubs it at the publish-error call site but
    /// anything older / another entry point could still leak.
    nonisolated static let sensitivePatterns: [(description: String, pattern: String, template: String)] = [
        ("HF token (hf_…)", #"hf_[A-Za-z0-9_-]{20,}"#, "<redacted>"),
        ("HF token (huggingface_…)", #"huggingface_[A-Za-z0-9_-]{20,}"#, "<redacted>"),
        // `Authorization: Bearer …` is what HTTPX logs on debug.
        ("Authorization Bearer", #"(?i)authorization:\s*bearer\s+[A-Za-z0-9_.-]{20,}"#, "<redacted>"),
        // Generic Bearer fallback for other HTTP clients.
        ("Bearer <token>", #"(?i)\bbearer\s+[A-Za-z0-9_.-]{20,}"#, "<redacted>"),
        // M196 (iter 133): cross-language parity with jang-server
        // iter-130 M193 `redact_for_log`. Before M196 the Swift
        // diagnostics scrubber covered HF tokens + Bearer but missed
        // three leak shapes the Python side catches:
        //   1. OpenAI keys (`sk-…`, `sk-proj-…`) — JANGStudio can end
        //      up with these if a user's test-inference workflow
        //      contacts an OpenAI-compatible endpoint for verification.
        //   2. Slack / Discord webhook URLs — users who wire a
        //      webhook_url into a jang-server run and then attach a
        //      diag bundle to a bug report leak the webhook secret
        //      without these.
        //   3. Query-string secrets (?api_key=, ?token=, ?access_token=,
        //      ?auth=) — any HTTP client error logged by the
        //      subprocess can echo the full URL with the secret.
        //
        // For the last two shapes, partial replacement preserves the
        // non-secret prefix for diagnostic context (the webhook host
        // and the query-param NAME stay so the operator can see WHERE
        // the secret was, just not WHAT it was). Uses capture-group
        // templates ($1<redacted>) — NSRegularExpression supports $1
        // substitution via stringByReplacingMatches's withTemplate.
        ("OpenAI key (sk-…)", #"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"#, "<redacted>"),
        ("Slack webhook secret", #"(/services/T[A-Z0-9]+/B[A-Z0-9]+/)([A-Za-z0-9]{8,})"#, "$1<redacted>"),
        ("Discord webhook secret", #"(/api/webhooks/[0-9]+/)([A-Za-z0-9_-]{16,})"#, "$1<redacted>"),
        ("URL query secret", #"([?&](?:api_key|token|access_token|auth)=)([^&\s"'<>]+)"#, "$1<redacted>"),
    ]

    /// Apply every pattern in `sensitivePatterns` to the input. Public so
    /// tests can pin the exact redaction behaviour.
    nonisolated static func scrubSensitive(_ text: String) -> String {
        var out = text
        for (_, pattern, template) in sensitivePatterns {
            guard let rx = try? NSRegularExpression(pattern: pattern, options: []) else { continue }
            let range = NSRange(out.startIndex..<out.endIndex, in: out)
            out = rx.stringByReplacingMatches(in: out, options: [], range: range,
                                              withTemplate: template)
        }
        return out
    }

    /// Anonymize file paths by replacing them with their basename. Drops
    /// directory structure from the diagnostics so a bug report doesn't
    /// reveal filesystem layout. Wired to `AppSettings.anonymizePathsInDiagnostics`.
    /// Only affects top-level string fields (sourceURL, outputURL) — logs /
    /// events go through `scrubSensitive` only.
    nonisolated static func anonymizePath(_ path: String) -> String {
        (path as NSString).lastPathComponent
    }

    /// Writes plan.json, run.log, events.jsonl, system.json, verify.json into a
    /// temp directory, then zips via `ditto -c -k`. Returns the final zip URL.
    ///
    /// M22e / M16: logs + events are passed through `scrubSensitive` BEFORE
    /// being written to disk — scrubbing post-zip is too late.
    /// M62-anonymize: when `anonymizePaths` is true, plan.sourceURL/outputURL
    /// are replaced with their basenames in the serialised plan.json.
    static func write(plan: ConversionPlan,
                      logLines: [String],
                      eventLines: [String],
                      verify: [VerifyCheck],
                      to desktop: URL,
                      anonymizePaths: Bool = false) throws -> URL {
        // Millisecond precision in the timestamp — M22d: a second click within
        // the same wall-clock second used to land in the same workDir and
        // reuse stale files. Each click now gets a unique directory.
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let stamp = fmt.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("JANGStudio-diag-\(stamp)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)

        // Serialise the plan with optional path anonymization. We encode into
        // a dictionary instead of Codable output so we can selectively rewrite
        // the path fields without touching the rest of the schema.
        var planDict: [String: Any] = [
            "profile": plan.profile,
            "family": plan.family.rawValue,
            "method": plan.method.rawValue,
            "hadamard": plan.hadamard,
            "run": plan.run.rawValue,
        ]
        if let src = plan.sourceURL?.path {
            planDict["sourceURL"] = anonymizePaths ? anonymizePath(src) : src
        }
        if let out = plan.outputURL?.path {
            planDict["outputURL"] = anonymizePaths ? anonymizePath(out) : out
        }
        try JSONSerialization.data(withJSONObject: planDict).write(
            to: workDir.appendingPathComponent("plan.json"))

        let scrubbedLogs = logLines.map(Self.scrubSensitive).joined(separator: "\n")
        try scrubbedLogs.write(to: workDir.appendingPathComponent("run.log"),
                               atomically: true, encoding: .utf8)
        let scrubbedEvents = eventLines.map(Self.scrubSensitive).joined(separator: "\n")
        try scrubbedEvents.write(to: workDir.appendingPathComponent("events.jsonl"),
                                 atomically: true, encoding: .utf8)
        let sys: [String: String] = [
            "macos": ProcessInfo.processInfo.operatingSystemVersionString,
            "ram_bytes": String(ProcessInfo.processInfo.physicalMemory),
            "app_version": (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "?",
        ]
        try JSONSerialization.data(withJSONObject: sys).write(to: workDir.appendingPathComponent("system.json"))
        let verifyData = verify.map { ["id": $0.id.rawValue, "status": $0.status.rawValue, "required": $0.required] as [String: Any] }
        try JSONSerialization.data(withJSONObject: verifyData).write(to: workDir.appendingPathComponent("verify.json"))

        let zipURL = desktop.appendingPathComponent("JANGStudio-diagnostics-\(stamp).zip")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-c", "-k", "--keepParent", workDir.path, zipURL.path]
        try p.run(); p.waitUntilExit()
        try? FileManager.default.removeItem(at: workDir)
        return zipURL
    }

    /// Async variant of `write` — offloads the `ditto` subprocess off MainActor
    /// so the "Copy Diagnostics" button doesn't beach-ball the UI during a
    /// multi-second zip of a large bundle (tens of MB of logs + events).
    ///
    /// M106 (iter 42): the synchronous `write` above was called on MainActor
    /// from RunStep's button handler. For a small bundle (<5 MB) the
    /// `ditto -c -k` runs in ~1s and is invisible, but a convert that
    /// produced thousands of JSONL tick events + long stderr could easily
    /// produce a 50 MB bundle — several seconds of frozen UI with nothing
    /// but a dead button to look at.
    ///
    /// Strategy: scrub + write the tempdir files on MainActor (fast, small
    /// JSONs), then hop to a DispatchQueue for the `ditto` subprocess wait
    /// via `withCheckedThrowingContinuation`. Same cancel-propagation
    /// wrapper pattern as iter-33's service-sweep (ProcessHandle from iter 30).
    static func writeAsync(plan: ConversionPlan,
                           logLines: [String],
                           eventLines: [String],
                           verify: [VerifyCheck],
                           to desktop: URL,
                           anonymizePaths: Bool = false) async throws -> URL {
        // Step 1: build tempdir + write scrubbed logs/events/etc. ON MainActor.
        // These writes are small (typically <1 MB total before the log tail)
        // and fast; doing them here avoids needing to make all the MainActor
        // @State reads Sendable via an isolated snapshot.
        let fmt = ISO8601DateFormatter()
        fmt.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let stamp = fmt.string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("JANGStudio-diag-\(stamp)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)

        var planDict: [String: Any] = [
            "profile": plan.profile,
            "family": plan.family.rawValue,
            "method": plan.method.rawValue,
            "hadamard": plan.hadamard,
            "run": plan.run.rawValue,
        ]
        if let src = plan.sourceURL?.path {
            planDict["sourceURL"] = anonymizePaths ? anonymizePath(src) : src
        }
        if let out = plan.outputURL?.path {
            planDict["outputURL"] = anonymizePaths ? anonymizePath(out) : out
        }
        try JSONSerialization.data(withJSONObject: planDict).write(
            to: workDir.appendingPathComponent("plan.json"))

        let scrubbedLogs = logLines.map(Self.scrubSensitive).joined(separator: "\n")
        try scrubbedLogs.write(to: workDir.appendingPathComponent("run.log"),
                               atomically: true, encoding: .utf8)
        let scrubbedEvents = eventLines.map(Self.scrubSensitive).joined(separator: "\n")
        try scrubbedEvents.write(to: workDir.appendingPathComponent("events.jsonl"),
                                 atomically: true, encoding: .utf8)
        let sys: [String: String] = [
            "macos": ProcessInfo.processInfo.operatingSystemVersionString,
            "ram_bytes": String(ProcessInfo.processInfo.physicalMemory),
            "app_version": (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "?",
        ]
        try JSONSerialization.data(withJSONObject: sys).write(to: workDir.appendingPathComponent("system.json"))
        let verifyData = verify.map { ["id": $0.id.rawValue, "status": $0.status.rawValue, "required": $0.required] as [String: Any] }
        try JSONSerialization.data(withJSONObject: verifyData).write(to: workDir.appendingPathComponent("verify.json"))

        let zipURL = desktop.appendingPathComponent("JANGStudio-diagnostics-\(stamp).zip")

        // Step 2: hop off MainActor for the ditto subprocess. Task-cancel
        // propagation via withTaskCancellationHandler (iter 33 pattern).
        let handle = ProcessHandle()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                DispatchQueue.global().async {
                    do {
                        let p = Process()
                        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
                        p.arguments = ["-c", "-k", "--keepParent", workDir.path, zipURL.path]
                        try p.run()
                        handle.set(process: p)
                        p.waitUntilExit()
                        if p.terminationStatus != 0 {
                            cont.resume(throwing: NSError(
                                domain: "DiagnosticsBundle.writeAsync",
                                code: Int(p.terminationStatus),
                                userInfo: [NSLocalizedDescriptionKey: "ditto exited \(p.terminationStatus)"]))
                            return
                        }
                        cont.resume()
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            handle.cancel()
        }
        try? FileManager.default.removeItem(at: workDir)
        return zipURL
    }
}
