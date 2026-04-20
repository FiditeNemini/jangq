// JANGStudio/Tests/JANGStudioTests/DiagnosticsBundleTests.swift
import XCTest
@testable import JANGStudio

final class DiagnosticsBundleTests: XCTestCase {
    @MainActor
    func test_writesZipWithExpectedEntries() throws {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/tmp/src")
        plan.outputURL = URL(fileURLWithPath: "/tmp/out")
        plan.profile = "JANG_4K"
        let logs = ["[1/5] detect", "[2/5] calibrate"]
        let events = [#"{"v":1,"type":"phase","n":1,"total":5,"name":"detect","ts":1.0}"#]
        let url = try DiagnosticsBundle.write(plan: plan, logLines: logs, eventLines: events,
                                              verify: [], to: FileManager.default.temporaryDirectory)
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
        XCTAssertTrue(url.lastPathComponent.hasSuffix(".zip"))
    }

    // MARK: - Iter 14: M22e / M16 — token scrubbing

    func test_scrubs_hf_token() {
        let scrubbed = DiagnosticsBundle.scrubSensitive(
            "ERROR: auth failed token=hf_abcdefghijklmnopqrstuvwxyz1234567890")
        XCTAssertFalse(scrubbed.contains("hf_abcdefghijklmnopqrstuvwxyz1234567890"))
        XCTAssertTrue(scrubbed.contains("<redacted>"))
    }

    func test_scrubs_huggingface_token() {
        let scrubbed = DiagnosticsBundle.scrubSensitive(
            "token=huggingface_abcdef_ghij-klmnop1234567890QRSTUV")
        XCTAssertFalse(scrubbed.contains("huggingface_abcdef_ghij-klmnop1234567890QRSTUV"))
        XCTAssertTrue(scrubbed.contains("<redacted>"))
    }

    func test_scrubs_authorization_bearer_header() {
        let scrubbed = DiagnosticsBundle.scrubSensitive(
            "HTTP request headers: Authorization: Bearer abc123def456ghi789jkl012")
        XCTAssertFalse(scrubbed.contains("abc123def456ghi789jkl012"))
        XCTAssertTrue(scrubbed.contains("<redacted>"))
    }

    func test_scrubs_generic_bearer_token() {
        let scrubbed = DiagnosticsBundle.scrubSensitive(
            "POST /api Bearer xyz9876543210abcdefghijklmnop")
        XCTAssertFalse(scrubbed.contains("xyz9876543210abcdefghijklmnop"))
        XCTAssertTrue(scrubbed.contains("<redacted>"))
    }

    func test_scrub_preserves_non_sensitive_text() {
        let input = "plain log line with no secrets: /Users/eric/model.safetensors shard 3/5"
        XCTAssertEqual(DiagnosticsBundle.scrubSensitive(input), input)
    }

    func test_scrub_short_hf_lookalike_not_redacted() {
        // The regex requires ≥20 chars after `hf_`. A short `hf_x` in a log
        // (e.g. a variable name) should NOT be redacted.
        let scrubbed = DiagnosticsBundle.scrubSensitive("var hf_short = thing")
        XCTAssertEqual(scrubbed, "var hf_short = thing")
    }

    // MARK: - Iter 133 M196: cross-language parity with jang-server redact_for_log

    func test_scrubs_openai_key() {
        // sk- and sk-proj- formats. Both need to be redacted because a
        // user whose test-inference workflow contacts an OpenAI
        // compatible endpoint could end up with the key in the error
        // log of a failed call.
        let scrubbedShort = DiagnosticsBundle.scrubSensitive(
            "POST /v1/chat failed: key=sk-abcdefghijklmnop1234567")
        XCTAssertFalse(scrubbedShort.contains("sk-abcdefghijklmnop1234567"))
        XCTAssertTrue(scrubbedShort.contains("<redacted>"))

        let scrubbedProj = DiagnosticsBundle.scrubSensitive(
            "env key=sk-proj-abcdefghijklmnopqrstuvw")
        XCTAssertFalse(scrubbedProj.contains("sk-proj-abcdefghijklmnopqrstuvw"))
        XCTAssertTrue(scrubbedProj.contains("<redacted>"))
    }

    func test_scrubs_slack_webhook_but_keeps_host() {
        // Slack webhook URLs carry the write secret in the path. The
        // scrub must redact the secret but leave `hooks.slack.com` in
        // place so the operator reading a bug report can tell WHICH
        // service the webhook pointed at. Template is "$1<redacted>"
        // which keeps the `/services/T0/B0/` prefix.
        let input = "Webhook delivered to https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXX"
        let scrubbed = DiagnosticsBundle.scrubSensitive(input)
        XCTAssertFalse(scrubbed.contains("XXXXXXXXXXXXXXXX"),
                       "secret token body must be redacted")
        XCTAssertTrue(scrubbed.contains("hooks.slack.com"),
                      "host must remain for diagnostic context")
        XCTAssertTrue(scrubbed.contains("/services/T00000000/B00000000/"),
                      "prefix up to the secret must remain")
        XCTAssertTrue(scrubbed.contains("<redacted>"))
    }

    func test_scrubs_discord_webhook_but_keeps_host() {
        let input = "https://discord.com/api/webhooks/1234567890/secret-token-goes-here-abc"
        let scrubbed = DiagnosticsBundle.scrubSensitive(input)
        XCTAssertFalse(scrubbed.contains("secret-token-goes-here-abc"),
                       "Discord webhook secret must be redacted")
        XCTAssertTrue(scrubbed.contains("discord.com"))
        XCTAssertTrue(scrubbed.contains("/api/webhooks/1234567890/"),
                      "path up to the secret must remain")
    }

    func test_scrubs_url_query_secret_keeps_param_name() {
        // Partial replacement: the param NAME stays (so operators know
        // WHICH param held a secret) but the VALUE is redacted. Python's
        // redact_for_log does the same via m.group(1) + REDACTED. This
        // pins cross-language parity.
        let input = "GET https://api.example.com/path?api_key=SECRETVALUE123&other=foo"
        let scrubbed = DiagnosticsBundle.scrubSensitive(input)
        XCTAssertFalse(scrubbed.contains("SECRETVALUE123"),
                       "secret value must be redacted")
        XCTAssertTrue(scrubbed.contains("api_key=<redacted>"),
                      "param name must remain visible so the vector is clear")
        XCTAssertTrue(scrubbed.contains("&other=foo"),
                      "trailing URL components must be preserved")
    }

    func test_scrubs_url_query_secret_multiple_param_names() {
        // Parity with Python: api_key, token, access_token, auth all
        // covered.
        for param in ["api_key", "token", "access_token", "auth"] {
            let input = "GET https://h/?\(param)=LEAKLEAKLEAKLEAK&k=v"
            let scrubbed = DiagnosticsBundle.scrubSensitive(input)
            XCTAssertFalse(scrubbed.contains("LEAKLEAKLEAKLEAK"),
                           "secret for param=\(param) must be redacted")
            XCTAssertTrue(scrubbed.contains("\(param)=<redacted>"),
                          "param name \(param) must remain in redacted form")
        }
    }

    func test_scrub_preserves_non_secret_urls() {
        // Must NOT eat legitimate URLs that happen to contain slashes
        // + query strings but no secret-bearing param names.
        let input = "GET https://huggingface.co/Qwen/Qwen3-8B/resolve/main/config.json?download=true"
        let scrubbed = DiagnosticsBundle.scrubSensitive(input)
        XCTAssertEqual(scrubbed, input,
                       "non-secret URL must pass through unchanged")
    }

    // MARK: - Iter 88 M165: URL-embedded token edge cases

    func test_scrub_token_in_url_query_string() {
        // HF client retries can log the full request URL including a token
        // query parameter. `&`, `=`, `?` aren't in the token char class
        // `[A-Za-z0-9_.-]`, so the greedy match stops cleanly at the
        // separator — redacting the token but preserving surrounding URL
        // structure. Pins this behavior so a future regex tweak can't
        // accidentally eat the `&key=` tail.
        let input = "GET https://hf.co/api/models?token=hf_abcdefghijklmnopqrstuvwxyz1234&revision=main"
        let scrubbed = DiagnosticsBundle.scrubSensitive(input)
        XCTAssertFalse(scrubbed.contains("hf_abcdefghijklmnopqrstuvwxyz1234"),
            "token body must be redacted")
        XCTAssertTrue(scrubbed.contains("&revision=main"),
            "URL tail after the token must be preserved — the scrub should not eat past the token's char class")
        XCTAssertTrue(scrubbed.contains("<redacted>"),
            "replacement marker must appear in place of the token")
    }

    func test_scrub_token_adjacent_to_json_delimiter() {
        // Tokens in JSON debug output: "token":"hf_…". The closing `"`
        // isn't in the class so match stops there. Guards against any
        // future "swallow the JSON delimiter" regex bug.
        let input = #"{"token":"hf_abcdefghijklmnopqrstuvwxyz1234","model":"qwen"}"#
        let scrubbed = DiagnosticsBundle.scrubSensitive(input)
        XCTAssertFalse(scrubbed.contains("hf_abcdefghijklmnopqrstuvwxyz1234"))
        XCTAssertTrue(scrubbed.contains(#""model":"qwen""#),
            "JSON fields after the redacted token must remain intact")
    }

    @MainActor
    func test_write_scrubs_tokens_in_log_and_events() throws {
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        let secret = "hf_SECRETTOKENabcdef1234567890XYZ"
        let logs = ["[error] publish failed: token=\(secret)"]
        let events = [#"{"v":1,"type":"error","msg":"auth=Bearer \#(secret)","ts":1.0}"#]
        let zipURL = try DiagnosticsBundle.write(plan: plan, logLines: logs, eventLines: events,
                                                 verify: [], to: FileManager.default.temporaryDirectory)
        // Unzip and inspect
        let unzipDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("diag-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: unzipDir, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-x", "-k", zipURL.path, unzipDir.path]
        try p.run(); p.waitUntilExit()
        // walk unzipped tree looking for the secret
        let enumerator = FileManager.default.enumerator(at: unzipDir, includingPropertiesForKeys: nil)!
        var saw = false
        for case let url as URL in enumerator {
            if url.pathExtension == "log" || url.pathExtension == "jsonl" {
                let content = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
                if content.contains(secret) { saw = true }
            }
        }
        XCTAssertFalse(saw, "secret token must NOT appear anywhere in the zipped bundle")
    }

    // MARK: - M62-anonymize — path redaction flag

    @MainActor
    func test_anonymize_paths_replaces_source_output_with_basename() throws {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/Users/eric/secrets/MyModel")
        plan.outputURL = URL(fileURLWithPath: "/Volumes/WorkDisk/output/MyModel-JANG_4K")
        plan.profile = "JANG_4K"
        let zipURL = try DiagnosticsBundle.write(plan: plan, logLines: [], eventLines: [],
                                                 verify: [], to: FileManager.default.temporaryDirectory,
                                                 anonymizePaths: true)
        let unzipDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("diag-anon-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: unzipDir, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-x", "-k", zipURL.path, unzipDir.path]
        try p.run(); p.waitUntilExit()
        // Find the plan.json inside
        let enumerator = FileManager.default.enumerator(at: unzipDir, includingPropertiesForKeys: nil)!
        var planJSON: String?
        for case let url as URL in enumerator {
            if url.lastPathComponent == "plan.json" {
                planJSON = try? String(contentsOf: url, encoding: .utf8)
            }
        }
        XCTAssertNotNil(planJSON)
        // Paths must be basenames only — full dir chain must be stripped.
        XCTAssertFalse(planJSON!.contains("/Users/eric/secrets"),
                       "anonymized plan.json must not contain source dir chain")
        XCTAssertFalse(planJSON!.contains("/Volumes/WorkDisk/output"),
                       "anonymized plan.json must not contain output dir chain")
        XCTAssertTrue(planJSON!.contains("MyModel"), "basename must remain")
    }

    @MainActor
    func test_anonymize_paths_disabled_preserves_full_path() throws {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/Users/eric/secrets/MyModel")
        plan.profile = "JANG_4K"
        let zipURL = try DiagnosticsBundle.write(plan: plan, logLines: [], eventLines: [],
                                                 verify: [], to: FileManager.default.temporaryDirectory,
                                                 anonymizePaths: false)
        let unzipDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("diag-full-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: unzipDir, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-x", "-k", zipURL.path, unzipDir.path]
        try p.run(); p.waitUntilExit()
        let enumerator = FileManager.default.enumerator(at: unzipDir, includingPropertiesForKeys: nil)!
        var planData: Data?
        for case let url as URL in enumerator {
            if url.lastPathComponent == "plan.json" {
                planData = try? Data(contentsOf: url)
            }
        }
        XCTAssertNotNil(planData)
        // Parse JSON and check the field directly (JSONSerialization escapes
        // slashes as `\/` in the raw bytes, so substring-match fails).
        let obj = try JSONSerialization.jsonObject(with: planData!) as? [String: Any]
        XCTAssertEqual(obj?["sourceURL"] as? String, "/Users/eric/secrets/MyModel",
                       "non-anonymized plan.json must retain full source path")
    }

    // MARK: - M22d: unique workDir per click

    @MainActor
    func test_back_to_back_writes_get_unique_filenames() throws {
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        let dest = FileManager.default.temporaryDirectory
        let a = try DiagnosticsBundle.write(plan: plan, logLines: ["a"], eventLines: [],
                                            verify: [], to: dest)
        let b = try DiagnosticsBundle.write(plan: plan, logLines: ["b"], eventLines: [],
                                            verify: [], to: dest)
        // Millisecond precision → back-to-back writes land in different files.
        XCTAssertNotEqual(a.lastPathComponent, b.lastPathComponent,
                          "two writes within a second must produce distinct zip filenames")
    }

    // MARK: - Iter 42: M106 writeAsync doesn't block MainActor on ditto

    @MainActor
    func test_writeAsync_produces_same_zip_shape_as_sync() async throws {
        // The async variant must produce an identical-shaped zip (same
        // entries: plan.json, run.log, events.jsonl, system.json, verify.json
        // plus the outer zip filename pattern). This pins feature parity so
        // the async path can't silently regress vs the sync path that iter
        // 14's M22 tests cover.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        let dest = FileManager.default.temporaryDirectory
        let zipURL = try await DiagnosticsBundle.writeAsync(
            plan: plan, logLines: ["[hi]"], eventLines: [#"{"v":1,"type":"info"}"#],
            verify: [], to: dest)
        XCTAssertTrue(FileManager.default.fileExists(atPath: zipURL.path))
        XCTAssertTrue(zipURL.lastPathComponent.hasSuffix(".zip"))
        XCTAssertTrue(zipURL.lastPathComponent.hasPrefix("JANGStudio-diagnostics-"))

        // Unzip and verify expected entries
        let unzipDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("diag-async-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: unzipDir, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-x", "-k", zipURL.path, unzipDir.path]
        try p.run(); p.waitUntilExit()
        // enumerator().makeIterator() is unavailable from async contexts on
        // Swift 6; use contentsOfDirectory recursively via a small helper.
        let foundEntries = Self.allFilenames(under: unzipDir)
        for required in ["plan.json", "run.log", "events.jsonl", "system.json", "verify.json"] {
            XCTAssertTrue(foundEntries.contains(required),
                          "async variant missing expected entry: \(required)")
        }
    }

    /// Recursive filename collection that doesn't rely on FileManager.enumerator
    /// (which can't be iterated from async contexts in Swift 6).
    private static func allFilenames(under root: URL) -> Set<String> {
        var out: Set<String> = []
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey]
        ) else { return out }
        for entry in entries {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: entry.path, isDirectory: &isDir), isDir.boolValue {
                out.formUnion(allFilenames(under: entry))
            } else {
                out.insert(entry.lastPathComponent)
            }
        }
        return out
    }

    @MainActor
    func test_writeAsync_scrubs_sensitive_like_sync() async throws {
        // M22e boundary — scrubSensitive must apply to BOTH sync AND async paths.
        let plan = ConversionPlan()
        plan.profile = "JANG_4K"
        let secret = "hf_SECRETTOKENabc1234567890XYZ"
        let logs = ["[error] token=\(secret)"]
        let zipURL = try await DiagnosticsBundle.writeAsync(
            plan: plan, logLines: logs, eventLines: [],
            verify: [], to: FileManager.default.temporaryDirectory)

        let unzipDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("diag-async-scrub-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: unzipDir, withIntermediateDirectories: true)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-x", "-k", zipURL.path, unzipDir.path]
        try p.run(); p.waitUntilExit()
        // Iterate files via recursive helper (enumerator unavailable in async).
        for url in Self.allFileURLs(under: unzipDir) where url.pathExtension == "log" {
            let content = try String(contentsOf: url, encoding: .utf8)
            XCTAssertFalse(content.contains(secret),
                           "writeAsync must scrub HF tokens (M22e pattern)")
        }
    }

    private static func allFileURLs(under root: URL) -> [URL] {
        var out: [URL] = []
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey]
        ) else { return out }
        for entry in entries {
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: entry.path, isDirectory: &isDir), isDir.boolValue {
                out.append(contentsOf: allFileURLs(under: entry))
            } else {
                out.append(entry)
            }
        }
        return out
    }
}
