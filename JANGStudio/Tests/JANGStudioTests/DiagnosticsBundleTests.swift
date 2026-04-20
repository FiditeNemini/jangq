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
}
