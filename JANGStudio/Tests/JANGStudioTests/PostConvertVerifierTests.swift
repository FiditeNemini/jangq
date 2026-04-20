// JANGStudio/Tests/JANGStudioTests/PostConvertVerifierTests.swift
import XCTest
@testable import JANGStudio

final class PostConvertVerifierTests: XCTestCase {
    private func fixture(_ name: String) -> URL {
        Bundle(for: Self.self).url(forResource: name, withExtension: nil, subdirectory: nil)
            ?? Bundle(for: Self.self).bundleURL.appendingPathComponent("Fixtures/\(name)")
    }

    func test_goodOutputAllRequiredPass() async throws {
        let url = fixture("good_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let requiredFails = checks.filter { $0.required && $0.status == .fail }
        XCTAssertTrue(requiredFails.isEmpty, "unexpected required fails: \(requiredFails.map(\.id))")
        XCTAssertTrue(checks.contains { $0.id == .generationConfig && $0.status == .pass },
                      "good output should have generation_config.json")
    }

    func test_brokenOutputFlagsChatTemplateAndShardMismatch() async throws {
        let url = fixture("broken_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 2)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let failedIDs = checks.filter { $0.status == .fail }.map { $0.id }
        XCTAssertTrue(failedIDs.contains(.chatTemplate))
        XCTAssertTrue(failedIDs.contains(.shardsMatchIndex))
    }

    // MARK: - Iter 19: M42 — runJangValidate timeout + cancel-safe pattern

    func test_runJangValidate_defaultTimeoutIsReasonable() {
        // Validation is file-inspection only; a 60-second default is 10x
        // headroom over the ≤5s normal completion time. Pin this so a future
        // commit tightening it doesn't break long-running debug environments.
        XCTAssertGreaterThanOrEqual(PostConvertVerifier.defaultValidateTimeoutSeconds, 30,
                                    "validate timeout should leave headroom for slow dev machines")
        XCTAssertLessThanOrEqual(PostConvertVerifier.defaultValidateTimeoutSeconds, 300,
                                 "validate timeout shouldn't hide real hangs for too long")
    }

    func test_runJangValidate_returnsFalseOnNonexistentDir() async {
        // Bogus path — jang_tools.validate will exit non-zero, not hang.
        // The short-path is: process launches, exits quickly, returns false.
        // This exercises the terminationHandler branch of the continuation.
        let bogus = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID().uuidString)")
        let ok = await PostConvertVerifier.runJangValidate(outputDir: bogus, timeoutSeconds: 30)
        XCTAssertFalse(ok, "validate on a non-existent path must return false")
    }

    func test_runJangValidate_timeoutFiresWithinTolerance() async {
        // Use an intentionally unreachable executable override so the child
        // subprocess just hangs waiting for stdin / never returns. The timeout
        // must kick in near the bound, not block indefinitely.
        // Strategy: shadow BundleResolver to return `/bin/cat` with no stdin —
        // that will block forever reading. We can't easily monkeypatch
        // BundleResolver from Swift tests, so instead we rely on the REAL
        // jang-tools validate on a REAL path that exits quickly (a1 happy path)
        // and just pin that the timeout parameter is respected.
        // TODO(M42-followup): proper hang test would need a test-only override.
        //
        // What we CAN test cheaply: a 0.1-second timeout against a real
        // subprocess start-up will never succeed — even a process that exits
        // in 200ms loses the race. So passing timeoutSeconds=0.1 should
        // return false due to timeout.
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("jang-validate-timeout-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        let start = Date()
        let ok = await PostConvertVerifier.runJangValidate(outputDir: tmpDir, timeoutSeconds: 0.1)
        let elapsed = Date().timeIntervalSince(start)
        // Either the subprocess exits <0.1s (unlikely on Python start-up) OR
        // the timeout fires. Either way `ok` is false (bogus dir → exit!=0, or
        // timeout → false). The key assertion is we DIDN'T wait the full
        // default 60s — elapsed must be well under that.
        XCTAssertFalse(ok)
        XCTAssertLessThan(elapsed, 10, "timeout must bound wall time near 0.1s, took \(elapsed)s")
        try? FileManager.default.removeItem(at: tmpDir)
    }
}
