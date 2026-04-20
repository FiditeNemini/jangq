import XCTest
@testable import JANGStudio

final class InferenceRunnerTests: XCTestCase {
    func test_result_decodes_from_json() throws {
        let json = #"""
        {"text":"Hello world","tokens":2,"tokens_per_sec":42.3,"elapsed_s":0.05,"load_time_s":0.3,"peak_rss_mb":3412.5,"model":"/path/to/model"}
        """#
        let r = try JSONDecoder().decode(InferenceResult.self, from: Data(json.utf8))
        XCTAssertEqual(r.text, "Hello world")
        XCTAssertEqual(r.tokens, 2)
        XCTAssertEqual(r.tokensPerSec, 42.3, accuracy: 0.01)
        XCTAssertEqual(r.peakRssMb, 3412.5, accuracy: 0.01)
    }

    func test_result_decodes_without_load_time() throws {
        let json = #"""
        {"text":"Hi","tokens":1,"tokens_per_sec":10.0,"elapsed_s":0.1,"peak_rss_mb":100.0,"model":"/m"}
        """#
        let r = try JSONDecoder().decode(InferenceResult.self, from: Data(json.utf8))
        XCTAssertNil(r.loadTimeS)
    }

    func test_inference_error_equatable() {
        let a = InferenceError(message: "x", code: 1)
        let b = InferenceError(message: "x", code: 1)
        XCTAssertEqual(a, b)
    }

    // MARK: - Iter 32: M100 — consumer Task-cancel must propagate to subprocess
    //
    // Applying iter-30/31's rigor: iter 3 (M19) fixed explicit `await
    // runner.cancel()` but never stress-tested the TASK-LEVEL cancel path.
    // InferenceRunner.generate() awaits `withCheckedContinuation` for
    // termination — that continuation does NOT participate in Task
    // cancellation, so a cancelled consumer Task leaves the subprocess
    // running to natural completion.

    private func makeTempScript(_ body: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ir-\(UUID().uuidString).sh")
        try "#!/bin/bash\n\(body)\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    func test_consumerTaskCancel_terminatesSubprocess() async throws {
        // Long-running script that writes timestamps every 200ms.
        let tickFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ir-tick-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: tickFile) }

        let slow = try makeTempScript("""
        while true; do
          date +%s%N > "\(tickFile.path)"
          sleep 0.2
        done
        """)
        let runner = InferenceRunner(
            modelPath: URL(fileURLWithPath: "/tmp/nope"),
            executableOverride: slow
        )

        // Consumer task awaits generate(). We cancel it mid-run.
        let consumerTask = Task<InferenceResult, Error> {
            return try await runner.generate(prompt: "x", maxTokens: 1)
        }

        // Wait for the subprocess to spawn + write at least one tick. 3s
        // tolerance matches iter 31's tuning for contention.
        for _ in 0..<30 {
            if FileManager.default.fileExists(atPath: tickFile.path) { break }
            try? await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: tickFile.path),
                      "subprocess should have written a tick within 3s")

        consumerTask.cancel()

        // Note: we do NOT `await consumerTask.value` — if the fix regresses,
        // that await hangs forever and the test harness times out at 10min
        // instead of the expected few seconds. Instead we sleep past the
        // SIGTERM + 3s SIGKILL escalation window and verify via mtime.
        try? await Task.sleep(for: .seconds(5))

        let mtime1 = (try? FileManager.default.attributesOfItem(atPath: tickFile.path)[.modificationDate] as? Date) ?? Date.distantPast
        try? await Task.sleep(for: .seconds(1))
        let mtime2 = (try? FileManager.default.attributesOfItem(atPath: tickFile.path)[.modificationDate] as? Date) ?? Date.distantPast
        XCTAssertEqual(mtime1, mtime2,
                       "tick-file mtime advanced after consumer-Task cancel — subprocess still running (M100 regression; same class as M96/M98)")
    }

    // MARK: - Iter 45: M121 — --no-thinking flag propagation
    //
    // Reasoning-model smoke-test toggle: when InferenceRunner.generate is
    // called with noThinking=true, the subprocess must receive --no-thinking
    // in its argv. When false (or omitted), it must NOT. Pre-M121 the flag
    // didn't exist — reasoning models consumed the 150-token smoke budget on
    // <think>…</think> wrappers and never emitted an answer.

    func test_noThinking_flag_added_when_true() async throws {
        let argsFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ir-args-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: argsFile) }

        // Script writes its argv (one arg per line) to argsFile, then exits
        // 3 so InferenceRunner surfaces an error — we only care about argv.
        let capture = try makeTempScript("""
        for a in "$@"; do echo "$a"; done > "\(argsFile.path)"
        exit 3
        """)
        let runner = InferenceRunner(
            modelPath: URL(fileURLWithPath: "/tmp/nope"),
            executableOverride: capture
        )
        do {
            _ = try await runner.generate(
                prompt: "2+2?",
                maxTokens: 16,
                noThinking: true
            )
        } catch {
            // expected — fake subprocess exits 3
        }
        let argv = (try? String(contentsOf: argsFile, encoding: .utf8)) ?? ""
        XCTAssertTrue(argv.contains("--no-thinking"),
                      "noThinking:true should add --no-thinking to argv. Got:\n\(argv)")
    }

    func test_noThinking_flag_absent_when_default_false() async throws {
        let argsFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ir-args-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: argsFile) }

        let capture = try makeTempScript("""
        for a in "$@"; do echo "$a"; done > "\(argsFile.path)"
        exit 3
        """)
        let runner = InferenceRunner(
            modelPath: URL(fileURLWithPath: "/tmp/nope"),
            executableOverride: capture
        )
        do {
            _ = try await runner.generate(prompt: "Hi", maxTokens: 8)
        } catch {
            // expected
        }
        let argv = (try? String(contentsOf: argsFile, encoding: .utf8)) ?? ""
        XCTAssertFalse(argv.contains("--no-thinking"),
                       "default call should NOT add --no-thinking — preserves existing reasoning-benchmark behavior. Got:\n\(argv)")
    }

    func test_explicit_cancel_still_works_via_actor_method() async throws {
        // Regression pin for iter 3's M19 fix: explicit await runner.cancel()
        // must ALSO continue to work after the iter-32 task-cancel fix. Both
        // paths must kill the subprocess.
        let slow = try makeTempScript("sleep 60")
        let runner = InferenceRunner(
            modelPath: URL(fileURLWithPath: "/tmp/nope"),
            executableOverride: slow
        )
        let t0 = Date()
        Task { try? await Task.sleep(for: .milliseconds(200)); await runner.cancel() }
        do {
            _ = try await runner.generate(prompt: "x", maxTokens: 1)
            XCTFail("expected cancellation error")
        } catch let e as InferenceError {
            XCTAssertTrue(e.wasCancelled,
                          "explicit cancel must surface InferenceError.cancelledCode")
        } catch {
            XCTFail("wrong error type \(error)")
        }
        let elapsed = Date().timeIntervalSince(t0)
        XCTAssertLessThan(elapsed, 3.5, "explicit cancel took \(elapsed)s")
    }
}
