// JANGStudio/Tests/JANGStudioTests/PythonRunnerTests.swift
import XCTest
@testable import JANGStudio

final class PythonRunnerTests: XCTestCase {
    private var fakeScript: URL {
        Bundle(for: Self.self).url(forResource: "fake_convert", withExtension: "sh")!
    }

    func test_run_streamsAllFiveGoldenPhases() async throws {
        let runner = PythonRunner(executableOverride: fakeScript, extraArgs: [])
        var phases: [Int] = []
        var sawDone = false
        for try await ev in runner.run() {
            switch ev.payload {
            case .phase(let n, _, _): phases.append(n)
            case .done(let ok, _, _): XCTAssertTrue(ok); sawDone = true
            default: break
            }
        }
        XCTAssertEqual(phases, [1, 2, 3, 4, 5])
        XCTAssertTrue(sawDone)
    }

    func test_nonZeroExit_throws() async {
        let failScript = try! makeTempScript("exit 3")
        let runner = PythonRunner(executableOverride: failScript, extraArgs: [])
        do {
            for try await _ in runner.run() {}
            XCTFail("expected throw")
        } catch let e as ProcessError {
            XCTAssertEqual(e.code, 3)
        } catch {
            XCTFail("wrong error \(error)")
        }
    }

    func test_cancelSIGTERMLandsWithinThreeSeconds() async throws {
        // Long-running fake: 60s sleep
        let slow = try! makeTempScript("sleep 60")
        let runner = PythonRunner(executableOverride: slow, extraArgs: [])
        let t0 = Date()
        Task { try? await Task.sleep(for: .milliseconds(200)); await runner.cancel() }
        for try await _ in runner.run() {}
        let elapsed = Date().timeIntervalSince(t0)
        XCTAssertLessThan(elapsed, 3.5, "cancel took \(elapsed)s")
    }

    // MARK: - Iter 31: M98 — consumer Task-cancel must propagate to subprocess
    //
    // Applying iter-30's meta-lesson: features spanning Task / stream /
    // subprocess boundaries need integration tests with real subprocesses.
    // PythonRunner was established in iter 3 but was never stress-tested for
    // the "consumer abandons the stream" path — the same failure mode that
    // iter 30 (M96) fixed for PublishService.
    //
    // Scenario: Task wraps a `for try await` over runner.run(). Something
    // cancels THAT Task — not via runner.cancel(), but via SwiftUI view
    // dismount / Task.cancel() / deallocation / parent-task hierarchy.
    // The AsyncThrowingStream continuation terminates; without an
    // onTermination handler, the subprocess keeps running.

    func test_consumerTaskCancel_terminatesSubprocess() async throws {
        // Stronger test: the subprocess writes a tick file every 200ms. If
        // consumer-Task cancel properly propagates, the file's mtime stops
        // advancing within the SIGTERM+SIGKILL window. If it keeps
        // advancing, the subprocess is still running → bug.
        let tickFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pyr-tick-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: tickFile) }

        let script = try! makeTempScript("""
        while true; do
          date +%s%N > "\(tickFile.path)"
          sleep 0.2
        done
        """)
        let runner = PythonRunner(executableOverride: script, extraArgs: [])

        // Start subprocess via runner.run() OUTSIDE the consumer Task so it
        // isn't torn down by task isolation. Hold the stream alive in an
        // async let binding, then start iterating via a cancellable task.
        let stream = runner.run()
        let consumerTask = Task<Void, Error> {
            for try await _ in stream {}
        }

        // Let the subprocess spawn + write at least one tick. Wide tolerance
        // because under parallel test contention, Process.run() spawn latency
        // can spike past 1s on CI/local M1 even though the script's first
        // write cycle is 200ms.
        for _ in 0..<30 {   // up to 3s
            if FileManager.default.fileExists(atPath: tickFile.path) { break }
            try? await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: tickFile.path),
                      "subprocess should have written a tick within 3s")

        // Cancel the CONSUMER Task. We do NOT call runner.cancel() — the
        // whole point is the Task-level cancel path must independently tear
        // down the subprocess via continuation.onTermination.
        consumerTask.cancel()
        _ = try? await consumerTask.value

        // Wait past the SIGKILL-after-3s escalation window.
        try? await Task.sleep(for: .seconds(4))

        // Record the tick file's mtime AFTER the cancel window; then wait
        // another second and re-check. If mtime advances, the subprocess is
        // still running (bug). If mtime stays put, propagation worked.
        let mtime1 = (try? FileManager.default.attributesOfItem(atPath: tickFile.path)[.modificationDate] as? Date) ?? Date.distantPast
        try? await Task.sleep(for: .seconds(1))
        let mtime2 = (try? FileManager.default.attributesOfItem(atPath: tickFile.path)[.modificationDate] as? Date) ?? Date.distantPast
        XCTAssertEqual(mtime1, mtime2,
                       "tick-file mtime advanced after consumer-Task cancel — subprocess is still running (M98 regression)")
    }

    // NOTE: a `test_streamAbandon_terminatesSubprocess` was considered but
    // removed because it's a different resource-leak scenario from M98. If
    // the producer closure (launch()) still holds the continuation and the
    // consumer drops the stream value without iterating, there's no signal
    // for onTermination to fire — the producer is still "running" from the
    // continuation's perspective. The realistic bug is consumer-task-cancel
    // (pinned by `test_consumerTaskCancel_terminatesSubprocess` above). A
    // pure stream-abandon cleanup would require language-level changes or
    // a finalizer pattern; logged as M99 for follow-up if needed.

    private func makeTempScript(_ body: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-\(UUID().uuidString).sh")
        try "#!/bin/bash\n\(body)\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }
}
