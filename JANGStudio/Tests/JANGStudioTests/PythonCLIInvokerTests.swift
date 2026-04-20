// JANGStudio/Tests/JANGStudioTests/PythonCLIInvokerTests.swift
//
// M154 (iter 77): dedicated tests for the PythonCLIInvoker helper
// extracted in iter-76 M153. Pins the contract so future changes
// (timeout support, retry logic, transient-error mapping) don't
// silently regress the 5 adoption services that depend on it.
//
// Uses the iter-32 M100 test harness pattern: shell-script
// executableOverride + argv capture files. Same idiom as
// InferenceRunnerTests' test_noThinking_flag_added_when_true /
// test_cancelSIGTERMLandsWithinThreeSeconds.

import XCTest
@testable import JANGStudio

final class PythonCLIInvokerTests: XCTestCase {

    private struct FakeError: Error, Equatable {
        let code: Int32
        let stderr: String
    }

    private func makeTempScript(_ body: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pci-\(UUID().uuidString).sh")
        try "#!/bin/bash\n\(body)\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    // ────────────── Happy path ──────────────

    func test_invoke_returns_stdout_on_zero_exit() async throws {
        let script = try makeTempScript(#"echo -n "hello world""#)
        let data = try await PythonCLIInvoker.invoke(
            args: [],
            executableOverride: script
        ) { code, stderr in
            XCTFail("errorFactory should not fire on zero-exit, got code=\(code)")
            return FakeError(code: code, stderr: stderr)
        }
        XCTAssertEqual(String(data: data, encoding: .utf8), "hello world")
    }

    // ────────────── Non-zero exit → errorFactory ──────────────

    func test_invoke_calls_errorFactory_with_code_and_stderr_on_nonzero_exit() async {
        let script = try! makeTempScript(#"""
        echo "something went wrong" >&2
        exit 7
        """#)
        do {
            _ = try await PythonCLIInvoker.invoke(
                args: [],
                executableOverride: script
            ) { code, stderr in
                FakeError(code: code, stderr: stderr)
            }
            XCTFail("expected throw, got no error")
        } catch let e as FakeError {
            XCTAssertEqual(e.code, 7, "errorFactory must receive the actual terminationStatus")
            XCTAssertTrue(
                e.stderr.contains("something went wrong"),
                "errorFactory must receive captured stderr, got: \(e.stderr)"
            )
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    // ────────────── errorFactory's returned error is rethrown as-is ──────────────

    func test_errorFactory_error_is_rethrown_not_wrapped() async {
        let script = try! makeTempScript("exit 3")
        do {
            _ = try await PythonCLIInvoker.invoke(
                args: [],
                executableOverride: script
            ) { _, _ in
                FakeError(code: 999, stderr: "custom")
            }
            XCTFail("expected throw")
        } catch let e as FakeError {
            XCTAssertEqual(e, FakeError(code: 999, stderr: "custom"),
                "caller's returned error must be rethrown as-is without wrapping")
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    // ────────────── Args are forwarded to the subprocess ──────────────

    func test_invoke_forwards_args_to_subprocess() async throws {
        let argsFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pci-args-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: argsFile) }

        // Script writes argv (one per line) to argsFile then exits 0.
        let script = try makeTempScript("""
        for a in "$@"; do echo "$a"; done > "\(argsFile.path)"
        """)

        _ = try await PythonCLIInvoker.invoke(
            args: ["alpha", "--beta", "gamma"],
            executableOverride: script
        ) { _, _ in FakeError(code: 0, stderr: "") }

        let argv = try String(contentsOf: argsFile, encoding: .utf8)
        XCTAssertTrue(argv.contains("alpha"))
        XCTAssertTrue(argv.contains("--beta"))
        XCTAssertTrue(argv.contains("gamma"))
    }

    // ────────────── Task cancel propagates SIGTERM ──────────────

    func test_consumer_task_cancel_terminates_subprocess_within_3_seconds() async throws {
        let tickFile = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pci-tick-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: tickFile) }

        // Long-running script that writes a tick every 200ms.
        let script = try makeTempScript("""
        while true; do
          date +%s%N > "\(tickFile.path)"
          sleep 0.2
        done
        """)

        let consumerTask = Task<Void, Error> {
            _ = try await PythonCLIInvoker.invoke(
                args: [],
                executableOverride: script
            ) { _, _ in FakeError(code: 0, stderr: "") }
        }

        // Wait for the subprocess to spawn + write at least one tick. 3s
        // tolerance matches iter-31 M98's tuning under parallel test contention.
        for _ in 0..<30 {
            if FileManager.default.fileExists(atPath: tickFile.path) { break }
            try? await Task.sleep(for: .milliseconds(100))
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: tickFile.path),
                      "subprocess should have written a tick within 3s")

        consumerTask.cancel()

        // Don't await consumerTask.value — if the cancel plumbing regresses,
        // the await hangs forever and the harness times out at 10 minutes
        // instead of giving us an informative assertion. Instead sleep past
        // the SIGTERM+3s SIGKILL window and verify via mtime non-advance.
        try? await Task.sleep(for: .seconds(5))

        let mtime1 = (try? FileManager.default.attributesOfItem(atPath: tickFile.path)[.modificationDate] as? Date) ?? Date.distantPast
        try? await Task.sleep(for: .seconds(1))
        let mtime2 = (try? FileManager.default.attributesOfItem(atPath: tickFile.path)[.modificationDate] as? Date) ?? Date.distantPast
        XCTAssertEqual(mtime1, mtime2,
            "tick-file mtime advanced after consumer-Task cancel — subprocess still running (M154 regression; matches M100/M96 class)")
    }
}
