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

    private func makeTempScript(_ body: String) throws -> URL {
        let url = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-\(UUID().uuidString).sh")
        try "#!/bin/bash\n\(body)\n".write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }
}
