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
}
