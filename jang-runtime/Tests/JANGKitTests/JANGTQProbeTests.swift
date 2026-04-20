import XCTest
@testable import JANGKit

final class JANGTQProbeTests: XCTestCase {
    /// Documents that JANGTQ detection works even when we eventually route to the real loader.
    /// Until commit 2 lands, this expects ModelError.modelLoadFailed (JANGTQ path exists) OR
    /// ModelError.jangtqNotYetSupported (prior behavior).
    func test_jangtq_detection_probe() async throws {
        // Create a minimal fake JANGTQ dir to exercise the detection branch
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("tq-probe-\(UUID())")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        // Minimal jang_config.json with JANGTQ marker
        let jangConfig = #"""
        {"format":"jang","format_version":"2.0","weight_format":"mxtq","quantization":{"method":"jangtq","bit_widths_used":[4]}}
        """#
        try jangConfig.write(
            to: tmp.appendingPathComponent("jang_config.json"),
            atomically: true, encoding: .utf8
        )
        try "{}".write(
            to: tmp.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8
        )

        do {
            _ = try await JANGKit.Model.load(at: tmp)
            XCTFail("expected an error from fake JANGTQ dir without weights")
        } catch let e as JANGKit.ModelError {
            // Either error is acceptable — we're confirming the detection branch is exercised.
            switch e {
            case .modelLoadFailed, .jangtqNotYetSupported:
                break  // expected: prior behavior is .jangtqNotYetSupported, post-commit-2 will be .modelLoadFailed
            default:
                XCTFail("unexpected JANGKit.ModelError: \(e)")
            }
        } catch {
            XCTFail("unexpected error type: \(error)")
        }
    }
}
