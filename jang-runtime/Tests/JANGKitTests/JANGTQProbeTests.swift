import XCTest
@testable import JANGKit

final class JANGTQProbeTests: XCTestCase {
    /// Verifies that JANGTQ detection works and routes to the real JANGTQ loader.
    /// A fake dir without real weights triggers a .modelLoadFailed (or .metalDeviceUnavailable
    /// on non-Apple-Silicon CI), not .jangtqNotYetSupported (that error no longer exists).
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
            XCTFail("expected an error from fake JANGTQ dir without real weights")
        } catch let e as JANGKit.ModelError {
            // Real JANGTQ loader is invoked; it fails on the fake dir.
            // .metalDeviceUnavailable is acceptable on non-Apple-Silicon hosts.
            switch e {
            case .modelLoadFailed, .metalDeviceUnavailable, .tokenizerLoadFailed:
                break  // expected — JANGTQ loading attempted, failed on missing weights/config
            default:
                XCTFail("unexpected JANGKit.ModelError: \(e)")
            }
        } catch {
            XCTFail("unexpected error type: \(error)")
        }
    }
}
