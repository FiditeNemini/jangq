import XCTest
@testable import JANGKit

final class ModelTests: XCTestCase {

    func test_sampling_config_defaults() {
        let c = JANGKit.SamplingConfig()
        XCTAssertEqual(c.temperature, 0.0)
        XCTAssertEqual(c.topP, 1.0)
        XCTAssertEqual(c.maxTokens, 200)
    }

    func test_generation_result_finish_reasons() {
        XCTAssertEqual(JANGKit.GenerationResult.FinishReason.stop.rawValue, "stop")
        XCTAssertEqual(JANGKit.GenerationResult.FinishReason.maxTokens.rawValue, "maxTokens")
        XCTAssertEqual(JANGKit.GenerationResult.FinishReason.cancelled.rawValue, "cancelled")
        XCTAssertEqual(JANGKit.GenerationResult.FinishReason.error.rawValue, "error")
    }

    func test_load_missing_jang_config_throws_modelLoadFailed() async {
        let url = URL(fileURLWithPath: "/tmp/definitely-not-there-xyz")
        do {
            _ = try await JANGKit.Model.load(at: url)
            XCTFail("expected throw")
        } catch let e as JANGKit.ModelError {
            switch e {
            case .modelLoadFailed: break  // expected — jang_config.json not found
            default: XCTFail("wrong error kind: \(e)")
            }
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }

    func test_load_jangtq_dir_throws_jangtqNotYetSupported() async throws {
        // Build a minimal jang_config.json with weight_format == "mxtq"
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("jangtq_test_\(Int.random(in: 0..<100000))")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let cfg: [String: Any] = ["weight_format": "mxtq", "version": 2]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        try data.write(to: dir.appendingPathComponent("jang_config.json"))

        do {
            _ = try await JANGKit.Model.load(at: dir)
            XCTFail("expected throw")
        } catch let e as JANGKit.ModelError {
            switch e {
            case .jangtqNotYetSupported: break  // expected
            default: XCTFail("wrong error kind: \(e)")
            }
        } catch {
            XCTFail("wrong error type: \(error)")
        }
    }
}
