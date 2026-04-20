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
}
