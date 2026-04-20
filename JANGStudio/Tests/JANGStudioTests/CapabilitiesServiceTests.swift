import XCTest
@testable import JANGStudio

final class CapabilitiesServiceTests: XCTestCase {
    func test_frozen_has_known_content() {
        let f = Capabilities.frozen
        XCTAssertTrue(f.jangtqWhitelist.contains("qwen3_5_moe"))
        XCTAssertTrue(f.jangtqWhitelist.contains("minimax_m2"))
        XCTAssertEqual(f.defaultBlockSize, 64)
        XCTAssertEqual(f.defaultMethod, "mse")
        XCTAssertEqual(f.methods.count, 3)
        XCTAssertTrue(f.tokenizerClassBlocklist.contains("TokenizersBackend"))
    }

    @MainActor
    func test_service_starts_with_frozen() {
        let s = CapabilitiesService()
        XCTAssertEqual(s.capabilities, .frozen)
        XCTAssertFalse(s.isFromBundle)
    }

    func test_dtype_info_decodes() throws {
        let json = #"{"name":"bfloat16","alias":"bf16","description":"Test"}"#
        let d = try JSONDecoder().decode(Capabilities.DtypeInfo.self, from: Data(json.utf8))
        XCTAssertEqual(d.alias, "bf16")
    }
}
