import XCTest
@testable import JANGStudio

final class ProfilesServiceTests: XCTestCase {
    func test_frozen_has_15_jang_profiles() {
        XCTAssertEqual(Profiles.frozen.jang.count, 15)
    }

    func test_frozen_has_3_jangtq_profiles() {
        XCTAssertEqual(Profiles.frozen.jangtq.count, 3)
    }

    func test_default_profile_is_jang_4k() {
        XCTAssertEqual(Profiles.frozen.defaultProfile, "JANG_4K")
        let defaults = Profiles.frozen.jang.filter { $0.isDefault }
        XCTAssertEqual(defaults.count, 1)
        XCTAssertEqual(defaults.first?.name, "JANG_4K")
    }

    func test_kquant_profiles_marked() {
        let kq = Profiles.frozen.jang.filter { $0.isKquant }.map { $0.name }
        XCTAssertTrue(kq.contains("JANG_3K"))
        XCTAssertTrue(kq.contains("JANG_4K"))
        XCTAssertTrue(kq.contains("JANG_5K"))
        XCTAssertTrue(kq.contains("JANG_6K"))
    }

    @MainActor
    func test_service_starts_with_frozen() {
        let s = ProfilesService()
        XCTAssertEqual(s.profiles, .frozen)
        XCTAssertFalse(s.isFromBundle)
    }
}
