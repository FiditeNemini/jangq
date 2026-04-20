import XCTest
@testable import JANGKit

final class JANGKitTests: XCTestCase {
    func test_version_is_nonempty() {
        XCTAssertFalse(JANGKit.version.isEmpty)
    }
}
