import XCTest
@testable import JANGCore

final class JangSpecFormatTests: XCTestCase {
    func testStructSizesMatchPython() {
        // These values must match jang_tools.jangspec.format exactly.
        XCTAssertEqual(JangSpecFormat.blobHeaderSize, 32)
        XCTAssertEqual(JangSpecFormat.tensorHeaderSize, 36)
        XCTAssertEqual(JangSpecFormat.indexEntrySize, 28)
        XCTAssertEqual(JangSpecFormat.indexHeaderSize, 24)
    }

    func testMagicNumbers() {
        // "JSPE" little-endian = 0x4550534A
        XCTAssertEqual(JangSpecFormat.blobMagic, 0x4550_534A)
        // "SJIX" little-endian = 0x58494A53
        XCTAssertEqual(JangSpecFormat.indexMagic, 0x58_494A_53)
    }

    func testAlignUp() {
        XCTAssertEqual(JangSpecFormat.alignUp(0), 0)
        XCTAssertEqual(JangSpecFormat.alignUp(1), 4096)
        XCTAssertEqual(JangSpecFormat.alignUp(4095), 4096)
        XCTAssertEqual(JangSpecFormat.alignUp(4096), 4096)
        XCTAssertEqual(JangSpecFormat.alignUp(4097), 8192)
    }

    func testExpertFilename() {
        XCTAssertEqual(JangSpecFormat.expertFilename(idx: 0), "target/experts-00000.bin")
        XCTAssertEqual(JangSpecFormat.expertFilename(idx: 42), "target/experts-00042.bin")
    }
}
