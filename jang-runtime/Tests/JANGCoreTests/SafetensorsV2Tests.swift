import XCTest
@testable import JANGCore

final class SafetensorsV2Tests: XCTestCase {
    /// Build a tiny synthetic safetensors file with 2 tensors.
    private func writeSynthetic(to url: URL) throws {
        // Tensor A: "alpha" — U32 shape [2, 3], 6 * 4 = 24 bytes
        // Tensor B: "beta"  — F16 shape [4],    4 * 2 = 8 bytes
        let alphaBytes: [UInt8] = [
            0x01, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00,
            0x05, 0x00, 0x00, 0x00,
            0x06, 0x00, 0x00, 0x00,
        ]
        let betaBytes: [UInt8] = [0x00, 0x3C, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44]
        // offsets in the data section:
        //   alpha: [0, 24]
        //   beta:  [24, 32]
        let headerObj: [String: Any] = [
            "alpha": [
                "dtype": "U32",
                "shape": [2, 3],
                "data_offsets": [0, 24],
            ],
            "beta": [
                "dtype": "F16",
                "shape": [4],
                "data_offsets": [24, 32],
            ],
        ]
        let headerJSON = try JSONSerialization.data(
            withJSONObject: headerObj, options: [.sortedKeys]
        )
        var out = Data()
        var headerSize = UInt64(headerJSON.count)
        withUnsafeBytes(of: &headerSize) { out.append(contentsOf: $0) }
        out.append(headerJSON)
        out.append(contentsOf: alphaBytes)
        out.append(contentsOf: betaBytes)
        try out.write(to: url)
    }

    func testParsesSyntheticFile() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("stv2-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try writeSynthetic(to: tmp)

        let file = try SafetensorsV2File(url: tmp)
        XCTAssertEqual(file.tensorNames.sorted(), ["alpha", "beta"])

        let alpha = try file.info(for: "alpha")
        XCTAssertEqual(alpha.dtype, .u32)
        XCTAssertEqual(alpha.shape, [2, 3])
        XCTAssertEqual(alpha.dataLength, 24)

        let beta = try file.info(for: "beta")
        XCTAssertEqual(beta.dtype, .f16)
        XCTAssertEqual(beta.shape, [4])
        XCTAssertEqual(beta.dataLength, 8)

        // Actual bytes match what we wrote.
        let alphaBytes = try file.bytes(for: "alpha")
        XCTAssertEqual(alphaBytes.count, 24)
        alphaBytes.withUnsafeBytes { raw in
            XCTAssertEqual(raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self), 1)
            XCTAssertEqual(raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self), 2)
            XCTAssertEqual(raw.loadUnaligned(fromByteOffset: 20, as: UInt32.self), 6)
        }
    }

    func testMissingTensorThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("stv2-missing-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try writeSynthetic(to: tmp)

        let file = try SafetensorsV2File(url: tmp)
        XCTAssertThrowsError(try file.info(for: "nope")) { err in
            guard case SafetensorsV2Error.missingTensor = err else {
                XCTFail("expected missingTensor, got \(err)")
                return
            }
        }
    }

    func testTruncatedFileThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("stv2-short-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Data([0x01, 0x02]).write(to: tmp)

        XCTAssertThrowsError(try SafetensorsV2File(url: tmp)) { err in
            guard case SafetensorsV2Error.truncated = err else {
                XCTFail("expected truncated, got \(err)")
                return
            }
        }
    }
}
