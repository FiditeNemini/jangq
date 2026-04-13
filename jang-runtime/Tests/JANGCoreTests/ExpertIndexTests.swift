import XCTest
@testable import JANGCore

final class ExpertIndexTests: XCTestCase {
    /// Build a synthetic index file matching the format Python writes.
    private func writeSynthetic(
        nLayers: Int,
        nExpertsPerLayer: Int,
        entries: [(layer: Int, expert: Int, file: Int, offset: Int, nbytes: Int)],
        to url: URL
    ) throws {
        var data = Data()

        // IndexHeader: <I magic, H version, H _pad, I n_layers, I n_experts_per_layer, Q n_entries>
        var magic: UInt32 = JangSpecFormat.indexMagic
        var version: UInt16 = 1
        var pad: UInt16 = 0
        var nL: UInt32 = UInt32(nLayers)
        var nE: UInt32 = UInt32(nExpertsPerLayer)
        var n: UInt64 = UInt64(entries.count)
        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &pad) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nL) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nE) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }

        // Each entry: <I layer, I expert, H file_id, H _pad, Q offset, Q nbytes>
        for e in entries {
            var l: UInt32 = UInt32(e.layer)
            var ex: UInt32 = UInt32(e.expert)
            var f: UInt16 = UInt16(e.file)
            var p: UInt16 = 0
            var off: UInt64 = UInt64(e.offset)
            var nb: UInt64 = UInt64(e.nbytes)
            withUnsafeBytes(of: &l) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &ex) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &f) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &p) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &off) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &nb) { data.append(contentsOf: $0) }
        }
        try data.write(to: url)
    }

    func testRoundTripSynthetic() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangcore-idx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let path = tmp.appendingPathComponent("experts.jsidx")
        try writeSynthetic(
            nLayers: 2,
            nExpertsPerLayer: 2,
            entries: [
                (0, 0, 0, 0, 4096),
                (0, 1, 0, 4096, 8192),
                (1, 0, 1, 0, 4096),
                (1, 1, 1, 4096, 4096),
            ],
            to: path
        )

        let idx = try ExpertIndex(contentsOf: path)
        XCTAssertEqual(idx.nLayers, 2)
        XCTAssertEqual(idx.nExpertsPerLayer, 2)
        XCTAssertEqual(idx.entries.count, 4)

        let hit = try idx.entry(layer: 1, expert: 0)
        XCTAssertEqual(hit.fileID, 1)
        XCTAssertEqual(hit.offset, 0)
        XCTAssertEqual(hit.nbytes, 4096)

        let hit2 = try idx.entry(layer: 0, expert: 1)
        XCTAssertEqual(hit2.offset, 4096)
        XCTAssertEqual(hit2.nbytes, 8192)
    }

    func testLookupMissingThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangcore-idx-miss-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let path = tmp.appendingPathComponent("experts.jsidx")
        try writeSynthetic(nLayers: 1, nExpertsPerLayer: 1, entries: [(0, 0, 0, 0, 4096)], to: path)

        let idx = try ExpertIndex(contentsOf: path)
        XCTAssertThrowsError(try idx.entry(layer: 99, expert: 99)) { err in
            guard case JangSpecError.missingEntry = err else {
                XCTFail("expected missingEntry error, got \(err)")
                return
            }
        }
    }

    func testBadMagicThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangcore-idx-badmagic-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let path = tmp.appendingPathComponent("bad.jsidx")
        try Data(count: JangSpecFormat.indexHeaderSize).write(to: path)

        XCTAssertThrowsError(try ExpertIndex(contentsOf: path)) { err in
            guard case JangSpecError.badMagic = err else {
                XCTFail("expected badMagic error, got \(err)")
                return
            }
        }
    }
}
