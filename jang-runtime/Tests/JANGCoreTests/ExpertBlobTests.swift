import XCTest
@testable import JANGCore

final class ExpertBlobTests: XCTestCase {
    /// Build a minimal synthetic blob matching the format Python writes.
    /// Uses 9 tensor entries (gate/up/down × qweight/scales/biases) with
    /// tiny payloads so we can hand-verify offsets.
    private func makeSyntheticBlob(bits: UInt8 = 4) -> Data {
        // One uint32 per qweight (4 bytes), one f16 per scale (2 bytes),
        // one f16 per bias (2 bytes), times 3 kinds = 24 bytes payload.
        let payload: [UInt8] = Array(repeating: 0xAB, count: 24)
        // But we want distinct bytes per tensor to detect offset bugs.
        var distinct = [UInt8]()
        for i in 0..<24 { distinct.append(UInt8(i) | 0x80) }

        let headerArea = JangSpecFormat.blobHeaderSize + 9 * JangSpecFormat.tensorHeaderSize
        let payloadOffset = headerArea
        let payloadBytes = distinct.count

        var data = Data()

        // BlobHeader: <I magic, H version, H n_tensors, I layer_idx, I expert_id, Q payload_offset, Q payload_bytes>
        var magic = JangSpecFormat.blobMagic
        var version: UInt16 = 1
        var nTensors: UInt16 = 9
        var layerIdx: UInt32 = 7
        var expertID: UInt32 = 3
        var payOff: UInt64 = UInt64(payloadOffset)
        var payBytes: UInt64 = UInt64(payloadBytes)
        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &layerIdx) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &expertID) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &payOff) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &payBytes) { data.append(contentsOf: $0) }

        // 9 tensor headers: <B kind, B bits, H _pad, I dtype, I d0, I d1, I d2, Q offset, Q nbytes>
        // Serialization order (matches Python _KINDS × _DTYPES):
        //   (gate, qweight), (gate, scales), (gate, biases),
        //   (up,   qweight), (up,   scales), (up,   biases),
        //   (down, qweight), (down, scales), (down, biases)
        let kindsDtypes: [(UInt8, UInt32, [UInt32])] = [
            (JangSpecFormat.TensorKind.gate.rawValue, JangSpecFormat.TensorDType.qweight.rawValue, [1, 1, 0]),
            (JangSpecFormat.TensorKind.gate.rawValue, JangSpecFormat.TensorDType.scales.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.gate.rawValue, JangSpecFormat.TensorDType.biases.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.up.rawValue,   JangSpecFormat.TensorDType.qweight.rawValue, [1, 1, 0]),
            (JangSpecFormat.TensorKind.up.rawValue,   JangSpecFormat.TensorDType.scales.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.up.rawValue,   JangSpecFormat.TensorDType.biases.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.down.rawValue, JangSpecFormat.TensorDType.qweight.rawValue, [1, 1, 0]),
            (JangSpecFormat.TensorKind.down.rawValue, JangSpecFormat.TensorDType.scales.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.down.rawValue, JangSpecFormat.TensorDType.biases.rawValue,  [1, 1, 0]),
        ]
        var runningOffset: UInt64 = 0
        let sliceSizes: [Int] = [
            4, 2, 2,  // gate: qweight=4, scales=2, biases=2
            4, 2, 2,  // up
            4, 2, 2,  // down
        ]
        for (i, triple) in kindsDtypes.enumerated() {
            var kind = triple.0
            var bitsLocal = bits
            var pad: UInt16 = 0
            var dtype = triple.1
            var d0 = triple.2[0]
            var d1 = triple.2[1]
            var d2 = triple.2[2]
            var off = runningOffset
            var nb = UInt64(sliceSizes[i])
            withUnsafeBytes(of: &kind) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &bitsLocal) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &pad) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &dtype) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &d0) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &d1) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &d2) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &off) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &nb) { data.append(contentsOf: $0) }
            runningOffset += UInt64(sliceSizes[i])
        }

        // Payload
        data.append(contentsOf: distinct)

        // Pad to alignment
        let padded = JangSpecFormat.alignUp(data.count)
        data.append(contentsOf: [UInt8](repeating: 0, count: padded - data.count))

        XCTAssertEqual(data.count % JangSpecFormat.blobAlignment, 0)
        return data
    }

    func testParseSyntheticBlob() throws {
        let bytes = makeSyntheticBlob()
        let blob = try ExpertBlob(rawBytes: bytes)

        XCTAssertEqual(blob.layerIdx, 7)
        XCTAssertEqual(blob.expertID, 3)
        XCTAssertEqual(blob.bits, 4)
        XCTAssertEqual(blob.tensors.count, 9)

        // The gate/qweight should be bytes 0..3 of the payload region.
        let gateQ = blob.tensor(kind: .gate, dtype: .qweight)
        XCTAssertNotNil(gateQ)
        XCTAssertEqual(gateQ!.count, 4)
        XCTAssertEqual(gateQ![0], UInt8(0) | 0x80)

        // The down/biases is the last 2 bytes of the payload.
        let downB = blob.tensor(kind: .down, dtype: .biases)
        XCTAssertNotNil(downB)
        XCTAssertEqual(downB!.count, 2)
        XCTAssertEqual(downB![0], UInt8(22) | 0x80)
        XCTAssertEqual(downB![1], UInt8(23) | 0x80)
    }

    func testBadMagicThrows() throws {
        var bytes = makeSyntheticBlob()
        bytes.replaceSubrange(0..<4, with: Data([0, 0, 0, 0]))
        XCTAssertThrowsError(try ExpertBlob(rawBytes: bytes)) { err in
            guard case JangSpecError.badMagic = err else {
                XCTFail("expected badMagic, got \(err)")
                return
            }
        }
    }
}
