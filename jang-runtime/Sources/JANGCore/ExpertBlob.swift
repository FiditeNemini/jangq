import Foundation

/// One tensor slice inside an expert blob.
///
/// The `slice` is a view on the blob's backing bytes. Consumers must not
/// retain it beyond the blob's lifetime unless they make their own copy.
public struct ExpertBlobTensor: Sendable {
    public let kind: JangSpecFormat.TensorKind
    public let dtype: JangSpecFormat.TensorDType
    public let bits: Int
    public let dims: [Int]       // zero-padded dims stripped
    public let slice: Data       // the raw bytes for this tensor
}

/// A parsed expert blob — header plus 9 tensor slices.
///
/// `ExpertBlob(rawBytes:)` validates the magic and header and decodes the
/// per-tensor offsets. The tensor byte ranges are held as `Data` slices
/// that reference the caller-owned backing buffer. For mmap'd reads this
/// is zero-copy; for in-memory test data the slices share storage with
/// the originating `Data`.
public struct ExpertBlob: Sendable {
    public let layerIdx: Int
    public let expertID: Int
    public let bits: Int
    public let tensors: [ExpertBlobTensor]

    public init(rawBytes data: Data) throws {
        guard data.count >= JangSpecFormat.blobHeaderSize else {
            throw JangSpecError.invalidBlob("blob too short for header (\(data.count) bytes)")
        }

        let (magic, version, nTensors, layer, expert, payloadOffset, payloadBytes):
            (UInt32, UInt16, UInt16, UInt32, UInt32, UInt64, UInt64) =
                data.withUnsafeBytes { raw in
                    let m = raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
                    let v = raw.loadUnaligned(fromByteOffset: 4, as: UInt16.self)
                    let n = raw.loadUnaligned(fromByteOffset: 6, as: UInt16.self)
                    let l = raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
                    let e = raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
                    let po = raw.loadUnaligned(fromByteOffset: 16, as: UInt64.self)
                    let pb = raw.loadUnaligned(fromByteOffset: 24, as: UInt64.self)
                    return (m, v, n, l, e, po, pb)
                }

        guard magic == JangSpecFormat.blobMagic else {
            throw JangSpecError.badMagic(
                expected: JangSpecFormat.blobMagic,
                actual: magic,
                at: URL(fileURLWithPath: "(blob)")
            )
        }
        guard version == 1 else {
            throw JangSpecError.unsupportedVersion(
                field: "blob", value: Int(version), supported: 1
            )
        }
        guard nTensors == 9 else {
            throw JangSpecError.invalidBlob("expected 9 tensor entries, got \(nTensors)")
        }

        let tensorCount = Int(nTensors)
        let headerArea = JangSpecFormat.blobHeaderSize + tensorCount * JangSpecFormat.tensorHeaderSize
        let payOff = Int(payloadOffset)
        let payBytes = Int(payloadBytes)

        guard payOff == headerArea else {
            throw JangSpecError.invalidBlob(
                "payload_offset \(payOff) does not match header area \(headerArea)"
            )
        }
        guard data.count >= payOff + payBytes else {
            throw JangSpecError.invalidBlob(
                "blob shorter than declared payload: \(data.count) < \(payOff + payBytes)"
            )
        }

        var bitsSeen: Int? = nil
        var parsed: [ExpertBlobTensor] = []
        parsed.reserveCapacity(tensorCount)

        for i in 0..<tensorCount {
            let cursor = JangSpecFormat.blobHeaderSize + i * JangSpecFormat.tensorHeaderSize
            let (kindRaw, bitsVal, dtypeRaw, d0, d1, d2, offRaw, nbRaw):
                (UInt8, UInt8, UInt32, UInt32, UInt32, UInt32, UInt64, UInt64) =
                    data.withUnsafeBytes { raw in
                        let k = raw.loadUnaligned(fromByteOffset: cursor + 0, as: UInt8.self)
                        let b = raw.loadUnaligned(fromByteOffset: cursor + 1, as: UInt8.self)
                        // 2 bytes pad at cursor + 2
                        let dt = raw.loadUnaligned(fromByteOffset: cursor + 4, as: UInt32.self)
                        let x = raw.loadUnaligned(fromByteOffset: cursor + 8, as: UInt32.self)
                        let y = raw.loadUnaligned(fromByteOffset: cursor + 12, as: UInt32.self)
                        let z = raw.loadUnaligned(fromByteOffset: cursor + 16, as: UInt32.self)
                        let o = raw.loadUnaligned(fromByteOffset: cursor + 20, as: UInt64.self)
                        let n = raw.loadUnaligned(fromByteOffset: cursor + 28, as: UInt64.self)
                        return (k, b, dt, x, y, z, o, n)
                    }

            guard let kind = JangSpecFormat.TensorKind(rawValue: kindRaw) else {
                throw JangSpecError.invalidBlob("unknown tensor kind \(kindRaw)")
            }
            guard let dtype = JangSpecFormat.TensorDType(rawValue: dtypeRaw) else {
                throw JangSpecError.invalidBlob("unknown tensor dtype \(dtypeRaw)")
            }

            let bi = Int(bitsVal)
            if let prev = bitsSeen {
                if prev != bi {
                    throw JangSpecError.invalidBlob("mixed bits in one blob: \(prev) vs \(bi)")
                }
            } else {
                bitsSeen = bi
            }

            let rawDims = [Int(d0), Int(d1), Int(d2)]
            let dims = rawDims.filter { $0 != 0 }

            let start = payOff + Int(offRaw)
            let end = start + Int(nbRaw)
            guard end <= data.count else {
                throw JangSpecError.invalidBlob(
                    "tensor slice out of range: \(start)..<\(end) (blob size \(data.count))"
                )
            }
            let slice = data.subdata(in: start..<end)

            parsed.append(
                ExpertBlobTensor(
                    kind: kind,
                    dtype: dtype,
                    bits: bi,
                    dims: dims,
                    slice: slice
                )
            )
        }

        self.layerIdx = Int(layer)
        self.expertID = Int(expert)
        self.bits = bitsSeen ?? 0
        self.tensors = parsed
    }

    /// Convenience: find the first tensor matching a given kind and dtype.
    public func tensor(kind: JangSpecFormat.TensorKind, dtype: JangSpecFormat.TensorDType) -> Data? {
        for t in tensors where t.kind == kind && t.dtype == dtype {
            return t.slice
        }
        return nil
    }
}
