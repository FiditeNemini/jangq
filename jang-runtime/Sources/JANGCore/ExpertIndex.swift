import Foundation

/// One row of `experts.jsidx`. Mirrors Python `ExpertIndexEntry`.
public struct ExpertIndexEntry: Sendable, Equatable {
    public let layerIdx: Int
    public let expertID: Int
    public let fileID: Int       // index into experts-NNNNN.bin
    public let offset: Int       // absolute byte offset in that file
    public let nbytes: Int       // total blob length (aligned)
}

/// Loaded `experts.jsidx` — flat array of entries plus layer/expert counts.
///
/// Lookup is O(1) via a layer-major dictionary keyed by `(layer, expert)`.
/// The backing file is read once at init; this struct holds the decoded
/// entries only, not the file handle.
public struct ExpertIndex: Sendable {
    public let nLayers: Int
    public let nExpertsPerLayer: Int
    public let entries: [ExpertIndexEntry]
    private let byKey: [Int: ExpertIndexEntry]

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= JangSpecFormat.indexHeaderSize else {
            throw JangSpecError.truncated(
                url, expected: JangSpecFormat.indexHeaderSize, actual: data.count
            )
        }

        // Read header. Use loadUnaligned — the on-disk layout is packed
        // little-endian and offsets are not natural Swift alignment.
        let header = data.withUnsafeBytes { raw -> (UInt32, UInt16, UInt32, UInt32, UInt64) in
            let magic = raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
            let version = raw.loadUnaligned(fromByteOffset: 4, as: UInt16.self)
            // 2 bytes pad at offset 6
            let nL = raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
            let nE = raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
            let n = raw.loadUnaligned(fromByteOffset: 16, as: UInt64.self)
            return (magic, version, nL, nE, n)
        }

        let (magic, version, nLRaw, nERaw, nRaw) = header
        guard magic == JangSpecFormat.indexMagic else {
            throw JangSpecError.badMagic(
                expected: JangSpecFormat.indexMagic, actual: magic, at: url
            )
        }
        guard version == 1 else {
            throw JangSpecError.unsupportedVersion(
                field: "index", value: Int(version), supported: 1
            )
        }

        let nEntries = Int(nRaw)
        let expectedSize = JangSpecFormat.indexHeaderSize + nEntries * JangSpecFormat.indexEntrySize
        guard data.count >= expectedSize else {
            throw JangSpecError.truncated(url, expected: expectedSize, actual: data.count)
        }

        var parsed: [ExpertIndexEntry] = []
        parsed.reserveCapacity(nEntries)

        data.withUnsafeBytes { raw in
            var cursor = JangSpecFormat.indexHeaderSize
            for _ in 0..<nEntries {
                let layer = raw.loadUnaligned(fromByteOffset: cursor + 0, as: UInt32.self)
                let expert = raw.loadUnaligned(fromByteOffset: cursor + 4, as: UInt32.self)
                let fileID = raw.loadUnaligned(fromByteOffset: cursor + 8, as: UInt16.self)
                // 2 bytes pad at offset cursor + 10
                let offset = raw.loadUnaligned(fromByteOffset: cursor + 12, as: UInt64.self)
                let nbytes = raw.loadUnaligned(fromByteOffset: cursor + 20, as: UInt64.self)
                parsed.append(
                    ExpertIndexEntry(
                        layerIdx: Int(layer),
                        expertID: Int(expert),
                        fileID: Int(fileID),
                        offset: Int(offset),
                        nbytes: Int(nbytes)
                    )
                )
                cursor += JangSpecFormat.indexEntrySize
            }
        }

        self.nLayers = Int(nLRaw)
        self.nExpertsPerLayer = Int(nERaw)
        self.entries = parsed

        var map: [Int: ExpertIndexEntry] = [:]
        map.reserveCapacity(parsed.count)
        for e in parsed {
            map[Self.key(layer: e.layerIdx, expert: e.expertID)] = e
        }
        self.byKey = map
    }

    /// Constant-time lookup by (layer, expert).
    public func entry(layer: Int, expert: Int) throws -> ExpertIndexEntry {
        guard let e = byKey[Self.key(layer: layer, expert: expert)] else {
            throw JangSpecError.missingEntry(layer: layer, expert: expert)
        }
        return e
    }

    @inline(__always)
    private static func key(layer: Int, expert: Int) -> Int {
        // Safe up to ~2^31 layers and ~2^32 experts, which will never happen.
        return (layer << 32) | expert
    }
}
