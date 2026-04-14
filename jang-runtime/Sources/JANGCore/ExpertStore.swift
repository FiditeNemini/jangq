import Foundation

/// Lazy mmap over a bundle's `experts-NNNNN.bin` shard files.
///
/// Plan 2 uses `Data(contentsOf:options:.mappedIfSafe)` for zero-copy
/// access on Apple Silicon. Plan 3 will add a parallel `MTLIOCommandQueue`
/// path for direct-to-GPU reads; the `load` signature here is the contract
/// that streaming path will satisfy.
public final class ExpertStore: @unchecked Sendable {
    public let bundleURL: URL
    public let index: ExpertIndex

    private var mapped: [Int: Data] = [:]
    private let lock = NSLock()

    public init(bundleURL: URL, index: ExpertIndex) {
        self.bundleURL = bundleURL
        self.index = index
    }

    /// Load one expert by (layer, expert) — returns a parsed `ExpertBlob`.
    /// Throws if the shard file is missing or the blob fails validation.
    public func load(layer: Int, expert: Int) throws -> ExpertBlob {
        let entry = try index.entry(layer: layer, expert: expert)
        let mm = try mapFile(id: entry.fileID)
        guard entry.offset + entry.nbytes <= mm.count else {
            throw JangSpecError.truncated(
                bundleURL.appendingPathComponent(JangSpecFormat.expertFilename(idx: entry.fileID)),
                expected: entry.offset + entry.nbytes,
                actual: mm.count
            )
        }
        let slice = mm.subdata(in: entry.offset..<(entry.offset + entry.nbytes))
        return try ExpertBlob(rawBytes: slice)
    }

    private func mapFile(id: Int) throws -> Data {
        lock.lock()
        defer { lock.unlock() }
        if let hit = mapped[id] {
            return hit
        }
        let url = bundleURL.appendingPathComponent(JangSpecFormat.expertFilename(idx: id))
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw JangSpecError.fileMissing(url)
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        mapped[id] = data
        return data
    }
}
