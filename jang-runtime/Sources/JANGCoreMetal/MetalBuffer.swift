import Foundation
import Metal

/// Helpers for turning `Data` slices into `MTLBuffer`s.
///
/// Plan 4 uses plain `makeBuffer(bytes:length:options:)` which copies into
/// a shared-storage buffer. A later plan may switch to `bytesNoCopy` for
/// zero-copy against mmap-backed `Data` when alignment allows; for tiny
/// fixtures the copy cost is negligible.
public enum MetalBuffer {
    public static func fromData(_ data: Data, device: MTLDevice) throws -> MTLBuffer {
        return try data.withUnsafeBytes { raw -> MTLBuffer in
            guard let base = raw.baseAddress, raw.count > 0 else {
                throw JANGCoreMetalError.bufferAllocFailed("empty Data")
            }
            guard let buf = device.makeBuffer(
                bytes: base,
                length: raw.count,
                options: [.storageModeShared]
            ) else {
                throw JANGCoreMetalError.bufferAllocFailed("\(raw.count) bytes")
            }
            return buf
        }
    }

    public static func empty(bytes: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: bytes, options: [.storageModeShared]) else {
            throw JANGCoreMetalError.bufferAllocFailed("empty \(bytes) bytes")
        }
        memset(buf.contents(), 0, bytes)
        return buf
    }
}
