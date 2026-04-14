import Foundation

/// Errors raised by `SafetensorsV2File`.
public enum SafetensorsV2Error: Error, CustomStringConvertible {
    case truncated(URL, expected: Int, actual: Int)
    case malformedHeader(URL, reason: String)
    case unknownDType(String)
    case missingTensor(String)

    public var description: String {
        switch self {
        case .truncated(let url, let e, let a):
            return "safetensors: truncated \(url.lastPathComponent): expected \(e) bytes, got \(a)"
        case .malformedHeader(let url, let r):
            return "safetensors: malformed header in \(url.lastPathComponent): \(r)"
        case .unknownDType(let s):
            return "safetensors: unknown dtype \(s)"
        case .missingTensor(let name):
            return "safetensors: missing tensor '\(name)'"
        }
    }
}

/// Subset of safetensors dtypes that JANG v2 emits.
public enum SafetensorsDType: String, Sendable {
    case u32 = "U32"
    case f16 = "F16"
    case bf16 = "BF16"
    case f32 = "F32"

    public var byteWidth: Int {
        switch self {
        case .u32: return 4
        case .f16, .bf16: return 2
        case .f32: return 4
        }
    }
}

/// Metadata for one tensor inside a safetensors file.
public struct SafetensorsV2Info: Sendable, Equatable {
    public let name: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    public let dataOffset: Int    // absolute byte offset in the file
    public let dataLength: Int    // byte length of the tensor payload
}

/// A mmap'd JANG v2 safetensors file.
///
/// Construction parses the JSON header and records per-tensor metadata.
/// Actual bytes are returned as `Data` slices that share storage with the
/// mmap'd backing — zero-copy on Apple Silicon unified memory.
public final class SafetensorsV2File: @unchecked Sendable {
    public let url: URL
    public let tensorNames: [String]

    private let mapped: Data
    private let dataSectionStart: Int
    private let infoByName: [String: SafetensorsV2Info]

    public init(url: URL) throws {
        self.url = url
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        self.mapped = data

        guard data.count >= 8 else {
            throw SafetensorsV2Error.truncated(url, expected: 8, actual: data.count)
        }
        let headerSize: UInt64 = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
        }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerSize)
        guard data.count >= headerEnd else {
            throw SafetensorsV2Error.truncated(url, expected: headerEnd, actual: data.count)
        }
        self.dataSectionStart = headerEnd

        let headerJSON = data.subdata(in: headerStart..<headerEnd)
        let anyObj = try JSONSerialization.jsonObject(
            with: headerJSON, options: [.fragmentsAllowed]
        )
        guard let obj = anyObj as? [String: Any] else {
            throw SafetensorsV2Error.malformedHeader(url, reason: "top level is not an object")
        }

        var out: [String: SafetensorsV2Info] = [:]
        out.reserveCapacity(obj.count)
        var names: [String] = []

        for (name, raw) in obj {
            // safetensors may include a "__metadata__" key; skip it.
            if name == "__metadata__" { continue }

            guard let entry = raw as? [String: Any] else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) is not an object"
                )
            }
            guard let dtypeString = entry["dtype"] as? String else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) missing dtype"
                )
            }
            guard let dtype = SafetensorsDType(rawValue: dtypeString) else {
                throw SafetensorsV2Error.unknownDType(dtypeString)
            }
            guard let shapeAny = entry["shape"] as? [Any] else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) missing shape"
                )
            }
            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) shape contains non-integers"
                )
            }
            guard let offsetsAny = entry["data_offsets"] as? [Any], offsetsAny.count == 2 else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) missing data_offsets"
                )
            }
            let offsets = offsetsAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard offsets.count == 2 else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) data_offsets not integers"
                )
            }
            let absStart = headerEnd + offsets[0]
            let absEnd = headerEnd + offsets[1]
            guard absEnd <= data.count else {
                throw SafetensorsV2Error.truncated(url, expected: absEnd, actual: data.count)
            }
            out[name] = SafetensorsV2Info(
                name: name,
                dtype: dtype,
                shape: shape,
                dataOffset: absStart,
                dataLength: absEnd - absStart
            )
            names.append(name)
        }

        self.infoByName = out
        self.tensorNames = names.sorted()
    }

    public func info(for name: String) throws -> SafetensorsV2Info {
        guard let i = infoByName[name] else {
            throw SafetensorsV2Error.missingTensor(name)
        }
        return i
    }

    /// Return a `Data` slice pointing at the tensor bytes. Zero-copy over the
    /// mmap'd backing; the slice lifetime is bounded by the file object.
    public func bytes(for name: String) throws -> Data {
        let info = try info(for: name)
        return mapped.subdata(in: info.dataOffset..<(info.dataOffset + info.dataLength))
    }
}
