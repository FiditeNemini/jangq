import Foundation

/// Errors raised by JANGCore when parsing a .jangspec bundle.
public enum JangSpecError: Error, CustomStringConvertible {
    case fileMissing(URL)
    case badMagic(expected: UInt32, actual: UInt32, at: URL)
    case unsupportedVersion(field: String, value: Int, supported: Int)
    case truncated(URL, expected: Int, actual: Int)
    case missingEntry(layer: Int, expert: Int)
    case invalidManifest(String)
    case invalidBlob(String)

    public var description: String {
        switch self {
        case .fileMissing(let url):
            return "jangspec: file missing: \(url.path)"
        case .badMagic(let expected, let actual, let url):
            return String(
                format: "jangspec: bad magic 0x%08x (expected 0x%08x) in %@",
                actual, expected, url.lastPathComponent
            )
        case .unsupportedVersion(let field, let value, let supported):
            return "jangspec: unsupported \(field) version \(value), this build supports \(supported)"
        case .truncated(let url, let expected, let actual):
            return "jangspec: truncated file \(url.lastPathComponent): expected \(expected) bytes, got \(actual)"
        case .missingEntry(let layer, let expert):
            return "jangspec: no index entry for (layer=\(layer), expert=\(expert))"
        case .invalidManifest(let msg):
            return "jangspec: invalid manifest: \(msg)"
        case .invalidBlob(let msg):
            return "jangspec: invalid expert blob: \(msg)"
        }
    }
}
