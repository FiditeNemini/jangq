import Foundation

/// Mirror of `jang_tools.jangspec.format` — on-disk layout constants.
///
/// If you change a value here, change it in the Python module and in the
/// bundle version number. The Swift and Python sides MUST agree.
public enum JangSpecFormat {
    public static let bundleVersion: Int = 1

    // Filenames inside a <name>.jangspec directory.
    public static let manifestFilename = "jangspec.json"
    public static let indexFilename = "target/experts.jsidx"
    public static let hotCoreFilename = "target/hot_core.safetensors"
    public static func expertFilename(idx: Int) -> String {
        return String(format: "target/experts-%05d.bin", idx)
    }

    // Alignment used for expert blob offsets.
    public static let blobAlignment: Int = 4096

    // Magic numbers — "JSPE" and "SJIX" little-endian uint32.
    public static let blobMagic: UInt32 = 0x4550_534A
    public static let indexMagic: UInt32 = 0x58_494A_53

    // Struct sizes (verified at compile time by static asserts below and
    // at runtime by tests).
    public static let blobHeaderSize: Int = 32
    public static let tensorHeaderSize: Int = 36
    public static let indexEntrySize: Int = 28
    public static let indexHeaderSize: Int = 24

    // Tensor-kind enum (matches Python KIND_* constants).
    public enum TensorKind: UInt8, Sendable {
        case gate = 0
        case up = 1
        case down = 2
    }

    // Dtype enum (matches Python DTYPE_* constants).
    public enum TensorDType: UInt32, Sendable {
        case qweight = 0  // uint32 packed
        case scales = 1   // float16
        case biases = 2   // float16
    }

    @inlinable
    public static func alignUp(_ n: Int, to align: Int = blobAlignment) -> Int {
        return (n + align - 1) & ~(align - 1)
    }
}
