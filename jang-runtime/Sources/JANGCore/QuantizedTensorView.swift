import Foundation

/// A typed handle over one JANG v2 quantized tensor.
///
/// Holds three `Data` slices (qweight, scales, biases) that reference the
/// mmap'd backing of a safetensors file, plus the inferred bit width and
/// logical shape. No Metal buffers yet — that's Plan 4. The slices are
/// zero-copy views on the mmap.
public struct QuantizedTensorView: Sendable {
    public let baseName: String
    public let bits: Int
    public let groupSize: Int
    public let inFeatures: Int
    public let outFeatures: Int
    public let qweightShape: [Int]    // e.g. [out, packedIn] or [E, out, packedIn]
    public let scalesShape: [Int]     // e.g. [out, nGroups] or [E, out, nGroups]
    public let qweight: Data          // U32 payload
    public let scales: Data           // F16 payload
    public let biases: Data           // F16 payload

    public var isExpertStacked: Bool {
        return qweightShape.count == 3
    }
}

/// A typed handle over one non-quantized tensor (norms, biases, embeddings).
public struct RawTensorView: Sendable {
    public let name: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    public let bytes: Data
}
