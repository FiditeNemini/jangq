import Foundation
import Metal
import JANGCore

/// 4-bit GEMV: y (fp32) = W (4-bit packed) @ x (fp16).
///
/// `W` is supplied as three `Data` slices following the JANG v2 convention:
///   qweight: uint32, shape (out, in/8)    — LSB-first 4-bit packing
///   scales : float16, shape (out, n_groups)
///   biases : float16, shape (out, n_groups)
/// with `n_groups = in / group_size` and `group_size` a power of 2.
///
/// The kernel uses one thread per output row. This is not a perf-tuned
/// implementation; Plan 4 exists to validate correctness against an MLX
/// reference, not to compete on tokens/sec.
public struct QuantizedMatmul4 {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jang_v2_quant_matmul_4bit_gemv")
    }

    public func run(
        qweight: Data,
        scales: Data,
        biases: Data,
        x: Data,
        inFeatures: Int,
        outFeatures: Int,
        groupSize: Int
    ) throws -> [Float] {
        let packedIn = inFeatures / 8
        let nGroups = inFeatures / groupSize
        let expectedQ = outFeatures * packedIn * 4
        let expectedS = outFeatures * nGroups * 2
        let expectedX = inFeatures * 2
        precondition(qweight.count == expectedQ, "qweight bytes \(qweight.count) != \(expectedQ)")
        precondition(scales.count == expectedS, "scales bytes \(scales.count) != \(expectedS)")
        precondition(biases.count == expectedS, "biases bytes \(biases.count) != \(expectedS)")
        precondition(x.count == expectedX, "x bytes \(x.count) != \(expectedX)")

        let dev = context.device

        let qBuf = try MetalBuffer.fromData(qweight, device: dev)
        let sBuf = try MetalBuffer.fromData(scales, device: dev)
        let bBuf = try MetalBuffer.fromData(biases, device: dev)
        let xBuf = try MetalBuffer.fromData(x, device: dev)
        let yBuf = try MetalBuffer.empty(bytes: outFeatures * MemoryLayout<Float>.stride, device: dev)

        var params = QuantMatmul4Params(
            in_features: UInt32(inFeatures),
            out_features: UInt32(outFeatures),
            group_size: UInt32(groupSize),
            n_groups: UInt32(nGroups),
            packed_in: UInt32(packedIn)
        )
        let paramBuf: MTLBuffer = try withUnsafeBytes(of: &params) { raw -> MTLBuffer in
            guard let base = raw.baseAddress else {
                throw JANGCoreMetalError.bufferAllocFailed("params")
            }
            guard let buf = dev.makeBuffer(
                bytes: base, length: raw.count, options: [.storageModeShared]
            ) else {
                throw JANGCoreMetalError.bufferAllocFailed("params \(raw.count)")
            }
            return buf
        }

        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw JANGCoreMetalError.dispatchFailed("no command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.dispatchFailed("no compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qBuf, offset: 0, index: 0)
        encoder.setBuffer(sBuf, offset: 0, index: 1)
        encoder.setBuffer(bBuf, offset: 0, index: 2)
        encoder.setBuffer(xBuf, offset: 0, index: 3)
        encoder.setBuffer(yBuf, offset: 0, index: 4)
        encoder.setBuffer(paramBuf, offset: 0, index: 5)

        let threadsPerThreadgroup = MTLSize(
            width: min(pipeline.maxTotalThreadsPerThreadgroup, 64),
            height: 1,
            depth: 1
        )
        let threadgroups = MTLSize(
            width: (outFeatures + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let err = commandBuffer.error {
            throw JANGCoreMetalError.dispatchFailed(String(describing: err))
        }

        var result = [Float](repeating: 0, count: outFeatures)
        let ptr = yBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<outFeatures {
            result[i] = ptr[i]
        }
        return result
    }
}

/// Must match Metal-side `struct QuantMatmul4Params`.
struct QuantMatmul4Params {
    var in_features: UInt32
    var out_features: UInt32
    var group_size: UInt32
    var n_groups: UInt32
    var packed_in: UInt32
}
