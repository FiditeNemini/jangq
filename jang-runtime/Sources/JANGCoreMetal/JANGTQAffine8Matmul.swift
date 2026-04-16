//
// Swift binding for the JANGTQ 8-bit MLX-format quantized GEMV.
// Created by Jinho Jang (eric@jangq.ai).
//
// Used by the JANGTQ Swift inference engine for any non-MoE 8-bit
// quantized linear: attention projections, embed_tokens, lm_head.
//
// Pre-allocate a `JANGTQAffine8Matmul` once per model — pipeline
// creation is cheap to amortize but not free per call.
//

import Foundation
import Metal

/// Parameters struct mirroring the Metal `QuantMatmul8Params`.
public struct JANGTQAffine8Params {
    public var inFeatures: UInt32
    public var outFeatures: UInt32
    public var groupSize: UInt32
    public var nGroups: UInt32
    public var packedIn: UInt32
}

/// 8-bit MLX-format quantized GEMV pipeline.
///
/// Provides `run(...)` that allocates an output buffer and dispatches the
/// kernel synchronously. For tight inference loops the caller should
/// re-use a pre-allocated output buffer via `runInto(...)`.
public struct JANGTQAffine8Matmul {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        // Default to the SIMD-reduced kernel — 10-30× faster than the
        // one-thread-per-row baseline for typical attention shapes. The naive
        // kernel is still available via `init(context:useNaive:)` for testing.
        self.pipeline = try context.pipeline(functionNamed: "jangtq_quant_matmul_8bit_gemv_simd")
    }

    public init(context: MetalContext, useNaive: Bool) throws {
        self.context = context
        let name = useNaive
            ? "jangtq_quant_matmul_8bit_gemv"
            : "jangtq_quant_matmul_8bit_gemv_simd"
        self.pipeline = try context.pipeline(functionNamed: name)
    }

    /// One-shot dispatch — allocates output buffer, runs, waits.
    public func run(
        qweightBuf: MTLBuffer,    // (out_features, in_features/4) uint32
        scalesBuf: MTLBuffer,     // (out_features, in_features/group_size) half
        biasesBuf: MTLBuffer,     // (out_features, in_features/group_size) half
        xBuf: MTLBuffer,          // (in_features,) half
        inFeatures: Int,
        outFeatures: Int,
        groupSize: Int = 64
    ) throws -> MTLBuffer {
        let outBytes = outFeatures * MemoryLayout<Float>.stride
        guard let yBuf = context.device.makeBuffer(length: outBytes, options: .storageModeShared) else {
            throw JANGCoreMetalError.libraryLoadFailed("affine8 out buffer alloc failed")
        }
        try runInto(
            qweightBuf: qweightBuf, scalesBuf: scalesBuf, biasesBuf: biasesBuf,
            xBuf: xBuf, yBuf: yBuf,
            inFeatures: inFeatures, outFeatures: outFeatures, groupSize: groupSize
        )
        return yBuf
    }

    /// Dispatch into a caller-owned output buffer. Output buffer must have
    /// at least `outFeatures * 4` bytes (fp32).
    public func runInto(
        qweightBuf: MTLBuffer,
        scalesBuf: MTLBuffer,
        biasesBuf: MTLBuffer,
        xBuf: MTLBuffer,
        yBuf: MTLBuffer,
        inFeatures: Int,
        outFeatures: Int,
        groupSize: Int = 64
    ) throws {
        guard let cb = context.queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.libraryLoadFailed("command encoder alloc failed")
        }
        encode(into: enc, qweightBuf: qweightBuf, scalesBuf: scalesBuf,
               biasesBuf: biasesBuf, xBuf: xBuf, yBuf: yBuf,
               inFeatures: inFeatures, outFeatures: outFeatures, groupSize: groupSize)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }

    /// Encode into a caller-managed encoder. Caller owns the cb lifecycle.
    /// Uses the SIMD-reduced kernel (32 threads per output row).
    public func encode(
        into enc: MTLComputeCommandEncoder,
        qweightBuf: MTLBuffer,
        scalesBuf: MTLBuffer,
        biasesBuf: MTLBuffer,
        xBuf: MTLBuffer,
        yBuf: MTLBuffer,
        inFeatures: Int,
        outFeatures: Int,
        groupSize: Int = 64
    ) {
        precondition(inFeatures % groupSize == 0)
        precondition(inFeatures % 4 == 0)

        var params = JANGTQAffine8Params(
            inFeatures: UInt32(inFeatures),
            outFeatures: UInt32(outFeatures),
            groupSize: UInt32(groupSize),
            nGroups: UInt32(inFeatures / groupSize),
            packedIn: UInt32(inFeatures / 4)
        )
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(qweightBuf, offset: 0, index: 0)
        enc.setBuffer(scalesBuf,  offset: 0, index: 1)
        enc.setBuffer(biasesBuf,  offset: 0, index: 2)
        enc.setBuffer(xBuf,       offset: 0, index: 3)
        enc.setBuffer(yBuf,       offset: 0, index: 4)
        withUnsafeBytes(of: &params) { raw in
            enc.setBytes(raw.baseAddress!, length: raw.count, index: 5)
        }
        // SIMD kernel: grid = out_features × 32 threads, threadgroup = 256 (8 simds = 8 rows per TG)
        let totalThreads = outFeatures * 32
        let tgWidth = min(256, totalThreads)
        enc.dispatchThreads(MTLSizeMake(totalThreads, 1, 1),
                            threadsPerThreadgroup: MTLSizeMake(tgWidth, 1, 1))
    }
}
