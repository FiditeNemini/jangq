//
// Self-contained tests for the JANGTQ Metal kernels.
// No fixture files needed — uses analytically-computed reference values
// for tiny inputs so each kernel's dataflow can be verified end-to-end.
//
// Coverage:
//   1. Hadamard butterfly correctness (single-block + multi-block)
//      against an in-Swift Walsh-Hadamard reference.
//
//   2. Fused gate+up+SwiGLU: with all-zeros packed weights and a
//      codebook of [a, b, c, d], every weight resolves to a (codebook[0]).
//      Output is then norms_g * sum(x_rot)*a, then SiLU(...) * norms_u * sum(x_rot)*a.
//      With norms=1 and a=1, output = SiLU(sum(x_rot)) * sum(x_rot).
//
//   3. Gather TQ matmul: same trick, all packed=0, codebook[0]=1, norms=1
//      → output[r] = sum(x_rot[k, :]) for every output row r.
//
// These tests do not depend on Python, MLX, or any model file. They run
// in well under a second and verify the full dispatch path
// (MetalContext load → pipeline create → buffer alloc → encode → wait).
//

import XCTest
import Metal
@testable import JANGCoreMetal

final class JANGTQMatmulTests: XCTestCase {

    // MARK: Helpers

    private func makeFloatBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer {
        let n = values.count * MemoryLayout<Float>.stride
        let buf = device.makeBuffer(length: n, options: .storageModeShared)!
        let p = buf.contents().bindMemory(to: Float.self, capacity: values.count)
        for i in 0..<values.count { p[i] = values[i] }
        return buf
    }

    private func makeHalfBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer {
        // half = float16 = 2 bytes. We store via Float16 native type.
        let n = values.count * MemoryLayout<Float16>.stride
        let buf = device.makeBuffer(length: n, options: .storageModeShared)!
        let p = buf.contents().bindMemory(to: Float16.self, capacity: values.count)
        for i in 0..<values.count { p[i] = Float16(values[i]) }
        return buf
    }

    private func makeUInt32Buffer(_ values: [UInt32], device: MTLDevice) -> MTLBuffer {
        let n = values.count * MemoryLayout<UInt32>.stride
        let buf = device.makeBuffer(length: n, options: .storageModeShared)!
        let p = buf.contents().bindMemory(to: UInt32.self, capacity: values.count)
        for i in 0..<values.count { p[i] = values[i] }
        return buf
    }

    private func readFloats(_ buf: MTLBuffer, count: Int) -> [Float] {
        let p = buf.contents().bindMemory(to: Float.self, capacity: count)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count { out[i] = p[i] }
        return out
    }

    /// In-Swift Walsh-Hadamard transform with the same normalization
    /// (1/sqrt(d) per block) as the Metal kernel.
    private func hadamardReference(_ x: [Float]) -> [Float] {
        let d = x.count
        var v = x
        var h = 1
        while h < d {
            var i = 0
            while i < d {
                for j in i..<(i + h) {
                    let a = v[j]
                    let b = v[j + h]
                    v[j] = a + b
                    v[j + h] = a - b
                }
                i += h * 2
            }
            h *= 2
        }
        let norm = 1.0 / Float(d).squareRoot()
        return v.map { $0 * norm }
    }

    // MARK: - Hadamard

    func testHadamardSingleBlockPow2() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQHadamard(context: ctx)
        let dim = 64

        // Random-ish input
        var x: [Float] = []
        for i in 0..<dim {
            x.append(Float(i % 7) - 3.0)
        }
        let signs = [Float](repeating: 1.0, count: dim)

        let xBuf = makeHalfBuffer(x, device: ctx.device)
        let signsBuf = makeFloatBuffer(signs, device: ctx.device)
        let outBuf = try kernel.run(xBuf: xBuf, signsBuf: signsBuf, batch: 1, dim: dim)
        let actual = readFloats(outBuf, count: dim)

        // half-precision input, fp32 output → reference also from half-rounded x
        let xh = x.map { Float(Float16($0)) }
        let expected = hadamardReference(xh)

        var maxAbs: Float = 0
        for i in 0..<dim { maxAbs = max(maxAbs, abs(actual[i] - expected[i])) }
        XCTAssertLessThan(maxAbs, 1e-3, "Hadamard d=\(dim) max diff = \(maxAbs)")
    }

    func testHadamardMultiBlockNonPow2() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQHadamard(context: ctx)
        // 96 = 64 + 32 (multi-block path). Each block normalized independently.
        let dim = 96

        var x: [Float] = []
        for i in 0..<dim {
            x.append(Float((i * 13) % 11) - 5.0)
        }
        let signs = [Float](repeating: 1.0, count: dim)

        let xBuf = makeHalfBuffer(x, device: ctx.device)
        let signsBuf = makeFloatBuffer(signs, device: ctx.device)
        let outBuf = try kernel.run(xBuf: xBuf, signsBuf: signsBuf, batch: 1, dim: dim)
        let actual = readFloats(outBuf, count: dim)

        // Reference: each block independently butterflied + normalized
        let xh = x.map { Float(Float16($0)) }
        let block0 = Array(xh[0..<64])
        let block1 = Array(xh[64..<96])
        var expected = hadamardReference(block0)
        expected.append(contentsOf: hadamardReference(block1))

        var maxAbs: Float = 0
        for i in 0..<dim { maxAbs = max(maxAbs, abs(actual[i] - expected[i])) }
        XCTAssertLessThan(maxAbs, 1e-3, "Hadamard multi-block d=\(dim) max diff = \(maxAbs)")
    }

    // MARK: - Fused gate+up+SwiGLU

    func testFusedGateUpSwiGLUIdentity() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQFusedGateUpSwiGLU(context: ctx)

        // Tiny test: K=2 experts, in=64, out=32. All packed = 0 (every code is 0).
        // With codebook = [1, 0, 0, 0]: every weight resolves to 1.
        // With norms_g = norms_u = 1: gate = sum(x_rot), up = sum(x_rot).
        // Output = SiLU(gate) * up = SiLU(s) * s where s = sum(x_rot).
        let K = 2
        let inF = 64
        let outF = 32
        let bits = 2
        let nExperts = 4

        let valsPerU32 = 32 / bits
        let packedCols = (inF + valsPerU32 - 1) / valsPerU32

        // x_rot = sequential floats so sum is well-known
        var xRot = [Float](repeating: 0, count: inF)
        for i in 0..<inF { xRot[i] = Float(i) * 0.01 - 0.3 }
        let s: Float = xRot.reduce(0, +)

        let xRotBuf = makeFloatBuffer(xRot, device: ctx.device)
        let packedZeros = [UInt32](repeating: 0, count: nExperts * outF * packedCols)
        let pgBuf = makeUInt32Buffer(packedZeros, device: ctx.device)
        let puBuf = makeUInt32Buffer(packedZeros, device: ctx.device)
        let normsOnes = [Float](repeating: 1.0, count: nExperts * outF)
        let ngBuf = makeHalfBuffer(normsOnes, device: ctx.device)
        let nuBuf = makeHalfBuffer(normsOnes, device: ctx.device)
        let codebook: [Float] = [1.0, 0.0, 0.0, 0.0]
        let cbBuf = makeFloatBuffer(codebook, device: ctx.device)
        let indices: [UInt32] = [0, 1]
        let idxBuf = makeUInt32Buffer(indices, device: ctx.device)

        let outBuf = try kernel.run(
            xRotBuf: xRotBuf,
            packedGateBuf: pgBuf, normsGateBuf: ngBuf,
            packedUpBuf: puBuf, normsUpBuf: nuBuf,
            codebookBuf: cbBuf,
            rhsIndicesBuf: idxBuf,
            K: K, inFeatures: inF, outFeatures: outF, bits: bits
        )

        let actual = readFloats(outBuf, count: K * outF)
        let expected: Float = (s / (1.0 + Float(exp(-Double(s))))) * s

        var maxAbs: Float = 0
        for v in actual { maxAbs = max(maxAbs, abs(v - expected)) }
        XCTAssertLessThan(maxAbs, 1e-2, "fused gate+up+swiglu identity max diff = \(maxAbs), expected = \(expected)")
    }

    // MARK: - Gather TQ matmul

    func testGatherIdentity() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQGatherMatmul(context: ctx)

        // Per-row mode: K=2 rows of x_rot, in=64, out=32. Every packed=0,
        // codebook[0]=1, norms=1 → out[k, r] = sum(x_rot[k, :]) for every r.
        let K = 2
        let inF = 64
        let outF = 32
        let bits = 2
        let nExperts = 4

        let valsPerU32 = 32 / bits
        let packedCols = (inF + valsPerU32 - 1) / valsPerU32

        var xRot = [Float](repeating: 0, count: K * inF)
        for k in 0..<K {
            for i in 0..<inF {
                xRot[k * inF + i] = Float(i + k * 100) * 0.005
            }
        }
        let xRotBuf = makeFloatBuffer(xRot, device: ctx.device)

        let packedZeros = [UInt32](repeating: 0, count: nExperts * outF * packedCols)
        let packedBuf = makeUInt32Buffer(packedZeros, device: ctx.device)
        let normsOnes = [Float](repeating: 1.0, count: nExperts * outF)
        let normsBuf = makeHalfBuffer(normsOnes, device: ctx.device)
        let codebook: [Float] = [1.0, 0.0, 0.0, 0.0]
        let cbBuf = makeFloatBuffer(codebook, device: ctx.device)
        let indices: [UInt32] = [0, 1]
        let idxBuf = makeUInt32Buffer(indices, device: ctx.device)

        let outBuf = try kernel.run(
            xRotBuf: xRotBuf,
            packedBuf: packedBuf,
            normsBuf: normsBuf,
            codebookBuf: cbBuf,
            rhsIndicesBuf: idxBuf,
            K: K, inFeatures: inF, outFeatures: outF, bits: bits
        )

        let actual = readFloats(outBuf, count: K * outF)

        for k in 0..<K {
            var s: Float = 0
            for i in 0..<inF { s += xRot[k * inF + i] }
            for r in 0..<outF {
                let v = actual[k * outF + r]
                XCTAssertLessThan(abs(v - s), 1e-3,
                    "gather row k=\(k) r=\(r): got \(v) expected \(s)")
            }
        }
    }

    // MARK: - Bundle pipeline creation

    func testKernelsBundleLoadsAllPipelines() throws {
        let ctx = try MetalContext()
        let kernels = try JANGTQKernels(context: ctx)
        XCTAssertNotNil(kernels.hadamard.pipeline)
        XCTAssertNotNil(kernels.fusedGateUp.pipeline)
        XCTAssertNotNil(kernels.gather.pipeline)
    }
}
