//
// JANGTQ 8-bit affine GEMV correctness tests.
//
// Builds tiny test cases with known weight values, dispatches the kernel,
// and compares against an in-Swift dequantize-and-multiply reference.
// No fixture files needed — all data generated procedurally per test.
//

import XCTest
import Metal
@testable import JANGCoreMetal

final class JANGTQAffine8Tests: XCTestCase {

    private func makeFloat32Buffer(_ count: Int, device: MTLDevice) -> MTLBuffer {
        device.makeBuffer(length: count * MemoryLayout<Float>.stride, options: .storageModeShared)!
    }

    private func makeHalfBuffer(_ values: [Float], device: MTLDevice) -> MTLBuffer {
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
        return (0..<count).map { p[$0] }
    }

    /// Reference dequant + matmul in pure Swift.
    /// W is stored as [out, packed_in] with 4 × 8-bit per uint32. Dequant
    /// per group: val = (q_int) * scale + bias.
    private func referenceMatmul(
        qweight: [UInt32], scales: [Float], biases: [Float],
        x: [Float], inFeatures: Int, outFeatures: Int, groupSize: Int
    ) -> [Float] {
        let nGroups = inFeatures / groupSize
        let packedIn = inFeatures / 4
        var y = [Float](repeating: 0, count: outFeatures)
        for o in 0..<outFeatures {
            var acc: Float = 0
            for g in 0..<nGroups {
                let scale = scales[o * nGroups + g]
                let bias  = biases[o * nGroups + g]
                let gStart = g * groupSize
                let wordsPerGroup = groupSize / 4
                for w in 0..<wordsPerGroup {
                    let iBase = gStart + w * 4
                    let word = qweight[o * packedIn + (gStart / 4) + w]
                    for k in 0..<4 {
                        let q = (word >> (k * 8)) & 0xFF
                        let dq = Float(q) * scale + bias
                        acc += dq * x[iBase + k]
                    }
                }
            }
            y[o] = acc
        }
        return y
    }

    /// Pack 8-bit values into uint32 words. `vals.count` must be a multiple of 4.
    private func pack8bit(_ vals: [UInt8]) -> [UInt32] {
        precondition(vals.count % 4 == 0)
        var out = [UInt32](repeating: 0, count: vals.count / 4)
        for i in 0..<out.count {
            let v0 = UInt32(vals[i*4 + 0])
            let v1 = UInt32(vals[i*4 + 1]) << 8
            let v2 = UInt32(vals[i*4 + 2]) << 16
            let v3 = UInt32(vals[i*4 + 3]) << 24
            out[i] = v0 | v1 | v2 | v3
        }
        return out
    }

    // MARK: - Tests

    func testAllZeros() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQAffine8Matmul(context: ctx)
        let inF = 64
        let outF = 16
        let gs = 64

        let qVals = [UInt8](repeating: 0, count: outF * inF)
        let qPacked = pack8bit(qVals)
        let scales = [Float](repeating: 1.0, count: outF * (inF / gs))
        let biases = [Float](repeating: 0.0, count: outF * (inF / gs))
        let x = (0..<inF).map { Float($0) }

        let qBuf = makeUInt32Buffer(qPacked, device: ctx.device)
        let sBuf = makeHalfBuffer(scales, device: ctx.device)
        let bBuf = makeHalfBuffer(biases, device: ctx.device)
        let xBuf = makeHalfBuffer(x, device: ctx.device)

        let yBuf = try kernel.run(
            qweightBuf: qBuf, scalesBuf: sBuf, biasesBuf: bBuf, xBuf: xBuf,
            inFeatures: inF, outFeatures: outF, groupSize: gs
        )
        let actual = readFloats(yBuf, count: outF)
        for v in actual { XCTAssertEqual(v, 0.0, accuracy: 1e-3) }
    }

    func testIdentityWeights() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQAffine8Matmul(context: ctx)
        // out=4, in=64, group_size=64, 1 group.
        // W[r] = [1, 1, ..., 1] (all q_int = 1, scale = 1, bias = 0) → val = 1
        // y[r] = sum(x) for every r.
        let inF = 64
        let outF = 4
        let gs = 64

        let qVals = [UInt8](repeating: 1, count: outF * inF)
        let qPacked = pack8bit(qVals)
        let scales = [Float](repeating: 1.0, count: outF * (inF / gs))
        let biases = [Float](repeating: 0.0, count: outF * (inF / gs))
        var x = [Float](repeating: 0, count: inF)
        for i in 0..<inF { x[i] = Float(i) * 0.01 }

        let qBuf = makeUInt32Buffer(qPacked, device: ctx.device)
        let sBuf = makeHalfBuffer(scales, device: ctx.device)
        let bBuf = makeHalfBuffer(biases, device: ctx.device)
        let xBuf = makeHalfBuffer(x, device: ctx.device)

        let yBuf = try kernel.run(
            qweightBuf: qBuf, scalesBuf: sBuf, biasesBuf: bBuf, xBuf: xBuf,
            inFeatures: inF, outFeatures: outF, groupSize: gs
        )
        let actual = readFloats(yBuf, count: outF)
        let expected = x.map { Float($0) }.reduce(0, +)  // = 20.16

        for v in actual {
            XCTAssertEqual(v, expected, accuracy: 1e-2,
                "expected sum(x)=\(expected), got \(v)")
        }
    }

    func testRandomVsReference() throws {
        let ctx = try MetalContext()
        let kernel = try JANGTQAffine8Matmul(context: ctx)
        let inF = 128
        let outF = 24
        let gs = 64
        let nGroups = inF / gs  // 2

        // Deterministic pseudo-random qweight + scales + biases + x
        var rng = UInt64(0xCAFE_F00D)
        func nextU64() -> UInt64 {
            rng = rng &* 6364136223846793005 &+ 1442695040888963407
            return rng
        }
        var qVals = [UInt8](repeating: 0, count: outF * inF)
        for i in 0..<qVals.count { qVals[i] = UInt8(nextU64() & 0xFF) }
        var scales = [Float](repeating: 0, count: outF * nGroups)
        for i in 0..<scales.count { scales[i] = Float(nextU64() & 0xFF) / 1024.0 - 0.125 }
        var biases = [Float](repeating: 0, count: outF * nGroups)
        for i in 0..<biases.count { biases[i] = Float(nextU64() & 0xFF) / 2048.0 - 0.0625 }
        var x = [Float](repeating: 0, count: inF)
        for i in 0..<inF { x[i] = Float(Int32(truncatingIfNeeded: nextU64()) % 100) / 100.0 }

        let qPacked = pack8bit(qVals)
        let qBuf = makeUInt32Buffer(qPacked, device: ctx.device)
        let sBuf = makeHalfBuffer(scales, device: ctx.device)
        let bBuf = makeHalfBuffer(biases, device: ctx.device)
        let xBuf = makeHalfBuffer(x, device: ctx.device)

        let yBuf = try kernel.run(
            qweightBuf: qBuf, scalesBuf: sBuf, biasesBuf: bBuf, xBuf: xBuf,
            inFeatures: inF, outFeatures: outF, groupSize: gs
        )
        let actual = readFloats(yBuf, count: outF)

        // Reference uses the half-rounded scales/biases/x to match what the
        // kernel actually sees (we wrote half buffers).
        let scalesH = scales.map { Float(Float16($0)) }
        let biasesH = biases.map { Float(Float16($0)) }
        let xH = x.map { Float(Float16($0)) }
        let expected = referenceMatmul(
            qweight: qPacked, scales: scalesH, biases: biasesH, x: xH,
            inFeatures: inF, outFeatures: outF, groupSize: gs
        )

        var maxAbs: Float = 0
        for i in 0..<outF { maxAbs = max(maxAbs, abs(actual[i] - expected[i])) }
        XCTAssertLessThan(maxAbs, 1e-3,
            "8-bit GEMV max abs diff = \(maxAbs)\nactual:   \(actual)\nexpected: \(expected)")
    }
}
