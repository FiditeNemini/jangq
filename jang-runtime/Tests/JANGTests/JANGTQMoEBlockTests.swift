//
// End-to-end JANGTQ MoE block test — loads synthetic model from disk,
// constructs the block via the loader, runs forward, verifies output shape
// and that the kernels actually executed without error.
//
// This is the integration test that proves the whole pipeline works:
//   safetensors → loader → MTLBuffers → MoE block → kernel dispatch → output.
//

import Foundation
import XCTest
@testable import JANG
import JANGCoreMetal
import Metal

final class JANGTQMoEBlockTests: XCTestCase {

    /// Build a tiny 1-layer 2-expert MiniMax-style JANGTQ model on disk.
    /// Returns (path, hidden_size, intermediate_size, K).
    private func buildModel() throws -> (URL, Int, Int, Int) {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_block_test_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        let hidden = 64
        let inter = 32
        let nExperts = 2
        let K = 2

        // config.json
        let config: [String: Any] = [
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": 128,
            "moe_intermediate_size": inter,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 1024,
            "num_local_experts": nExperts,
            "num_experts_per_tok": K,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-5,
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: dir.appendingPathComponent("config.json"))

        // jang_config.json
        let jangCfg: [String: Any] = [
            "version": 2, "weight_format": "mxtq",
            "profile": "JANGTQ_2L",
            "source_model": ["name": "MiniMax-Test", "architecture": "minimax_m2"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2, "attention": 8],
            "quantization": ["group_size": 64, "bits_default": 2, "method": "affine+mxtq"],
        ]
        try JSONSerialization.data(withJSONObject: jangCfg)
            .write(to: dir.appendingPathComponent("jang_config.json"))

        // Per-expert weights. Use packed=0 across the board so the kernel
        // looks up codebook[0] for every weight — analytically tractable.
        let valsPerU32 = 16  // 2-bit
        let packedInGate = hidden / valsPerU32   // 64 / 16 = 4
        let packedInDown = inter  / valsPerU32   // 32 / 16 = 2

        var tensors: [(String, String, [Int], Data)] = []
        for e in 0..<nExperts {
            // gate (w1): out=inter, in=hidden
            tensors.append(contentsOf: makeTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w1",
                outFeatures: inter, packedIn: packedInGate, fillValue: 0
            ))
            // down (w2): out=hidden, in=inter
            tensors.append(contentsOf: makeTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w2",
                outFeatures: hidden, packedIn: packedInDown, fillValue: 0
            ))
            // up (w3): out=inter, in=hidden
            tensors.append(contentsOf: makeTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w3",
                outFeatures: inter, packedIn: packedInGate, fillValue: 0
            ))
        }
        try writeShard(
            url: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: tensors
        )

        // Sidecar: codebook[0] = 1, others = 0; signs all +1 → identity-ish.
        // With packed=0 everywhere, every weight resolves to codebook[0] = 1.
        // Norms are all 1 (set in makeTriplet below).
        // So gate = up = sum(x_rot), x_act = SiLU(s) * s, down = sum(x_act_rot).
        let signsHidden = [Float](repeating: 1.0, count: hidden)
        let signsInter  = [Float](repeating: 1.0, count: inter)
        let cbHidden: [Float] = [1.0, 0.0, 0.0, 0.0]
        let cbInter:  [Float] = [1.0, 0.0, 0.0, 0.0]
        let sidecar: [(String, String, [Int], Data)] = [
            ("signs.\(hidden).42", "F32", [hidden], floatsToData(signsHidden)),
            ("signs.\(inter).42",  "F32", [inter],  floatsToData(signsInter)),
            ("codebook.\(hidden).2", "F32", [4], floatsToData(cbHidden)),
            ("codebook.\(inter).2",  "F32", [4], floatsToData(cbInter)),
        ]
        try writeShard(
            url: dir.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: sidecar
        )

        return (dir, hidden, inter, K)
    }

    /// Build (.tq_packed, .tq_norms, .tq_bits) tensors filled with `fillValue`
    /// in packed and 1.0 in norms.
    private func makeTriplet(
        base: String, outFeatures: Int, packedIn: Int, fillValue: UInt32
    ) -> [(String, String, [Int], Data)] {
        var packedBytes = Data(count: outFeatures * packedIn * 4)
        packedBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            for i in 0..<(outFeatures * packedIn) { p[i] = fillValue }
        }
        var normsBytes = Data(count: outFeatures * 2)
        normsBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for r in 0..<outFeatures { p[r] = 1.0 }
        }
        var bitsValue: Int32 = 2
        let bitsBytes = Data(bytes: &bitsValue, count: 4)
        return [
            (base + ".tq_packed", "U32", [outFeatures, packedIn], packedBytes),
            (base + ".tq_norms",  "F16", [outFeatures],           normsBytes),
            (base + ".tq_bits",   "I32", [],                      bitsBytes),
        ]
    }

    private func floatsToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 4)
        for v in values {
            var x = v
            bytes.append(Data(bytes: &x, count: 4))
        }
        return bytes
    }

    private func writeShard(url: URL, tensors: [(String, String, [Int], Data)]) throws {
        var headerDict: [String: Any] = [:]
        var offset = 0
        for (name, dtype, shape, data) in tensors {
            headerDict[name] = [
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, offset + data.count],
            ] as [String: Any]
            offset += data.count
        }
        let headerData = try JSONSerialization.data(withJSONObject: headerDict)
        let paddedHeaderSize = (headerData.count + 7) & ~7
        let padBytes = paddedHeaderSize - headerData.count

        var fileData = Data()
        var sizeLE: UInt64 = UInt64(paddedHeaderSize)
        fileData.append(Data(bytes: &sizeLE, count: 8))
        fileData.append(headerData)
        fileData.append(Data(repeating: 0x20, count: padBytes))
        for (_, _, _, data) in tensors { fileData.append(data) }
        try fileData.write(to: url)
    }

    // MARK: - Tests

    func testEndToEndForwardPass() throws {
        let (dir, hidden, inter, K) = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }

        // Load the model and build the kernel bundle
        let loader = JANGTQLoader(device: device)
        let bundle = try loader.load(from: dir)
        let ctx = try MetalContext()
        let kernels = try JANGTQKernels(context: ctx)

        // Build the MoE block for layer 0
        let block = try JANGTQMoEBlock(
            layerIndex: 0,
            moePrefix: "block_sparse_moe",
            bundle: bundle,
            kernels: kernels
        )
        XCTAssertEqual(block.inFeatures, hidden)
        XCTAssertEqual(block.outFeatures, inter)
        XCTAssertEqual(block.nExperts, 2)
        XCTAssertEqual(block.bits, 2)

        // Build a constant input x = [1.0, 1.0, ..., 1.0] (hidden=64).
        // With signs all +1 and Hadamard normalized by 1/sqrt(64) = 1/8:
        //   x_rot[0] = sqrt(d) (DC component) and x_rot[i>0] = 0
        // So sum(x_rot) = sqrt(64) = 8.
        // With packed=0 + codebook[0]=1: gate = up = sum(x_rot) = 8.
        // SiLU(8) ≈ 7.997, so x_act ≈ 7.997 * 8 = 63.97 (constant across outs)
        // After Hadamard rotate of constant vector of length 32: again
        //   x_act_rot[0] = sqrt(32) * 63.97 ≈ 361.9, others = 0.
        // Then down = norms * sum(x_act_rot) where sum picks up only
        // x_act_rot[0] = 361.9 (since others are 0).
        // So y[k, r] should be ≈ 361.9 for every (k, r).
        let xValues = [Float16](repeating: 1.0, count: hidden)
        let xBuf = device.makeBuffer(length: hidden * MemoryLayout<Float16>.stride, options: .storageModeShared)!
        let xPtr = xBuf.contents().bindMemory(to: Float16.self, capacity: hidden)
        for i in 0..<hidden { xPtr[i] = xValues[i] }

        let indices: [UInt32] = [0, 1]  // pick experts 0 and 1
        let idxBuf = device.makeBuffer(length: K * 4, options: .storageModeShared)!
        let idxPtr = idxBuf.contents().bindMemory(to: UInt32.self, capacity: K)
        for i in 0..<K { idxPtr[i] = indices[i] }

        let yBuf = try block.runMLP(xHalfBuf: xBuf, selectedExpertsBuf: idxBuf, K: K)

        // Output shape: K rows × hidden cols
        XCTAssertEqual(yBuf.length, K * hidden * MemoryLayout<Float>.stride)
        let yPtr = yBuf.contents().bindMemory(to: Float.self, capacity: K * hidden)

        // Compute expected analytically:
        //   sum(x_rot) = sqrt(hidden)              (Hadamard of all-ones)
        //   gate = up = norm_g * sum(x_rot)        (norm=1, codebook[0]=1)
        //   x_act = SiLU(gate) * up                = SiLU(s) * s
        //   sum(x_act_rot) = sqrt(inter) * x_act   (Hadamard of constant)
        //   y = norm_d * sum(x_act_rot)            = sqrt(inter) * x_act
        let s: Float = sqrt(Float(hidden))           // 8
        let xAct: Float = (s / (1.0 + Float(exp(-Double(s))))) * s
        let expectedY: Float = sqrt(Float(inter)) * xAct

        var maxDiff: Float = 0
        for k in 0..<K {
            for r in 0..<hidden {
                let v = yPtr[k * hidden + r]
                maxDiff = max(maxDiff, abs(v - expectedY))
            }
        }
        // Tolerance accounts for half-precision input and the float→half
        // staging step in JANGTQMoEBlock.copyFloatToHalf.
        XCTAssertLessThan(maxDiff, 0.5,
            "MoE end-to-end max diff = \(maxDiff), expected ≈ \(expectedY)")
    }

    func testLayerWithMissingProjectionsThrows() throws {
        let (dir, _, _, _) = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let loader = JANGTQLoader(device: device)
        let bundle = try loader.load(from: dir)
        let ctx = try MetalContext()
        let kernels = try JANGTQKernels(context: ctx)

        // Layer 99 doesn't exist — should throw
        XCTAssertThrowsError(try JANGTQMoEBlock(
            layerIndex: 99, moePrefix: "block_sparse_moe",
            bundle: bundle, kernels: kernels
        ))
    }
}
