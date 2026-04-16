//
// Verify the loader correctly picks up MLX-format affine 8-bit weights
// (attention/embed/lm_head) ALONGSIDE the TQ MoE weights.
//
// Builds a tiny synthetic model with both kinds of tensors in the same
// safetensors shard and checks the bundle exposes them via the right
// dictionaries.
//

import Foundation
import XCTest
@testable import JANG
import Metal

final class JANGTQAffineLoadTests: XCTestCase {

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

    private func makeAffine8Triplet(
        base: String, outFeatures: Int, inFeatures: Int, groupSize: Int = 64
    ) -> [(String, String, [Int], Data)] {
        // 8-bit packs 4 vals per uint32 → in/4 packed columns
        let packedIn = inFeatures / 4
        let nGroups = inFeatures / groupSize

        var weightBytes = Data(count: outFeatures * packedIn * 4)
        weightBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            for i in 0..<(outFeatures * packedIn) { p[i] = UInt32(i & 0xFFFFFFFF) }
        }
        var scalesBytes = Data(count: outFeatures * nGroups * 2)
        scalesBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for i in 0..<(outFeatures * nGroups) { p[i] = Float16(0.01) }
        }
        var biasesBytes = Data(count: outFeatures * nGroups * 2)
        biasesBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for i in 0..<(outFeatures * nGroups) { p[i] = Float16(-0.1) }
        }
        return [
            (base + ".weight", "U32", [outFeatures, packedIn], weightBytes),
            (base + ".scales", "F16", [outFeatures, nGroups],  scalesBytes),
            (base + ".biases", "F16", [outFeatures, nGroups],  biasesBytes),
        ]
    }

    private func makeNorm(name: String, dim: Int) -> (String, String, [Int], Data) {
        var bytes = Data(count: dim * 2)
        bytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for i in 0..<dim { p[i] = 1.0 }
        }
        return (name, "F16", [dim], bytes)
    }

    private func floatsToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 4)
        for v in values {
            var x = v
            bytes.append(Data(bytes: &x, count: 4))
        }
        return bytes
    }

    private func buildModel() throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_affine_test_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        // config.json — same MoE setup as previous tests
        let cfg: [String: Any] = [
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 256,
            "num_local_experts": 2,
            "num_experts_per_tok": 2,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-5,
        ]
        try JSONSerialization.data(withJSONObject: cfg)
            .write(to: dir.appendingPathComponent("config.json"))

        let jangCfg: [String: Any] = [
            "version": 2, "weight_format": "mxtq",
            "profile": "JANGTQ_2L",
            "source_model": ["name": "Test", "architecture": "minimax_m2"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2, "attention": 8],
            "quantization": ["group_size": 64, "bits_default": 2, "method": "affine+mxtq"],
        ]
        try JSONSerialization.data(withJSONObject: jangCfg)
            .write(to: dir.appendingPathComponent("jang_config.json"))

        var tensors: [(String, String, [Int], Data)] = []

        // Affine 8-bit attention weights for layer 0
        // q_proj: hidden(64) → num_heads*head_dim = 4*16 = 64
        tensors.append(contentsOf: makeAffine8Triplet(
            base: "model.layers.0.self_attn.q_proj", outFeatures: 64, inFeatures: 64
        ))
        // k_proj/v_proj: hidden(64) → num_kv_heads*head_dim = 2*16 = 32
        tensors.append(contentsOf: makeAffine8Triplet(
            base: "model.layers.0.self_attn.k_proj", outFeatures: 32, inFeatures: 64
        ))
        tensors.append(contentsOf: makeAffine8Triplet(
            base: "model.layers.0.self_attn.v_proj", outFeatures: 32, inFeatures: 64
        ))
        // o_proj: 64 → 64
        tensors.append(contentsOf: makeAffine8Triplet(
            base: "model.layers.0.self_attn.o_proj", outFeatures: 64, inFeatures: 64
        ))

        // Per-layer norms (input + post-attention) — half tensors
        tensors.append(makeNorm(name: "model.layers.0.input_layernorm.weight", dim: 64))
        tensors.append(makeNorm(name: "model.layers.0.post_attention_layernorm.weight", dim: 64))
        tensors.append(makeNorm(name: "model.layers.0.self_attn.q_norm.weight", dim: 64))
        tensors.append(makeNorm(name: "model.layers.0.self_attn.k_norm.weight", dim: 32))

        // MoE expert weights (TQ format) — same shape as JANGTQLoaderTests
        let inter = 32, hidden = 64, packedInGate = 4, packedInDown = 2
        for e in 0..<2 {
            for (suffix, outF, packedIn) in [
                ("w1", inter, packedInGate),
                ("w2", hidden, packedInDown),
                ("w3", inter, packedInGate),
            ] {
                let base = "model.layers.0.block_sparse_moe.experts.\(e).\(suffix)"
                var packedBytes = Data(count: outF * packedIn * 4)
                packedBytes.withUnsafeMutableBytes { raw in
                    let p = raw.bindMemory(to: UInt32.self)
                    for i in 0..<(outF * packedIn) { p[i] = 0 }
                }
                var normsBytes = Data(count: outF * 2)
                normsBytes.withUnsafeMutableBytes { raw in
                    let p = raw.bindMemory(to: Float16.self)
                    for i in 0..<outF { p[i] = 1.0 }
                }
                var bitsValue: Int32 = 2
                let bitsBytes = Data(bytes: &bitsValue, count: 4)
                tensors.append((base + ".tq_packed", "U32", [outF, packedIn], packedBytes))
                tensors.append((base + ".tq_norms",  "F16", [outF],           normsBytes))
                tensors.append((base + ".tq_bits",   "I32", [],               bitsBytes))
            }
        }

        // Router (gate Linear at half precision — NOT quantized)
        var routerBytes = Data(count: 2 * 64 * 2)
        routerBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Float16.self)
            for i in 0..<(2 * 64) { p[i] = Float16(0.001) }
        }
        tensors.append(("model.layers.0.block_sparse_moe.gate.weight", "F16", [2, 64], routerBytes))

        // Final norm + embed/lm_head (all standard-format)
        tensors.append(makeNorm(name: "model.norm.weight", dim: 64))
        tensors.append(contentsOf: makeAffine8Triplet(
            base: "model.embed_tokens", outFeatures: 256, inFeatures: 64
        ))
        tensors.append(contentsOf: makeAffine8Triplet(
            base: "lm_head", outFeatures: 256, inFeatures: 64
        ))

        try writeShard(
            url: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: tensors
        )

        // Sidecar
        let sidecar: [(String, String, [Int], Data)] = [
            ("signs.64.42", "F32", [64], floatsToData([Float](repeating: 1.0, count: 64))),
            ("signs.32.42", "F32", [32], floatsToData([Float](repeating: 1.0, count: 32))),
            ("codebook.64.2", "F32", [4], floatsToData([1.0, 0, 0, 0])),
            ("codebook.32.2", "F32", [4], floatsToData([1.0, 0, 0, 0])),
        ]
        try writeShard(
            url: dir.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: sidecar
        )

        return dir
    }

    func testLoadsAffineAndTQTogether() throws {
        let dir = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw XCTSkip("No Metal device")
        }
        let bundle = try JANGTQLoader(device: device).load(from: dir)

        // Affine 8-bit attention weights present
        XCTAssertNotNil(bundle.affineWeights["model.layers.0.self_attn.q_proj"])
        XCTAssertNotNil(bundle.affineWeights["model.layers.0.self_attn.k_proj"])
        XCTAssertNotNil(bundle.affineWeights["model.layers.0.self_attn.v_proj"])
        XCTAssertNotNil(bundle.affineWeights["model.layers.0.self_attn.o_proj"])
        XCTAssertNotNil(bundle.affineWeights["model.embed_tokens"])
        XCTAssertNotNil(bundle.affineWeights["lm_head"])

        // Inferred bits + group_size + dimensions correct
        let qProj = bundle.affineWeights["model.layers.0.self_attn.q_proj"]!
        XCTAssertEqual(qProj.bits, 8)
        XCTAssertEqual(qProj.groupSize, 64)
        XCTAssertEqual(qProj.inFeatures, 64)
        XCTAssertEqual(qProj.outFeatures, 64)

        // Half tensors picked up
        XCTAssertNotNil(bundle.halfTensors["model.layers.0.input_layernorm.weight"])
        XCTAssertNotNil(bundle.halfTensors["model.layers.0.post_attention_layernorm.weight"])
        XCTAssertNotNil(bundle.halfTensors["model.norm.weight"])
        XCTAssertNotNil(bundle.halfTensors["model.layers.0.block_sparse_moe.gate.weight"])
        XCTAssertNotNil(bundle.halfTensors["model.layers.0.self_attn.q_norm.weight"])

        // TQ MoE weights still grouped correctly
        XCTAssertEqual(bundle.nStackedGroups, 3)
        XCTAssertNotNil(bundle.weights["model.layers.0.block_sparse_moe.switch_mlp.gate_proj"])
        XCTAssertNotNil(bundle.weights["model.layers.0.block_sparse_moe.switch_mlp.up_proj"])
        XCTAssertNotNil(bundle.weights["model.layers.0.block_sparse_moe.switch_mlp.down_proj"])
    }
}
