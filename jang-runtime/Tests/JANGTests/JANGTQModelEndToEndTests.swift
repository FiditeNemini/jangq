//
// End-to-end JANGTQ model test — full single-token forward pass through
// a synthetic 1-layer 2-expert model. Verifies that:
//
//   embed → attention(q/k/v/o) → residual → MoE(router + fused_gu + gather + combine) → residual
//   → final norm → lm_head → logits
//
// runs without crashing AND produces deterministic output. With all-zero
// weights, the output should be the lm_head bias-driven baseline.
//

import Foundation
import XCTest
@testable import JANG
import JANGCoreMetal
import Metal

final class JANGTQModelEndToEndTests: XCTestCase {

    private func writeShard(url: URL, tensors: [(String, String, [Int], Data)]) throws {
        var headerDict: [String: Any] = [:]
        var offset = 0
        for (name, dtype, shape, data) in tensors {
            headerDict[name] = [
                "dtype": dtype, "shape": shape,
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

    private func floatsToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 4)
        for v in values { var x = v; bytes.append(Data(bytes: &x, count: 4)) }
        return bytes
    }
    private func halvesToData(_ values: [Float]) -> Data {
        var bytes = Data(capacity: values.count * 2)
        for v in values { var h = Float16(v); bytes.append(Data(bytes: &h, count: 2)) }
        return bytes
    }

    private func affine8Triplet(
        base: String, outFeatures: Int, inFeatures: Int, groupSize: Int = 64,
        fillQ: UInt8 = 0, fillScale: Float = 0.0, fillBias: Float = 0.0
    ) -> [(String, String, [Int], Data)] {
        let packedIn = inFeatures / 4
        let nGroups = inFeatures / groupSize
        var weight = Data(count: outFeatures * packedIn * 4)
        weight.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            let w: UInt32 = UInt32(fillQ) | UInt32(fillQ) << 8 | UInt32(fillQ) << 16 | UInt32(fillQ) << 24
            for i in 0..<(outFeatures * packedIn) { p[i] = w }
        }
        let scales = halvesToData([Float](repeating: fillScale, count: outFeatures * nGroups))
        let biases = halvesToData([Float](repeating: fillBias, count: outFeatures * nGroups))
        return [
            (base + ".weight", "U32", [outFeatures, packedIn], weight),
            (base + ".scales", "F16", [outFeatures, nGroups],  scales),
            (base + ".biases", "F16", [outFeatures, nGroups],  biases),
        ]
    }

    private func tqTriplet(
        base: String, outFeatures: Int, packedIn: Int, fill: UInt32 = 0
    ) -> [(String, String, [Int], Data)] {
        var packedBytes = Data(count: outFeatures * packedIn * 4)
        packedBytes.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: UInt32.self)
            for i in 0..<(outFeatures * packedIn) { p[i] = fill }
        }
        let normsBytes = halvesToData([Float](repeating: 1.0, count: outFeatures))
        var bitsValue: Int32 = 2
        let bitsBytes = Data(bytes: &bitsValue, count: 4)
        return [
            (base + ".tq_packed", "U32", [outFeatures, packedIn], packedBytes),
            (base + ".tq_norms",  "F16", [outFeatures],           normsBytes),
            (base + ".tq_bits",   "I32", [],                      bitsBytes),
        ]
    }

    private func makeNorm(_ name: String, dim: Int, value: Float = 1.0) -> (String, String, [Int], Data) {
        return (name, "F16", [dim], halvesToData([Float](repeating: value, count: dim)))
    }

    private func buildModel() throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_e2e_test_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        // Pick dims that make 8-bit packing work cleanly.
        // hidden = 64, head_dim = 16 → nHeads = 4, nKVHeads = 2
        // group_size for 8-bit must divide in_features
        let hidden = 64
        let inter = 32
        let nHeads = 4
        let nKVHeads = 2
        let headDim = 16
        let nExperts = 4
        let K = 2
        let vocab = 32
        // Use group_size that divides BOTH hidden(=64) and headDim*nHeads(=64) etc.
        // group_size = 16 works for everything in this tiny test.
        let gs = 16

        let cfg: [String: Any] = [
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": 128,
            "moe_intermediate_size": inter,
            "num_hidden_layers": 1,
            "num_attention_heads": nHeads,
            "num_key_value_heads": nKVHeads,
            "head_dim": headDim,
            "vocab_size": vocab,
            "num_local_experts": nExperts,
            "num_experts_per_tok": K,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
        ]
        try JSONSerialization.data(withJSONObject: cfg)
            .write(to: dir.appendingPathComponent("config.json"))

        let jangCfg: [String: Any] = [
            "version": 2, "weight_format": "mxtq", "profile": "JANGTQ_2L",
            "source_model": ["name": "Test", "architecture": "minimax_m2"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2, "attention": 8],
            "quantization": ["group_size": gs, "bits_default": 2, "method": "affine+mxtq"],
        ]
        try JSONSerialization.data(withJSONObject: jangCfg)
            .write(to: dir.appendingPathComponent("jang_config.json"))

        var tensors: [(String, String, [Int], Data)] = []

        // Embed and lm_head — small q=1 baseline so dequant gives non-zero.
        // q=1, scale=0.01, bias=0 → val = 0.01 → embedded vector is (0.01,)*hidden
        tensors.append(contentsOf: affine8Triplet(
            base: "model.embed_tokens", outFeatures: vocab, inFeatures: hidden,
            groupSize: gs, fillQ: 1, fillScale: 0.01, fillBias: 0.0
        ))
        tensors.append(contentsOf: affine8Triplet(
            base: "lm_head", outFeatures: vocab, inFeatures: hidden,
            groupSize: gs, fillQ: 1, fillScale: 0.01, fillBias: 0.0
        ))

        // Norms — all ones
        tensors.append(makeNorm("model.norm.weight", dim: hidden))
        tensors.append(makeNorm("model.layers.0.input_layernorm.weight", dim: hidden))
        tensors.append(makeNorm("model.layers.0.post_attention_layernorm.weight", dim: hidden))
        tensors.append(makeNorm("model.layers.0.self_attn.q_norm.weight", dim: nHeads * headDim))
        tensors.append(makeNorm("model.layers.0.self_attn.k_norm.weight", dim: nKVHeads * headDim))

        // Attention projections — q=0, scale=0, bias=0 → all zero outputs
        tensors.append(contentsOf: affine8Triplet(
            base: "model.layers.0.self_attn.q_proj",
            outFeatures: nHeads * headDim, inFeatures: hidden, groupSize: gs
        ))
        tensors.append(contentsOf: affine8Triplet(
            base: "model.layers.0.self_attn.k_proj",
            outFeatures: nKVHeads * headDim, inFeatures: hidden, groupSize: gs
        ))
        tensors.append(contentsOf: affine8Triplet(
            base: "model.layers.0.self_attn.v_proj",
            outFeatures: nKVHeads * headDim, inFeatures: hidden, groupSize: gs
        ))
        tensors.append(contentsOf: affine8Triplet(
            base: "model.layers.0.self_attn.o_proj",
            outFeatures: hidden, inFeatures: nHeads * headDim, groupSize: gs
        ))

        // MoE expert weights (TQ, all zero packed → use codebook[0]=0 below)
        let pcGate = hidden / 16
        let pcDown = inter / 16
        for e in 0..<nExperts {
            tensors.append(contentsOf: tqTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w1",
                outFeatures: inter, packedIn: pcGate
            ))
            tensors.append(contentsOf: tqTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w2",
                outFeatures: hidden, packedIn: pcDown
            ))
            tensors.append(contentsOf: tqTriplet(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w3",
                outFeatures: inter, packedIn: pcGate
            ))
        }

        // Router gate (n_experts × hidden, zeros → uniform sigmoid 0.5)
        tensors.append((
            "model.layers.0.block_sparse_moe.gate.weight", "F16", [nExperts, hidden],
            halvesToData([Float](repeating: 0.0, count: nExperts * hidden))
        ))
        tensors.append((
            "model.layers.0.block_sparse_moe.e_score_correction_bias", "F16", [nExperts],
            halvesToData([Float](repeating: 0.0, count: nExperts))
        ))

        try writeShard(
            url: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: tensors
        )

        // Sidecar — codebook[0]=0 so all MoE outputs are 0
        let sidecar: [(String, String, [Int], Data)] = [
            ("signs.\(hidden).42", "F32", [hidden], floatsToData([Float](repeating: 1.0, count: hidden))),
            ("signs.\(inter).42",  "F32", [inter],  floatsToData([Float](repeating: 1.0, count: inter))),
            ("codebook.\(hidden).2", "F32", [4], floatsToData([0, 0, 0, 0])),
            ("codebook.\(inter).2",  "F32", [4], floatsToData([0, 0, 0, 0])),
        ]
        try writeShard(
            url: dir.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: sidecar
        )

        return dir
    }

    func testFullModelForwardPass() throws {
        let dir = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }

        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let ctx = try MetalContext()
        let model = try JANGTQModel(bundle: bundle, context: ctx, maxSeqLen: 16)

        // Forward token 0 at position 0
        let logits = try model.forward(tokenId: 0, position: 0)
        XCTAssertEqual(logits.length, model.config.vocabSize * MemoryLayout<Float>.stride)

        // With all-zero attention/MoE output:
        //   x_after_layer = embed_row + 0 + 0 = embed_row
        //   embed_row = (0.01,)*hidden  (q=1, scale=0.01)
        // After RMSNorm with all-ones gamma: rms = 0.01, output = (1.0,)*hidden
        //   (each elem = 0.01 / 0.01 * 1.0 = 1.0 — actually rms = sqrt(0.0001) = 0.01)
        //   output[i] = (0.01 / 0.01) * 1.0 = 1.0
        // Then lm_head: q=1, scale=0.01, bias=0 → val=0.01
        //   logits[v] = sum_h (0.01 * 1.0) = 0.01 * hidden = 0.64
        // So all logits should be ~0.64.
        let p = logits.contents().bindMemory(to: Float.self, capacity: model.config.vocabSize)
        let expected: Float = 0.01 * Float(model.config.hiddenSize)
        var maxDiff: Float = 0
        for i in 0..<model.config.vocabSize {
            maxDiff = max(maxDiff, abs(p[i] - expected))
        }
        XCTAssertLessThan(maxDiff, 0.1,
            "forward produced unexpected logits: max diff = \(maxDiff) from expected \(expected)")

        // Greedy sample — should be a valid token index
        let sampler = JANGTQSampler()
        let tok = sampler.argmax(logits: logits, vocabSize: model.config.vocabSize)
        XCTAssertGreaterThanOrEqual(tok, 0)
        XCTAssertLessThan(tok, model.config.vocabSize)
    }

    func testMultiStepDecodeAdvancesCache() throws {
        let dir = try buildModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let ctx = try MetalContext()
        let model = try JANGTQModel(bundle: bundle, context: ctx, maxSeqLen: 8)

        // 4 forward passes → cache.currentLength should advance to 4
        for pos in 0..<4 {
            _ = try model.forward(tokenId: pos % 5, position: pos)
        }
        XCTAssertEqual(model.cache.currentLength, 4)

        model.reset()
        XCTAssertEqual(model.cache.currentLength, 0)
    }
}
