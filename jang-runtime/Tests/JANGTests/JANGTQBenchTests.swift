//
// Single-pass perf bench for the JANGTQ Swift inference engine.
// Builds a 1-layer model at MiniMax M2.7 dimensions, runs N forward
// passes, and reports tokens/sec.
//
// This is NOT a correctness test — it's a wall-clock measurement of
// the GPU decode hot path (embed → attention → MoE → norm → lm_head).
// The 1-layer setup measures per-layer cost; multiply by 62 for
// MiniMax M2.7's full decode time estimate.
//
// Disabled by default (very slow) — enable via env var JANGTQ_BENCH=1.
//

import Foundation
import XCTest
@testable import JANG
import JANGCoreMetal
import Metal

final class JANGTQBenchTests: XCTestCase {

    private func buildMiniLikeModel() throws -> URL {
        let fm = FileManager.default
        let dir = fm.temporaryDirectory.appendingPathComponent(
            "jangtq_bench_\(UUID().uuidString)"
        )
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        // MiniMax M2.7 dimensions (1 layer for benching)
        let hidden = 3072
        let inter = 1536
        let nHeads = 48
        let nKVHeads = 8
        let headDim = 128
        let nExperts = 16  // smaller than real 256 to reduce file size
        let K = 8
        let vocab = 1024   // smaller for quick load
        let gs = 64

        let cfg: [String: Any] = [
            "model_type": "minimax_m2",
            "architectures": ["MiniMaxM2ForCausalLM"],
            "hidden_size": hidden,
            "intermediate_size": 12288,
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
            "source_model": ["name": "Bench", "architecture": "minimax_m2"],
            "mxtq_seed": 42,
            "mxtq_bits": ["routed_expert": 2, "attention": 8],
            "quantization": ["group_size": gs, "bits_default": 2],
        ]
        try JSONSerialization.data(withJSONObject: jangCfg)
            .write(to: dir.appendingPathComponent("jang_config.json"))

        // Helper closures
        func halvesData(_ count: Int, fill: Float) -> Data {
            var bytes = Data(count: count * 2)
            bytes.withUnsafeMutableBytes { raw in
                let p = raw.bindMemory(to: Float16.self)
                for i in 0..<count { p[i] = Float16(fill) }
            }
            return bytes
        }
        func floatsData(_ count: Int, fill: Float) -> Data {
            var bytes = Data(count: count * 4)
            bytes.withUnsafeMutableBytes { raw in
                let p = raw.bindMemory(to: Float.self)
                for i in 0..<count { p[i] = fill }
            }
            return bytes
        }
        func affine8(base: String, outF: Int, inF: Int) -> [(String, String, [Int], Data)] {
            let packedIn = inF / 4
            let nGroups = inF / gs
            var w = Data(count: outF * packedIn * 4)
            w.withUnsafeMutableBytes { raw in
                let p = raw.bindMemory(to: UInt32.self)
                let word: UInt32 = 0  // q=0 across the board
                for i in 0..<(outF * packedIn) { p[i] = word }
            }
            return [
                (base + ".weight", "U32", [outF, packedIn], w),
                (base + ".scales", "F16", [outF, nGroups],  halvesData(outF * nGroups, fill: 0.001)),
                (base + ".biases", "F16", [outF, nGroups],  halvesData(outF * nGroups, fill: 0.0)),
            ]
        }
        func tq(base: String, outF: Int, packedIn: Int) -> [(String, String, [Int], Data)] {
            var p = Data(count: outF * packedIn * 4)
            p.withUnsafeMutableBytes { raw in
                let q = raw.bindMemory(to: UInt32.self)
                for i in 0..<(outF * packedIn) { q[i] = 0 }
            }
            var bv: Int32 = 2
            return [
                (base + ".tq_packed", "U32", [outF, packedIn], p),
                (base + ".tq_norms",  "F16", [outF],           halvesData(outF, fill: 1.0)),
                (base + ".tq_bits",   "I32", [],               Data(bytes: &bv, count: 4)),
            ]
        }
        func norm(_ name: String, dim: Int) -> (String, String, [Int], Data) {
            (name, "F16", [dim], halvesData(dim, fill: 1.0))
        }

        var tensors: [(String, String, [Int], Data)] = []
        tensors.append(contentsOf: affine8(base: "model.embed_tokens", outF: vocab, inF: hidden))
        tensors.append(contentsOf: affine8(base: "lm_head", outF: vocab, inF: hidden))
        tensors.append(norm("model.norm.weight", dim: hidden))
        tensors.append(norm("model.layers.0.input_layernorm.weight", dim: hidden))
        tensors.append(norm("model.layers.0.post_attention_layernorm.weight", dim: hidden))
        tensors.append(norm("model.layers.0.self_attn.q_norm.weight", dim: nHeads * headDim))
        tensors.append(norm("model.layers.0.self_attn.k_norm.weight", dim: nKVHeads * headDim))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.q_proj",
            outF: nHeads * headDim, inF: hidden
        ))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.k_proj",
            outF: nKVHeads * headDim, inF: hidden
        ))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.v_proj",
            outF: nKVHeads * headDim, inF: hidden
        ))
        tensors.append(contentsOf: affine8(
            base: "model.layers.0.self_attn.o_proj",
            outF: hidden, inF: nHeads * headDim
        ))

        // Expert MLP weights
        let pcGate = hidden / 16
        let pcDown = inter / 16
        for e in 0..<nExperts {
            tensors.append(contentsOf: tq(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w1",
                outF: inter, packedIn: pcGate
            ))
            tensors.append(contentsOf: tq(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w2",
                outF: hidden, packedIn: pcDown
            ))
            tensors.append(contentsOf: tq(
                base: "model.layers.0.block_sparse_moe.experts.\(e).w3",
                outF: inter, packedIn: pcGate
            ))
        }
        tensors.append((
            "model.layers.0.block_sparse_moe.gate.weight", "F16", [nExperts, hidden],
            halvesData(nExperts * hidden, fill: 0.0)
        ))
        tensors.append((
            "model.layers.0.block_sparse_moe.e_score_correction_bias", "F16", [nExperts],
            halvesData(nExperts, fill: 0.0)
        ))

        try writeShard(
            url: dir.appendingPathComponent("model-00001-of-00001.safetensors"),
            tensors: tensors
        )

        // Sidecar with sqrt-2 distinct codebooks
        let sidecar: [(String, String, [Int], Data)] = [
            ("signs.\(hidden).42", "F32", [hidden], floatsData(hidden, fill: 1.0)),
            ("signs.\(inter).42",  "F32", [inter],  floatsData(inter, fill: 1.0)),
            ("codebook.\(hidden).2", "F32", [4],
             {
                var bytes = Data(count: 16)
                bytes.withUnsafeMutableBytes { raw in
                    let p = raw.bindMemory(to: Float.self)
                    p[0] = -0.027; p[1] = -0.008; p[2] = 0.008; p[3] = 0.027
                }
                return bytes
             }()),
            ("codebook.\(inter).2",  "F32", [4],
             {
                var bytes = Data(count: 16)
                bytes.withUnsafeMutableBytes { raw in
                    let p = raw.bindMemory(to: Float.self)
                    p[0] = -0.038; p[1] = -0.011; p[2] = 0.011; p[3] = 0.038
                }
                return bytes
             }()),
        ]
        try writeShard(
            url: dir.appendingPathComponent("jangtq_runtime.safetensors"),
            tensors: sidecar
        )

        return dir
    }

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

    func testBenchOneLayerDecode() throws {
        // Skip unless explicitly enabled
        guard ProcessInfo.processInfo.environment["JANGTQ_BENCH"] == "1" else {
            throw XCTSkip("Bench disabled — set JANGTQ_BENCH=1 to run")
        }

        let dir = try buildMiniLikeModel()
        defer { try? FileManager.default.removeItem(at: dir) }

        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("no Metal") }
        let bundle = try JANGTQLoader(device: device).load(from: dir)
        let ctx = try MetalContext()
        let model = try JANGTQModel(bundle: bundle, context: ctx, maxSeqLen: 64)

        // Warmup
        for p in 0..<4 { _ = try model.forward(tokenId: p % 5, position: p) }
        model.reset()

        // Measure 30 decode steps
        let nSteps = 30
        let t0 = Date()
        for p in 0..<nSteps {
            _ = try model.forward(tokenId: p % 7, position: p)
        }
        let elapsed = Date().timeIntervalSince(t0)
        let perStep = elapsed / Double(nSteps)
        print("--- JANGTQ Swift bench (1-layer MiniMax M2.7 dims) ---")
        print("  steps: \(nSteps)")
        print("  total: \(String(format: "%.3f", elapsed)) s")
        print("  per-step: \(String(format: "%.2f", perStep * 1000)) ms = \(String(format: "%.1f", 1.0 / perStep)) tok/s")
        print("  est. 62-layer per-step: \(String(format: "%.2f", perStep * 1000 * 62)) ms = \(String(format: "%.1f", 1.0 / (perStep * 62))) tok/s")
    }
}
