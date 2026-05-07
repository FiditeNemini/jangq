#!/usr/bin/env swift
// Standalone ZAYA runtime contract printer for Swift/vMLX integration.
//
// Run:
//   swift ZayaRuntimeContract.swift [/path/to/ZAYA1-8B]

import Foundation

let defaultPath = "/Users/eric/jang/models/Zyphra/ZAYA1-8B"
let modelPath = CommandLine.arguments.dropFirst().first ?? defaultPath
let modelURL = URL(fileURLWithPath: modelPath)

func readJSON(_ name: String) throws -> [String: Any] {
    let url = modelURL.appendingPathComponent(name)
    let data = try Data(contentsOf: url)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw NSError(domain: "ZayaRuntimeContract", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "\(name) is not a JSON object"
        ])
    }
    return object
}

func int(_ dict: [String: Any], _ key: String) -> Int {
    if let value = dict[key] as? Int { return value }
    if let value = dict[key] as? NSNumber { return value.intValue }
    fatalError("missing integer config key: \(key)")
}

func string(_ dict: [String: Any], _ key: String) -> String {
    return dict[key] as? String ?? ""
}

func jsonValueKind(_ value: Any?) -> String {
    guard let value else { return "missing" }
    if value is NSNull { return "null" }
    if value is String { return "string" }
    if value is NSNumber { return "number" }
    if value is [Any] { return "array" }
    if value is [String: Any] { return "object" }
    return "\(type(of: value))"
}

struct ZayaRuntimePolicy {
    let continuousBatching: String
    let pagedKV: String
    let prefixCache: String
    let turboQuantKV: String
    let chunkedPrefill: String
}

do {
    let config = try readJSON("config.json")
    let index = try readJSON("model.safetensors.index.json")
    let tokenizer = try readJSON("tokenizer_config.json")

    guard let weightMap = index["weight_map"] as? [String: String] else {
        fatalError("model.safetensors.index.json has no weight_map")
    }

    let hidden = int(config, "hidden_size")
    let layers = int(config, "num_hidden_layers")
    let heads = int(config, "num_attention_heads")
    let qHeads = int(config, "cca_num_q_heads")
    let kvHeads = int(config, "num_query_groups")
    let experts = int(config, "num_experts")
    let topK = int(config, "moe_router_topk")
    let headDim = hidden / heads
    let qDim = qHeads * headDim
    let kvDim = kvHeads * headDim
    let convChannels = qDim + kvDim
    let attnLayers = Array(stride(from: 0, to: layers, by: 2))
    let moeLayers = Array(stride(from: 1, to: layers, by: 2))

    let policy = ZayaRuntimePolicy(
        continuousBatching: "compatible when each slot owns independent KV plus CCA conv/prev_hs state",
        pagedKV: "compatible for standard attention KV; CCA state must be stored/restored beside blocks",
        prefixCache: "disabled for first port; official vLLM asserts prefix caching off",
        turboQuantKV: "KV-only experimental; do not TQ-encode CCA conv/prev_hs",
        chunkedPrefill: "hold until CCA state copy tests pass"
    )

    print("ZAYA source: \(modelPath)")
    print("model_type: \(string(config, "model_type"))")
    print("tensors: \(weightMap.count)")
    if let metadata = index["metadata"] as? [String: Any] {
        print("total_parameters: \(metadata["total_parameters"] ?? "unknown")")
        print("total_size: \(metadata["total_size"] ?? "unknown")")
    }

    print("")
    print("Layer schedule")
    print("  attention layers: \(attnLayers.count), even indices \(attnLayers.prefix(6)) ... \(attnLayers.suffix(6))")
    print("  moe layers: \(moeLayers.count), odd indices \(moeLayers.prefix(6)) ... \(moeLayers.suffix(6))")

    print("")
    print("Attention geometry")
    print("  hidden=\(hidden), heads=\(heads), head_dim=\(headDim)")
    print("  CCA q_heads=\(qHeads), kv_heads=\(kvHeads)")
    print("  q_dim=\(qDim), k_dim=\(kvDim), v_dim=\(kvDim), conv_qk_channels=\(convChannels)")
    print("  KV cache per attention layer: [B, \(kvHeads), T, \(headDim)]")
    print("  CCA conv state per attention layer: [B, \(convChannels), 2]")
    print("  CCA prev_hs per attention layer: [B, \(hidden)]")

    print("")
    print("MoE geometry")
    print("  experts=\(experts), router logits=\(experts + 1), topk=\(topK)")
    print("  linear_fc1 is fused gate/up and should be split for JANGTQ fused GateUp kernels")
    print("  linear_fc2 maps SwiGLU output back to hidden")

    print("")
    print("Tokenizer/template")
    print("  tokenizer_class: \(tokenizer["tokenizer_class"] ?? "unknown")")
    print("  tokenizer_config.chat_template: \(jsonValueKind(tokenizer["chat_template"]))")
    print("  chat_template.jinja present: \(FileManager.default.fileExists(atPath: modelURL.appendingPathComponent("chat_template.jinja").path))")
    print("  config eos_token_id: \(config["eos_token_id"] ?? "unknown")")
    if let generation = try? readJSON("generation_config.json") {
        print("  generation_config eos_token_id: \(generation["eos_token_id"] ?? "unknown")")
    }

    print("")
    print("Runtime policy")
    print("  continuous batching: \(policy.continuousBatching)")
    print("  paged KV: \(policy.pagedKV)")
    print("  prefix cache: \(policy.prefixCache)")
    print("  TurboQuant KV: \(policy.turboQuantKV)")
    print("  chunked prefill: \(policy.chunkedPrefill)")
} catch {
    fputs("ERROR: \(error)\n", stderr)
    exit(1)
}
