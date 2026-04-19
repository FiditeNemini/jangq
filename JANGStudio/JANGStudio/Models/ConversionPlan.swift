// JANGStudio/JANGStudio/Models/ConversionPlan.swift
import Foundation
import Observation

enum Family: String, Codable, CaseIterable { case jang, jangtq }
enum QuantMethod: String, Codable, CaseIterable { case mse, rtn, mseAll }
enum SourceDtype: String, Codable { case bf16, fp16, fp8, jangV2, unknown }
enum RunState: String, Codable { case idle, running, succeeded, failed, cancelled }

struct ArchitectureSummary: Codable, Equatable {
    let modelType: String
    let isMoE: Bool
    let numExperts: Int
    let isVL: Bool
    let isVideoVL: Bool
    let hasGenerationConfig: Bool
    let dtype: SourceDtype
    let totalBytes: Int64
    let shardCount: Int
}

struct ArchitectureOverrides: Codable, Equatable {
    var forceDtype: SourceDtype? = nil
    var forceBlockSize: Int? = nil
    var skipPatterns: [String] = []
    var calibrationJSONL: URL? = nil
}

/// v1 JANGTQ whitelist — KEEP IN SYNC with jang-tools/jang_tools/inspect_source.py.
let JANGTQ_V1_WHITELIST: Set<String> = ["qwen3_5_moe", "minimax_m2"]

@Observable
final class ConversionPlan: Codable {
    var sourceURL: URL?
    var detected: ArchitectureSummary?
    var overrides = ArchitectureOverrides()
    var family: Family = .jang
    var profile: String = "JANG_4K"
    var method: QuantMethod = .mse
    var hadamard: Bool = false
    var outputURL: URL?
    var run: RunState = .idle

    init() {}

    var isStep1Complete: Bool { sourceURL != nil && detected != nil }
    var isStep2Complete: Bool { isStep1Complete }          // step 2 only requires confirmation
    var isStep3Complete: Bool { isStep2Complete && outputURL != nil }
    var isStep4Complete: Bool { run == .succeeded }

    var isJANGTQAllowed: Bool {
        guard let mt = detected?.modelType else { return false }
        return JANGTQ_V1_WHITELIST.contains(mt)
    }

    // MARK: - UserDefaults persistence

    // @Observable rewrites stored properties so synthesized Codable doesn't work.
    // Provide explicit encode/decode keyed on the same names.
    enum CodingKeys: String, CodingKey {
        case sourceURL, detected, overrides, family, profile, method, hadamard, outputURL
    }

    required init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sourceURL  = try c.decodeIfPresent(URL.self,                  forKey: .sourceURL)
        detected   = try c.decodeIfPresent(ArchitectureSummary.self,  forKey: .detected)
        overrides  = try c.decodeIfPresent(ArchitectureOverrides.self, forKey: .overrides) ?? ArchitectureOverrides()
        family     = try c.decodeIfPresent(Family.self,               forKey: .family)    ?? .jang
        profile    = try c.decodeIfPresent(String.self,               forKey: .profile)   ?? "JANG_4K"
        method     = try c.decodeIfPresent(QuantMethod.self,          forKey: .method)    ?? .mse
        hadamard   = try c.decodeIfPresent(Bool.self,                 forKey: .hadamard)  ?? false
        outputURL  = try c.decodeIfPresent(URL.self,                  forKey: .outputURL)
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(sourceURL, forKey: .sourceURL)
        try c.encodeIfPresent(detected,  forKey: .detected)
        try c.encode(overrides,          forKey: .overrides)
        try c.encode(family,             forKey: .family)
        try c.encode(profile,            forKey: .profile)
        try c.encode(method,             forKey: .method)
        try c.encode(hadamard,           forKey: .hadamard)
        try c.encodeIfPresent(outputURL, forKey: .outputURL)
    }

    func encodeForDefaults() throws -> Data { try JSONEncoder().encode(self) }
    static func decodeFromDefaults(_ data: Data) throws -> ConversionPlan {
        try JSONDecoder().decode(ConversionPlan.self, from: data)
    }
}
