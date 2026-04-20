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

    init(modelType: String, isMoE: Bool, numExperts: Int, isVL: Bool,
         isVideoVL: Bool = false, hasGenerationConfig: Bool = false,
         dtype: SourceDtype, totalBytes: Int64, shardCount: Int) {
        self.modelType = modelType
        self.isMoE = isMoE
        self.numExperts = numExperts
        self.isVL = isVL
        self.isVideoVL = isVideoVL
        self.hasGenerationConfig = hasGenerationConfig
        self.dtype = dtype
        self.totalBytes = totalBytes
        self.shardCount = shardCount
    }
}

struct ArchitectureOverrides: Codable, Equatable {
    var forceDtype: SourceDtype? = nil
    var forceBlockSize: Int? = nil
    var skipPatterns: [String] = []
    // M200 (iter 137): `calibrationJSONL` removed — zero downstream
    // consumers across the entire JANGStudio codebase, would have
    // been a settings-UI lie if any wizard step had offered a picker.
    // No UI currently exposes it, so removing the declaration doesn't
    // affect any user-visible surface; this is pure dead-field cleanup
    // bundled with the defaultCalibrationSamples removal.
    // Codable forward-compat: pre-M200 persisted ArchitectureOverrides
    // JSON blobs (e.g., via plan export) that contain the key will
    // decode cleanly because JSONDecoder tolerates unknown keys.
}

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

    /// Seed a fresh plan with the user's configured defaults. Only fields that
    /// are safe to auto-populate at wizard entry get touched — specifically
    /// the knobs that live in Settings → General → Defaults. `sourceURL`,
    /// `detected`, `outputURL`, and `run` are intentionally untouched: those
    /// are per-conversion state, not user defaults.
    ///
    /// Introduced iter 10 (M62 chain): previously `defaultProfile`,
    /// `defaultFamily`, `defaultMethod`, `defaultHadamardEnabled` were
    /// persisted but never read anywhere — the wizard always started on
    /// `JANG_4K` / `jang` / `mse` / hadamard=false regardless of what the
    /// user set in Settings.
    @MainActor
    func applyDefaults(from settings: AppSettings) {
        // Profile: only apply if the settings value is non-empty — defends
        // against first-launch or corrupted UserDefaults where profile is "".
        if !settings.defaultProfile.isEmpty {
            profile = settings.defaultProfile
        }
        if let fam = Family(rawValue: settings.defaultFamily) {
            family = fam
        }
        switch settings.defaultMethod.lowercased() {
        case "mse": method = .mse
        case "rtn": method = .rtn
        case "mse-all", "mseall", "mse_all": method = .mseAll
        default: break   // unknown value → keep current default
        }
        hadamard = settings.defaultHadamardEnabled
    }

    /// Step 1 completes only when we've picked a folder AND detection found a
    /// real model there — meaning at least one .safetensors shard is present.
    /// A folder with just a config.json and nothing else is NOT a complete step 1;
    /// surfacing this prevents silent progression when the user picks the wrong
    /// folder (e.g., a parent directory, a docs folder, or a broken download).
    var isStep1Complete: Bool {
        sourceURL != nil && detected != nil && (detected?.shardCount ?? 0) > 0
    }
    var isStep2Complete: Bool { isStep1Complete }          // step 2 only requires confirmation
    var isStep3Complete: Bool { isStep2Complete && outputURL != nil }
    var isStep4Complete: Bool { run == .succeeded }

    func isJANGTQAllowed(for whitelist: [String]) -> Bool {
        guard let mt = detected?.modelType else { return false }
        return whitelist.contains(mt)
    }

    @available(*, deprecated, message: "Use isJANGTQAllowed(for:) with CapabilitiesService.capabilities.jangtqWhitelist")
    var isJANGTQAllowed: Bool {
        guard let mt = detected?.modelType else { return false }
        return ["qwen3_5_moe", "minimax_m2"].contains(mt)
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
