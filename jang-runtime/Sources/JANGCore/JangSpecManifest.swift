import Foundation

/// Mirror of Python `jang_tools.jangspec.manifest.Manifest`.
///
/// Decoded from `jangspec.json` via `JSONDecoder`. Field names must match
/// Python's `dataclasses.asdict` output exactly.
public struct JangSpecManifest: Codable, Sendable, Equatable {
    public var bundleVersion: Int
    public var sourceJang: String
    public var sourceJangDir: String
    public var targetArch: String
    public var nLayers: Int
    public var nExpertsPerLayer: Int
    public var targetTopK: Int
    public var tokenizerHash: String
    public var hotCoreTensors: [String]
    public var expertTensorNames: [String]
    public var nExpertsTotal: Int
    public var hotCoreBytes: Int
    public var expertBytes: Int
    public var hasDraft: Bool
    public var hasRouterPrior: Bool
    public var draftJang: String
    public var toolVersion: String
    public var schema: String

    enum CodingKeys: String, CodingKey {
        case bundleVersion = "bundle_version"
        case sourceJang = "source_jang"
        case sourceJangDir = "source_jang_dir"
        case targetArch = "target_arch"
        case nLayers = "n_layers"
        case nExpertsPerLayer = "n_experts_per_layer"
        case targetTopK = "target_top_k"
        case tokenizerHash = "tokenizer_hash"
        case hotCoreTensors = "hot_core_tensors"
        case expertTensorNames = "expert_tensor_names"
        case nExpertsTotal = "n_experts_total"
        case hotCoreBytes = "hot_core_bytes"
        case expertBytes = "expert_bytes"
        case hasDraft = "has_draft"
        case hasRouterPrior = "has_router_prior"
        case draftJang = "draft_jang"
        case toolVersion = "tool_version"
        case schema
    }

    public static func load(from url: URL) throws -> JangSpecManifest {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let m = try decoder.decode(JangSpecManifest.self, from: data)
        guard m.bundleVersion == JangSpecFormat.bundleVersion else {
            throw JangSpecError.unsupportedVersion(
                field: "bundle",
                value: m.bundleVersion,
                supported: JangSpecFormat.bundleVersion
            )
        }
        return m
    }
}
