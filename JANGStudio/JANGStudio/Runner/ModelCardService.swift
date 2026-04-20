import Foundation

struct ModelCardResult: Codable, Equatable {
    let license: String
    let baseModel: String
    let quantizationConfig: QuantizationConfig
    let cardMarkdown: String

    struct QuantizationConfig: Codable, Equatable {
        let family: String
        let profile: String
        let actualBits: Double
        let blockSize: Int?
        let sizeGb: Double?

        enum CodingKeys: String, CodingKey {
            case family, profile
            case actualBits = "actual_bits"
            case blockSize = "block_size"
            case sizeGb = "size_gb"
        }
    }

    enum CodingKeys: String, CodingKey {
        case license
        case baseModel = "base_model"
        case quantizationConfig = "quantization_config"
        case cardMarkdown = "card_markdown"
    }
}

enum ModelCardServiceError: Error, LocalizedError {
    case cliError(code: Int32, stderr: String)
    case decodeError(String)

    var errorDescription: String? {
        switch self {
        case .cliError(let c, let s):
            return "jang-tools modelcard exited \(c): \(s.trimmingCharacters(in: .whitespacesAndNewlines))"
        case .decodeError(let s): return s
        }
    }
}

@MainActor
enum ModelCardService {
    static func generate(modelPath: URL) async throws -> ModelCardResult {
        let data = try await invoke(args: [
            "-m", "jang_tools", "modelcard",
            "--model", modelPath.path,
            "--json",
        ])
        do {
            return try JSONDecoder().decode(ModelCardResult.self, from: data)
        } catch {
            throw ModelCardServiceError.decodeError("\(error)")
        }
    }

    /// Write the card to a README.md inside the model dir.
    static func writeReadme(modelPath: URL, content: String) throws {
        let readme = modelPath.appendingPathComponent("README.md")
        try content.data(using: .utf8)?.write(to: readme, options: .atomic)
    }

    private nonisolated static func invoke(args: [String]) async throws -> Data {
        // M153 (iter 76): migrated to shared PythonCLIInvoker.
        // The M101 (iter 33) Task-cancel pattern now lives in the shared helper.
        try await PythonCLIInvoker.invoke(args: args) { code, stderr in
            ModelCardServiceError.cliError(code: code, stderr: stderr)
        }
    }
}
