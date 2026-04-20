// JANGStudio/JANGStudio/Runner/RecommendationService.swift
//
// Fetches beginner-friendly recommendations from `jang-tools recommend --json`.
// Used by SourceStep to auto-fill the wizard's subsequent steps with smart
// defaults — profile, family, method, hadamard, force_dtype — plus surface
// a plain-English beginner summary and any warnings.

import Foundation

struct Recommendation: Codable, Equatable {
    struct Detected: Codable, Equatable {
        let modelType: String
        let familyClass: String
        let paramCountBillions: Double
        let expertCount: Int
        let isMoE: Bool
        let isVL: Bool
        let isVideoVL: Bool
        let sourceDtype: String
        let hasToolParser: Bool
        let hasReasoningParser: Bool
        let isGatedModel: Bool
        let nameOrPath: String
    }

    struct Alternative: Codable, Equatable, Identifiable {
        let family: String?
        let profile: String
        let useWhen: String

        var id: String { "\(family ?? "jang")-\(profile)" }
    }

    struct Recommended: Codable, Equatable {
        let family: String
        let profile: String
        let method: String
        let hadamard: Bool
        let blockSize: Int
        let forceDtype: String?
        let alternatives: [Alternative]
    }

    struct WhyEachChoice: Codable, Equatable {
        let family: String
        let profile: String
        let method: String
        let hadamard: String
        let blockSize: String
        let forceDtype: String
    }

    let detected: Detected
    let recommended: Recommended
    let beginnerSummary: String
    let warnings: [String]
    let whyEachChoice: WhyEachChoice
}

enum RecommendationServiceError: Error, LocalizedError {
    case cliError(code: Int32, stderr: String)
    case decodeError(String)

    var errorDescription: String? {
        switch self {
        case .cliError(let code, let stderr):
            return "jang-tools recommend exited \(code): \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"
        case .decodeError(let m): return "couldn't decode recommend JSON: \(m)"
        }
    }
}

@MainActor
enum RecommendationService {
    /// Fetch the recommendation for a source model dir via `jang-tools recommend --json`.
    static func fetch(modelURL: URL) async throws -> Recommendation {
        let data = try await invoke(args: [
            "-m", "jang_tools", "recommend",
            "--model", modelURL.path,
            "--json",
        ])
        do {
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode(Recommendation.self, from: data)
        } catch {
            throw RecommendationServiceError.decodeError("\(error)")
        }
    }

    private nonisolated static func invoke(args: [String]) async throws -> Data {
        // M153 (iter 76): migrated from local copy to shared PythonCLIInvoker.
        // The iter-33 M101 Task-cancel propagation now lives in the shared
        // helper; this service only provides the typed-error factory.
        try await PythonCLIInvoker.invoke(args: args) { code, stderr in
            RecommendationServiceError.cliError(code: code, stderr: stderr)
        }
    }
}
