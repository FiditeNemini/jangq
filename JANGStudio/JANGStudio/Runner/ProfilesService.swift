// JANGStudio/JANGStudio/Runner/ProfilesService.swift
import Foundation
import Observation

struct ProfileInfo: Codable, Equatable, Identifiable, Hashable, Sendable {
    let name: String
    let criticalBits: Int?
    let importantBits: Int?
    let compressBits: Int?
    let avgBits: Double
    let description: String
    let isDefault: Bool
    let isKquant: Bool

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name
        case criticalBits = "critical_bits"
        case importantBits = "important_bits"
        case compressBits = "compress_bits"
        case avgBits = "avg_bits"
        case description
        case isDefault = "is_default"
        case isKquant = "is_kquant"
    }
}

struct JANGTQProfileInfo: Codable, Equatable, Identifiable, Hashable, Sendable {
    let name: String
    let bits: Int
    let minSourceDtype: [String]
    let description: String

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name
        case bits
        case minSourceDtype = "min_source_dtype"
        case description
    }
}

struct Profiles: Codable, Equatable, Sendable {
    let jang: [ProfileInfo]
    let jangtq: [JANGTQProfileInfo]
    let defaultProfile: String
    let bitToProfile: [String: String]

    enum CodingKeys: String, CodingKey {
        case jang, jangtq
        case defaultProfile = "default_profile"
        case bitToProfile = "bit_to_profile"
    }

    /// Frozen fallback. Must stay in sync with jang-tools/jang_tools/profiles_cli.py.
    static let frozen: Profiles = .init(
        jang: [
            .init(name: "JANG_1L", criticalBits: 8, importantBits: 8, compressBits: 2, avgBits: 2.6, description: "Maximum-protection 1-bit tier", isDefault: false, isKquant: false),
            .init(name: "JANG_2S", criticalBits: 6, importantBits: 4, compressBits: 2, avgBits: 2.5, description: "Tightest 2-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_2M", criticalBits: 8, importantBits: 4, compressBits: 2, avgBits: 2.7, description: "Balanced 2-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_2L", criticalBits: 8, importantBits: 6, compressBits: 2, avgBits: 2.9, description: "Best-quality 2-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_3S", criticalBits: 6, importantBits: 3, compressBits: 3, avgBits: 3.15, description: "Small boost on attention only", isDefault: false, isKquant: false),
            .init(name: "JANG_3M", criticalBits: 8, importantBits: 3, compressBits: 3, avgBits: 3.25, description: "Full attention at 8-bit, rest 3-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_3L", criticalBits: 8, importantBits: 4, compressBits: 3, avgBits: 3.4, description: "Attention 8-bit, embeddings 4-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_3K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 3.0, description: "K-quant 3-bit (budget-neutral)", isDefault: false, isKquant: true),
            .init(name: "JANG_4S", criticalBits: 6, importantBits: 4, compressBits: 4, avgBits: 4.1, description: "Small boost", isDefault: false, isKquant: false),
            .init(name: "JANG_4M", criticalBits: 8, importantBits: 4, compressBits: 4, avgBits: 4.2, description: "Full attention at 8-bit, rest 4-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_4L", criticalBits: 8, importantBits: 6, compressBits: 4, avgBits: 4.4, description: "Attention 8-bit, embeddings 6-bit", isDefault: false, isKquant: false),
            .init(name: "JANG_4K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 4.0, description: "K-quant 4-bit — THE DEFAULT", isDefault: true, isKquant: true),
            .init(name: "JANG_5K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 5.0, description: "K-quant 5-bit", isDefault: false, isKquant: true),
            .init(name: "JANG_6K", criticalBits: nil, importantBits: nil, compressBits: nil, avgBits: 6.0, description: "K-quant 6-bit", isDefault: false, isKquant: true),
            .init(name: "JANG_6M", criticalBits: 8, importantBits: 6, compressBits: 6, avgBits: 6.2, description: "Near-lossless", isDefault: false, isKquant: false),
        ],
        jangtq: [
            .init(name: "JANGTQ2", bits: 2, minSourceDtype: ["bfloat16", "float8_e4m3fn"], description: "2-bit TurboQuant"),
            .init(name: "JANGTQ3", bits: 3, minSourceDtype: ["bfloat16", "float8_e4m3fn"], description: "3-bit TurboQuant"),
            .init(name: "JANGTQ4", bits: 4, minSourceDtype: ["bfloat16", "float8_e4m3fn"], description: "4-bit TurboQuant — near-lossless"),
        ],
        defaultProfile: "JANG_4K",
        bitToProfile: ["1": "JANG_1L", "2": "JANG_2S", "3": "JANG_3K", "4": "JANG_4K", "5": "JANG_5K", "6": "JANG_6K", "7": "JANG_6M", "8": "JANG_6M"]
    )
}

@Observable
@MainActor
final class ProfilesService {
    private(set) var profiles: Profiles = .frozen
    private(set) var isFromBundle: Bool = false
    private(set) var lastError: String? = nil

    func refresh() async {
        if isFromBundle { return }
        do {
            let data = try await Self.invokeCLI(args: ["-m", "jang_tools", "profiles", "--json"])
            let decoded = try JSONDecoder().decode(Profiles.self, from: data)
            self.profiles = decoded
            self.isFromBundle = true
            self.lastError = nil
        } catch {
            self.lastError = "\(error)"
        }
    }

    private static func invokeCLI(args: [String]) async throws -> Data {
        // M101 (iter 33): Task-cancel propagation — see ModelCardService.invoke.
        let handle = ProcessHandle()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                DispatchQueue.global().async {
                    do {
                        let proc = Process()
                        proc.executableURL = BundleResolver.pythonExecutable
                        proc.arguments = args
                        let out = Pipe()
                        let err = Pipe()
                        proc.standardOutput = out
                        proc.standardError = err
                        try proc.run()
                        handle.set(process: proc)
                        proc.waitUntilExit()
                        if proc.terminationStatus != 0 {
                            let e = err.fileHandleForReading.readDataToEndOfFile()
                            throw NSError(domain: "ProfilesService", code: Int(proc.terminationStatus),
                                          userInfo: [NSLocalizedDescriptionKey: String(data: e, encoding: .utf8) ?? ""])
                        }
                        let data = out.fileHandleForReading.readDataToEndOfFile()
                        cont.resume(returning: data)
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            handle.cancel()
        }
    }
}
