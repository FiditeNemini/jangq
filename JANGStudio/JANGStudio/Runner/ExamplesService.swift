// JANGStudio/JANGStudio/Runner/ExamplesService.swift
import Foundation

enum ExampleLanguage: String, Codable, CaseIterable, Identifiable {
    case python, swift, server, hf

    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .python: return "Python"
        case .swift: return "Swift"
        case .server: return "Server"
        case .hf: return "HuggingFace"
        }
    }
    var fileExtension: String {
        switch self {
        case .python: return "py"
        case .swift: return "swift"
        case .server: return "sh"
        case .hf: return "md"
        }
    }
    var codeLang: String {   // for syntax highlighting hints
        switch self {
        case .python: return "python"
        case .swift: return "swift"
        case .server: return "bash"
        case .hf: return "markdown"
        }
    }
}

struct ExampleSnippet: Codable, Equatable {
    let lang: String
    let snippet: String
    let model: String
}

enum ExamplesServiceError: Error, LocalizedError {
    case cliError(code: Int32, stderr: String)
    case decodeError(String)

    var errorDescription: String? {
        switch self {
        case .cliError(let code, let stderr):
            return "jang-tools examples exited \(code): \(stderr.trimmingCharacters(in: .whitespacesAndNewlines))"
        case .decodeError(let msg):
            return "couldn't decode examples JSON: \(msg)"
        }
    }
}

@MainActor
enum ExamplesService {
    /// Fetch a snippet for the given model + language. Throws on CLI failure.
    static func fetch(modelPath: URL, lang: ExampleLanguage) async throws -> ExampleSnippet {
        let data = try await invoke(args: [
            "-m", "jang_tools", "examples",
            "--model", modelPath.path,
            "--lang", lang.rawValue,
            "--json",
        ])
        do {
            return try JSONDecoder().decode(ExampleSnippet.self, from: data)
        } catch {
            throw ExamplesServiceError.decodeError("\(error)")
        }
    }

    private nonisolated static func invoke(args: [String]) async throws -> Data {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global().async {
                do {
                    let proc = Process()
                    proc.executableURL = BundleResolver.pythonExecutable
                    proc.arguments = args
                    let out = Pipe(); let err = Pipe()
                    proc.standardOutput = out
                    proc.standardError = err
                    try proc.run()
                    proc.waitUntilExit()
                    if proc.terminationStatus != 0 {
                        let stderr = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
                        cont.resume(throwing: ExamplesServiceError.cliError(code: proc.terminationStatus, stderr: stderr))
                        return
                    }
                    cont.resume(returning: out.fileHandleForReading.readDataToEndOfFile())
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }
}
