import Foundation

/// Builds a Gemma-4-26B `.jangspec` fixture bundle via the Python
/// `jang spec build` CLI. The bundle is cached under
/// `/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec` and reused
/// across test runs as long as the manifest is present.
enum Fixtures {
    /// Override with JANG_TEST_GEMMA4_BUNDLE env var. Otherwise resolves to
    /// `~/jang/models/Gemma-4-26B-A4B-it-JANG_4M` and skips the test if absent.
    static let sourceModelPath: String = {
        if let env = ProcessInfo.processInfo.environment["JANG_TEST_GEMMA4_BUNDLE"] {
            return env
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent("jang/models/Gemma-4-26B-A4B-it-JANG_4M").path
    }()
    static let cacheDir = URL(fileURLWithPath: "/tmp/jangcore-fixtures")
    static let bundleURL = cacheDir.appendingPathComponent("Gemma-4-26B-A4B-it-JANG_4M.jangspec")

    static func gemmaBundle() throws -> URL {
        let manifest = bundleURL.appendingPathComponent("jangspec.json")
        if FileManager.default.fileExists(atPath: manifest.path) {
            return bundleURL
        }
        guard FileManager.default.fileExists(atPath: sourceModelPath) else {
            throw NSError(
                domain: "Fixtures",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Gemma-4-26B-A4B-it-JANG_4M fixture model not found at \(sourceModelPath). Set JANGSPEC_TEST_MODEL env var if using a different path (fixture helper does not read it yet)."]
            )
        }
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let proc = Process()
        proc.launchPath = "/bin/bash"
        proc.arguments = [
            "-c",
            "jang spec build '\(sourceModelPath)' --out '\(bundleURL.path)' --force"
        ]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else {
            throw NSError(
                domain: "Fixtures",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "jang spec build failed with exit code \(proc.terminationStatus)"]
            )
        }
        return bundleURL
    }
}
