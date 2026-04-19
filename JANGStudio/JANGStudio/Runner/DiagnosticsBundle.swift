// JANGStudio/JANGStudio/Runner/DiagnosticsBundle.swift
import Foundation

@MainActor
enum DiagnosticsBundle {
    /// Writes plan.json, run.log, events.jsonl, system.json, verify.json into a
    /// temp directory, then zips via `ditto -c -k`. Returns the final zip URL.
    static func write(plan: ConversionPlan,
                      logLines: [String],
                      eventLines: [String],
                      verify: [VerifyCheck],
                      to desktop: URL) throws -> URL {
        let stamp = ISO8601DateFormatter().string(from: Date()).replacingOccurrences(of: ":", with: "-")
        let workDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("JANGStudio-diag-\(stamp)")
        try FileManager.default.createDirectory(at: workDir, withIntermediateDirectories: true)

        try JSONEncoder().encode(plan).write(to: workDir.appendingPathComponent("plan.json"))
        try logLines.joined(separator: "\n").write(to: workDir.appendingPathComponent("run.log"),
                                                    atomically: true, encoding: .utf8)
        try eventLines.joined(separator: "\n").write(to: workDir.appendingPathComponent("events.jsonl"),
                                                      atomically: true, encoding: .utf8)
        let sys: [String: String] = [
            "macos": ProcessInfo.processInfo.operatingSystemVersionString,
            "ram_bytes": String(ProcessInfo.processInfo.physicalMemory),
            "app_version": (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String) ?? "?",
        ]
        try JSONSerialization.data(withJSONObject: sys).write(to: workDir.appendingPathComponent("system.json"))
        let verifyData = verify.map { ["id": $0.id.rawValue, "status": $0.status.rawValue, "required": $0.required] as [String: Any] }
        try JSONSerialization.data(withJSONObject: verifyData).write(to: workDir.appendingPathComponent("verify.json"))

        let zipURL = desktop.appendingPathComponent("JANGStudio-diagnostics-\(stamp).zip")
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        p.arguments = ["-c", "-k", "--keepParent", workDir.path, zipURL.path]
        try p.run(); p.waitUntilExit()
        try? FileManager.default.removeItem(at: workDir)
        return zipURL
    }
}
