// JANGStudio/JANGStudio/Runner/BundleResolver.swift
import Foundation

enum BundleResolver {
    /// Path to the bundled CPython 3.11 interpreter.
    /// Honors $JANGSTUDIO_PYTHON_OVERRIDE for local dev (points at your homebrew python3).
    /// See CONTRIBUTING.md "Dev mode".
    static var pythonExecutable: URL {
        if let override = ProcessInfo.processInfo.environment["JANGSTUDIO_PYTHON_OVERRIDE"],
           !override.isEmpty {
            return URL(fileURLWithPath: override)
        }
        return Bundle.main.bundleURL
            .appendingPathComponent("Contents/Resources/python/bin/python3")
    }

    /// True if the resolved python exists and is executable.
    static func healthCheck() -> Bool {
        let url = pythonExecutable
        return FileManager.default.isExecutableFile(atPath: url.path)
    }
}
