// JANGStudio/JANGStudio/Runner/BundleResolver.swift
import Foundation

enum BundleResolver {
    /// Dedicated UserDefaults key for the user-configured Python path. AppSettings
    /// mirrors `pythonOverridePath` here on every persist() so BundleResolver
    /// (nonisolated, called from PythonRunner / InferenceRunner / etc.) can read
    /// it without depending on the @MainActor AppSettings instance.
    /// Introduced iter 9 to close M61 — prior to this, the Advanced-tab picker
    /// was pure decoration: users set a path, nothing happened.
    static let pythonOverrideDefaultsKey = "JANGStudioPythonOverride"

    /// Path to the Python interpreter. Priority (first non-empty wins):
    /// 1. User setting from AppSettings (via UserDefaults key above)
    /// 2. `$JANGSTUDIO_PYTHON_OVERRIDE` env var (CONTRIBUTING.md dev mode)
    /// 3. Bundled CPython inside the .app
    static var pythonExecutable: URL {
        if let userDefault = UserDefaults.standard.string(forKey: pythonOverrideDefaultsKey),
           !userDefault.isEmpty {
            return URL(fileURLWithPath: userDefault)
        }
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
