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

    /// Leaf-consumer UserDefaults keys for settings that the child Python
    /// subprocess consumes via environment variables. AppSettings mirrors
    /// these on every persist/load. Introduced iter 11 to close the M62
    /// env-passthrough batch (tickThrottleMs / mlxThreadCount / customJangToolsPath).
    static let tickThrottleMsDefaultsKey = "JANGStudioTickThrottleMs"
    static let mlxThreadCountDefaultsKey = "JANGStudioMLXThreadCount"
    static let customJangToolsPathDefaultsKey = "JANGStudioCustomJangToolsPath"

    /// Compose the env-var dictionary the child Python subprocess should inherit,
    /// merging user settings on top of the inherited process env. Callers merge
    /// the return value into their own `proc.environment` without clobbering
    /// the PYTHONUNBUFFERED / PYTHONNOUSERSITE the runners already set.
    ///
    /// Invariants:
    /// - An empty / zero / unset setting produces NO env var (fall through to
    ///   Python defaults). Zero-ing out a value in the UI shouldn't accidentally
    ///   set `OMP_NUM_THREADS=0` and wedge the process.
    /// - PYTHONPATH prepends the custom jang_tools path rather than replacing,
    ///   so bundle jang_tools still resolves for everything the custom path
    ///   doesn't override.
    static func childProcessEnvAdditions(
        inherited: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String: String] {
        var additions: [String: String] = [:]
        let defaults = UserDefaults.standard

        let throttle = defaults.integer(forKey: tickThrottleMsDefaultsKey)
        if throttle > 0 {
            additions["JANG_TICK_THROTTLE_MS"] = String(throttle)
        }

        let threads = defaults.integer(forKey: mlxThreadCountDefaultsKey)
        if threads > 0 {
            // MLX and underlying BLAS both honour these. Write both so the
            // user doesn't have to know which layer consumes which.
            additions["OMP_NUM_THREADS"] = String(threads)
            additions["MLX_NUM_THREADS"] = String(threads)
        }

        if let custom = defaults.string(forKey: customJangToolsPathDefaultsKey), !custom.isEmpty {
            let existingPythonPath = inherited["PYTHONPATH"] ?? ""
            additions["PYTHONPATH"] = existingPythonPath.isEmpty
                ? custom
                : "\(custom):\(existingPythonPath)"
        }

        return additions
    }

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
