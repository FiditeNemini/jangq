// JANGStudio/JANGStudio/Models/AppSettings.swift
import Foundation
import Observation

enum LogVerbosity: String, Codable, CaseIterable, Identifiable {
    case normal, verbose, debug
    var id: String { rawValue }
    var displayName: String {
        switch self {
        case .normal: return "Normal"
        case .verbose: return "Verbose"
        case .debug: return "Debug"
        }
    }
}

enum UpdateChannel: String, Codable, CaseIterable, Identifiable {
    case stable, beta
    var id: String { rawValue }
    var displayName: String { rawValue.capitalized }
}

/// All user-configurable settings. Persisted to UserDefaults under "JANGStudioSettings".
/// Every value has a sensible default and can be reset via `reset()`.
@Observable
@MainActor
final class AppSettings {
    // MARK: - General
    var defaultOutputParentPath: String = ""   // empty = use source's parent
    var defaultProfile: String = "JANG_4K"
    var defaultFamily: String = "jang"
    var defaultMethod: String = "mse"
    var defaultHadamardEnabled: Bool = false
    var defaultCalibrationSamples: Int = 256
    var outputNamingTemplate: String = "{basename}-{profile}"
    var autoDeletePartialOnCancel: Bool = false
    var revealInFinderOnFinish: Bool = true

    /// Default HuggingFace org/user for publish. Empty = user must type the
    /// full `org/name` each time. When non-empty, PublishToHuggingFaceSheet
    /// pre-fills the repo field with `{defaultHFOrg}/{modelBasename}`.
    /// Introduced iter 25 (M48) — prior default was basename-only which
    /// always failed HFRepoValidator ("must be org/model-name, one slash").
    var defaultHFOrg: String = ""

    // MARK: - Advanced
    var pythonOverridePath: String = ""   // empty = use bundled
    var customJangToolsPath: String = ""  // empty = use bundled jang_tools
    var logVerbosity: LogVerbosity = .normal
    var jsonlLogRetentionLines: Int = 10_000
    var logFileOutputDir: String = ""     // empty = ~/Library/Logs/JANGStudio
    var tickThrottleMs: Int = 100
    var maxBundleSizeWarningMb: Int = 450

    // MARK: - Performance
    var mlxThreadCount: Int = 0            // 0 = auto (system cpu count)
    var metalPipelineCacheEnabled: Bool = true
    var preAllocateRam: Bool = false
    var preAllocateRamGb: Int = 4
    var convertConcurrency: Int = 1

    // MARK: - Diagnostics
    var copyDiagnosticsAlwaysVisible: Bool = true
    var anonymizePathsInDiagnostics: Bool = false
    var githubIssuesUrl: String = "https://github.com/jjang-ai/jangq/issues"
    var autoOpenIssueTrackerOnCrash: Bool = false

    // MARK: - Updates
    var updateChannel: UpdateChannel = .stable
    var autoCheckForUpdates: Bool = true

    private static let defaultsKey = "JANGStudioSettings"

    init() {
        load()
    }

    /// Reset every field to its initial default.
    func reset() {
        defaultOutputParentPath = ""
        defaultProfile = "JANG_4K"
        defaultFamily = "jang"
        defaultMethod = "mse"
        defaultHadamardEnabled = false
        defaultCalibrationSamples = 256
        outputNamingTemplate = "{basename}-{profile}"
        autoDeletePartialOnCancel = false
        revealInFinderOnFinish = true
        defaultHFOrg = ""
        pythonOverridePath = ""
        customJangToolsPath = ""
        logVerbosity = .normal
        jsonlLogRetentionLines = 10_000
        logFileOutputDir = ""
        tickThrottleMs = 100
        maxBundleSizeWarningMb = 450
        mlxThreadCount = 0
        metalPipelineCacheEnabled = true
        preAllocateRam = false
        preAllocateRamGb = 4
        convertConcurrency = 1
        copyDiagnosticsAlwaysVisible = true
        anonymizePathsInDiagnostics = false
        githubIssuesUrl = "https://github.com/jjang-ai/jangq/issues"
        autoOpenIssueTrackerOnCrash = false
        updateChannel = .stable
        autoCheckForUpdates = true
        persist()
    }

    /// Expand the output naming template with actual values.
    /// Supported tokens: {basename}, {profile}, {family}, {date}, {time}, {user}
    func renderOutputName(basename: String, profile: String, family: String) -> String {
        let date = ISO8601DateFormatter().string(from: Date()).prefix(10)
        let time = ISO8601DateFormatter().string(from: Date()).suffix(8).prefix(5)
        let user = ProcessInfo.processInfo.environment["USER"] ?? "user"
        return outputNamingTemplate
            .replacingOccurrences(of: "{basename}", with: basename)
            .replacingOccurrences(of: "{profile}", with: profile)
            .replacingOccurrences(of: "{family}", with: family)
            .replacingOccurrences(of: "{date}", with: String(date))
            .replacingOccurrences(of: "{time}", with: String(time))
            .replacingOccurrences(of: "{user}", with: user)
    }

    func persist() {
        let snapshot = Snapshot(from: self)
        do {
            let data = try JSONEncoder().encode(snapshot)
            UserDefaults.standard.set(data, forKey: Self.defaultsKey)
        } catch {
            // M111 (iter 37): previously `try?` silently swallowed encode
            // failures — if encoding broke (schema migration edge-case,
            // non-finite Double slipping in via a future Settings field),
            // the user's settings wouldn't persist and nobody would know
            // why. UserDefaults keeps the old blob intact (we didn't call
            // .set on failure) so data-loss is bounded, but visibility is
            // critical for diagnostics. Log to stderr so Copy Diagnostics
            // captures it (per M107 pattern + iter 14's scrubSensitive
            // pipeline).
            FileHandle.standardError.write(
                Data("[AppSettings] persist failed: \(error)\n".utf8))
        }
        mirrorLeafConsumerKeys()
    }

    private func load() {
        // M147 (iter 69): distinguish "nothing saved yet" (first launch —
        // OK silent) from "saved data failed to decode" (schema migration
        // or corrupted blob — warrants logging to match persist()'s
        // symmetric stderr logging, iter-37 M111). Pre-M147 the combined
        // `guard let data = …, let s = try? decode…` silently abandoned
        // the user's saved settings on ANY decode failure. They'd launch,
        // see fresh defaults, and have no signal that their customization
        // was lost. Copy Diagnostics (iter-14 M22 pipeline) captures
        // stderr, so logging here surfaces in bug reports.
        guard let data = UserDefaults.standard.data(forKey: Self.defaultsKey) else {
            return   // first launch — no error, just no data
        }
        let s: Snapshot
        do {
            s = try JSONDecoder().decode(Snapshot.self, from: data)
        } catch {
            FileHandle.standardError.write(
                Data("[AppSettings] load failed (settings decode error — using defaults): \(error)\n".utf8))
            return
        }
        s.apply(to: self)
        // Re-sync the dedicated leaf-consumer mirrors after loading — otherwise
        // a fresh process start wouldn't pick up the saved python override
        // until the user touched Settings.
        mirrorLeafConsumerKeys()
    }

    /// Copy settings that leaf consumers (BundleResolver, future non-MainActor
    /// services) need into dedicated UserDefaults keys. These consumers can't
    /// hold an AppSettings reference (isolation + lifecycle issues), so they
    /// read a single typed key instead of decoding the whole Snapshot blob.
    private func mirrorLeafConsumerKeys() {
        let defaults = UserDefaults.standard
        // Python override — BundleResolver reads this. Remove the key when empty
        // so the env-var / bundled fallbacks take over cleanly.
        if pythonOverridePath.isEmpty {
            defaults.removeObject(forKey: BundleResolver.pythonOverrideDefaultsKey)
        } else {
            defaults.set(pythonOverridePath, forKey: BundleResolver.pythonOverrideDefaultsKey)
        }

        // Child-process env passthrough (M62 — iter 11). Only non-default
        // values get mirrored so BundleResolver.childProcessEnvAdditions can
        // treat "missing key" and "default value" as the same "fall through
        // to Python default" state.
        if tickThrottleMs == 100 {
            defaults.removeObject(forKey: BundleResolver.tickThrottleMsDefaultsKey)
        } else {
            defaults.set(tickThrottleMs, forKey: BundleResolver.tickThrottleMsDefaultsKey)
        }

        if mlxThreadCount == 0 {
            defaults.removeObject(forKey: BundleResolver.mlxThreadCountDefaultsKey)
        } else {
            defaults.set(mlxThreadCount, forKey: BundleResolver.mlxThreadCountDefaultsKey)
        }

        if customJangToolsPath.isEmpty {
            defaults.removeObject(forKey: BundleResolver.customJangToolsPathDefaultsKey)
        } else {
            defaults.set(customJangToolsPath, forKey: BundleResolver.customJangToolsPathDefaultsKey)
        }
    }
}

// MARK: - Snapshot for UserDefaults persistence

private struct Snapshot: Codable {
    var defaultOutputParentPath: String
    var defaultProfile: String
    var defaultFamily: String
    var defaultMethod: String
    var defaultHadamardEnabled: Bool
    var defaultCalibrationSamples: Int
    var outputNamingTemplate: String
    var autoDeletePartialOnCancel: Bool
    var revealInFinderOnFinish: Bool
    var defaultHFOrg: String = ""   // iter-25 field; default for pre-iter-25 snapshots
    var pythonOverridePath: String
    var customJangToolsPath: String
    var logVerbosity: String
    var jsonlLogRetentionLines: Int
    var logFileOutputDir: String
    var tickThrottleMs: Int
    var maxBundleSizeWarningMb: Int
    var mlxThreadCount: Int
    var metalPipelineCacheEnabled: Bool
    var preAllocateRam: Bool
    var preAllocateRamGb: Int
    var convertConcurrency: Int
    var copyDiagnosticsAlwaysVisible: Bool
    var anonymizePathsInDiagnostics: Bool
    var githubIssuesUrl: String
    var autoOpenIssueTrackerOnCrash: Bool
    var updateChannel: String
    var autoCheckForUpdates: Bool

    @MainActor init(from s: AppSettings) {
        defaultOutputParentPath = s.defaultOutputParentPath
        defaultProfile = s.defaultProfile
        defaultFamily = s.defaultFamily
        defaultMethod = s.defaultMethod
        defaultHadamardEnabled = s.defaultHadamardEnabled
        defaultCalibrationSamples = s.defaultCalibrationSamples
        outputNamingTemplate = s.outputNamingTemplate
        autoDeletePartialOnCancel = s.autoDeletePartialOnCancel
        revealInFinderOnFinish = s.revealInFinderOnFinish
        defaultHFOrg = s.defaultHFOrg
        pythonOverridePath = s.pythonOverridePath
        customJangToolsPath = s.customJangToolsPath
        logVerbosity = s.logVerbosity.rawValue
        jsonlLogRetentionLines = s.jsonlLogRetentionLines
        logFileOutputDir = s.logFileOutputDir
        tickThrottleMs = s.tickThrottleMs
        maxBundleSizeWarningMb = s.maxBundleSizeWarningMb
        mlxThreadCount = s.mlxThreadCount
        metalPipelineCacheEnabled = s.metalPipelineCacheEnabled
        preAllocateRam = s.preAllocateRam
        preAllocateRamGb = s.preAllocateRamGb
        convertConcurrency = s.convertConcurrency
        copyDiagnosticsAlwaysVisible = s.copyDiagnosticsAlwaysVisible
        anonymizePathsInDiagnostics = s.anonymizePathsInDiagnostics
        githubIssuesUrl = s.githubIssuesUrl
        autoOpenIssueTrackerOnCrash = s.autoOpenIssueTrackerOnCrash
        updateChannel = s.updateChannel.rawValue
        autoCheckForUpdates = s.autoCheckForUpdates
    }

    @MainActor func apply(to s: AppSettings) {
        s.defaultOutputParentPath = defaultOutputParentPath
        s.defaultProfile = defaultProfile
        s.defaultFamily = defaultFamily
        s.defaultMethod = defaultMethod
        s.defaultHadamardEnabled = defaultHadamardEnabled
        s.defaultCalibrationSamples = defaultCalibrationSamples
        s.outputNamingTemplate = outputNamingTemplate
        s.autoDeletePartialOnCancel = autoDeletePartialOnCancel
        s.revealInFinderOnFinish = revealInFinderOnFinish
        s.defaultHFOrg = defaultHFOrg
        s.pythonOverridePath = pythonOverridePath
        s.customJangToolsPath = customJangToolsPath
        // M66 (iter 96): surface stale-UserDefaults coercion to stderr.
        // Pre-M66, `LogVerbosity(rawValue: logVerbosity) ?? .normal`
        // silently reset the user's custom setting to the default when
        // the persisted string didn't match any enum case (triggered by:
        // schema renames in app updates, cross-version downgrades with
        // newer enum cases, or manual `defaults write` with a typo).
        // User saw default behavior with no hint why. iter-35 M107 /
        // iter-80 M157 pattern: log to stderr so Copy Diagnostics picks
        // it up. Apply the default AFTER logging.
        if let parsed = LogVerbosity(rawValue: logVerbosity) {
            s.logVerbosity = parsed
        } else {
            FileHandle.standardError.write(Data(
                "[AppSettings] logVerbosity=\"\(logVerbosity)\" is not a valid case; coercing to .normal. If you recently downgraded JANG Studio, re-save your preferred verbosity in Settings → Diagnostics.\n".utf8))
            s.logVerbosity = .normal
        }
        s.jsonlLogRetentionLines = jsonlLogRetentionLines
        s.logFileOutputDir = logFileOutputDir
        s.tickThrottleMs = tickThrottleMs
        s.maxBundleSizeWarningMb = maxBundleSizeWarningMb
        s.mlxThreadCount = mlxThreadCount
        s.metalPipelineCacheEnabled = metalPipelineCacheEnabled
        s.preAllocateRam = preAllocateRam
        s.preAllocateRamGb = preAllocateRamGb
        s.convertConcurrency = convertConcurrency
        s.copyDiagnosticsAlwaysVisible = copyDiagnosticsAlwaysVisible
        s.anonymizePathsInDiagnostics = anonymizePathsInDiagnostics
        s.githubIssuesUrl = githubIssuesUrl
        s.autoOpenIssueTrackerOnCrash = autoOpenIssueTrackerOnCrash
        // M66 (iter 96): same silent-coercion surface as logVerbosity above.
        if let parsed = UpdateChannel(rawValue: updateChannel) {
            s.updateChannel = parsed
        } else {
            FileHandle.standardError.write(Data(
                "[AppSettings] updateChannel=\"\(updateChannel)\" is not a valid case; coercing to .stable. Re-save in Settings → Updates if this is unexpected.\n".utf8))
            s.updateChannel = .stable
        }
        s.autoCheckForUpdates = autoCheckForUpdates
    }
}
