// JANGStudio/JANGStudio/Wizard/SettingsWindow.swift
import SwiftUI

struct SettingsWindow: View {
    @Bindable var settings: AppSettings
    @Environment(ProfilesService.self) private var profilesSvc
    @Environment(CapabilitiesService.self) private var capsSvc

    var body: some View {
        TabView {
            GeneralTab(settings: settings, profilesSvc: profilesSvc, capsSvc: capsSvc)
                .tabItem { Label("General", systemImage: "gear") }

            AdvancedTab(settings: settings)
                .tabItem { Label("Advanced", systemImage: "gearshape.2") }

            PerformanceTab(settings: settings)
                .tabItem { Label("Performance", systemImage: "speedometer") }

            DiagnosticsTab(settings: settings)
                .tabItem { Label("Diagnostics", systemImage: "stethoscope") }

            UpdatesTab(settings: settings)
                .tabItem { Label("Updates", systemImage: "arrow.triangle.2.circlepath") }
        }
        .frame(width: 600, height: 480)
        .task { await observeAndPersist(settings) }
    }
}

// MARK: - General tab

private struct GeneralTab: View {
    @Bindable var settings: AppSettings
    let profilesSvc: ProfilesService
    let capsSvc: CapabilitiesService

    var body: some View {
        Form {
            Section("Defaults") {
                HStack {
                    Text("Output folder parent")
                    Spacer()
                    Text(settings.defaultOutputParentPath.isEmpty
                         ? "Same as source"
                         : settings.defaultOutputParentPath)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button("Choose…") { pickDir { settings.defaultOutputParentPath = $0 } }
                    if !settings.defaultOutputParentPath.isEmpty {
                        Button("Clear") { settings.defaultOutputParentPath = "" }
                    }
                }

                Picker("Default profile", selection: $settings.defaultProfile) {
                    ForEach(profilesSvc.profiles.jang) { p in
                        Text(p.name).tag(p.name)
                    }
                }

                Picker("Default family", selection: $settings.defaultFamily) {
                    Text("JANG").tag("jang")
                    Text("JANGTQ (auto when supported)").tag("jangtq")
                }

                Picker("Default method", selection: $settings.defaultMethod) {
                    ForEach(capsSvc.capabilities.methods, id: \.name) { m in
                        Text(m.name.uppercased()).tag(m.name)
                    }
                }

                Toggle("Enable Hadamard rotation by default", isOn: $settings.defaultHadamardEnabled)
            }

            Section("Calibration") {
                Stepper(value: $settings.defaultCalibrationSamples, in: 64...1024, step: 64) {
                    LabeledContent("Sample count", value: "\(settings.defaultCalibrationSamples)")
                }
            }

            Section("Output naming") {
                TextField("Template", text: $settings.outputNamingTemplate)
                    .textFieldStyle(.roundedBorder)
                Text("Tokens: {basename} {profile} {family} {date} {time} {user}")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("Preview: \(settings.renderOutputName(basename: "my-model", profile: settings.defaultProfile, family: settings.defaultFamily))")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            Section("Behavior") {
                Toggle("Auto-delete partial output on cancel", isOn: $settings.autoDeletePartialOnCancel)
                Toggle("Reveal in Finder on finish", isOn: $settings.revealInFinderOnFinish)
            }

            Section {
                HStack {
                    Spacer()
                    Button("Reset to defaults") { settings.reset() }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func pickDir(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            completion(url.path)
        }
    }
}

// MARK: - Advanced tab

private struct AdvancedTab: View {
    @Bindable var settings: AppSettings

    var body: some View {
        Form {
            Section("Runtime overrides") {
                HStack {
                    Text("Python override")
                    Spacer()
                    Text(settings.pythonOverridePath.isEmpty
                         ? "Use bundled"
                         : settings.pythonOverridePath)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button("Choose…") { pickFile { settings.pythonOverridePath = $0 } }
                    if !settings.pythonOverridePath.isEmpty {
                        Button("Clear") { settings.pythonOverridePath = "" }
                    }
                }
                Text("Path to a Python 3.11+ interpreter. Overrides $JANGSTUDIO_PYTHON_OVERRIDE at app scope.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Custom jang-tools path")
                    Spacer()
                    Text(settings.customJangToolsPath.isEmpty ? "Use bundled" : settings.customJangToolsPath)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button("Choose…") { pickDir { settings.customJangToolsPath = $0 } }
                    if !settings.customJangToolsPath.isEmpty {
                        Button("Clear") { settings.customJangToolsPath = "" }
                    }
                }
            }

            Section("Logging") {
                Picker("Verbosity", selection: $settings.logVerbosity) {
                    ForEach(LogVerbosity.allCases) { v in
                        Text(v.displayName).tag(v)
                    }
                }
                Stepper(value: $settings.jsonlLogRetentionLines, in: 1000...50000, step: 1000) {
                    LabeledContent("UI log ring size", value: "\(settings.jsonlLogRetentionLines) lines")
                }

                HStack {
                    Text("Log file dir")
                    Spacer()
                    Text(settings.logFileOutputDir.isEmpty
                         ? "~/Library/Logs/JANGStudio"
                         : settings.logFileOutputDir)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Button("Choose…") { pickDir { settings.logFileOutputDir = $0 } }
                    if !settings.logFileOutputDir.isEmpty {
                        Button("Clear") { settings.logFileOutputDir = "" }
                    }
                }
            }

            Section("Throttling") {
                Stepper(value: $settings.tickThrottleMs, in: 50...500, step: 10) {
                    LabeledContent("Tick throttle", value: "\(settings.tickThrottleMs) ms")
                }
                Text("Controls how often Python emits JSONL tick events. Higher = less UI churn on long runs.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section("Bundle") {
                Stepper(value: $settings.maxBundleSizeWarningMb, in: 200...1000, step: 50) {
                    LabeledContent("Max bundle size warning", value: "\(settings.maxBundleSizeWarningMb) MB")
                }
                Text("If the embedded Python bundle exceeds this, the build script warns. CI enforces 450 MB.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func pickFile(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { completion(url.path) }
    }

    private func pickDir(_ completion: @escaping (String) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        if panel.runModal() == .OK, let url = panel.url { completion(url.path) }
    }
}

// MARK: - Performance tab

private struct PerformanceTab: View {
    @Bindable var settings: AppSettings
    private let maxThreads = ProcessInfo.processInfo.processorCount

    var body: some View {
        Form {
            Section("Threading") {
                Stepper(value: $settings.mlxThreadCount, in: 0...maxThreads) {
                    LabeledContent("MLX thread count",
                                   value: settings.mlxThreadCount == 0
                                   ? "Auto (\(maxThreads))"
                                   : "\(settings.mlxThreadCount)")
                }
                Text("0 = automatic. System has \(maxThreads) cores.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section("Metal") {
                Toggle("Enable Metal pipeline cache", isOn: $settings.metalPipelineCacheEnabled)
                Text("Caches compiled Metal kernels on disk. Speeds up re-launches after app update.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section("Memory") {
                Toggle("Pre-allocate RAM at launch", isOn: $settings.preAllocateRam)
                if settings.preAllocateRam {
                    Stepper(value: $settings.preAllocateRamGb, in: 1...128) {
                        LabeledContent("Pre-allocate", value: "\(settings.preAllocateRamGb) GB")
                    }
                }
                Text("Reserves RAM up-front. Useful on machines with hungry background apps.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section("Concurrency") {
                Stepper(value: $settings.convertConcurrency, in: 1...4) {
                    LabeledContent("Concurrent conversions", value: "\(settings.convertConcurrency)")
                }
                if settings.convertConcurrency > 1 {
                    Label("Experimental — high RAM pressure with parallel conversions.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Diagnostics tab

private struct DiagnosticsTab: View {
    @Bindable var settings: AppSettings

    var body: some View {
        Form {
            Section("Diagnostics UX") {
                Toggle("Always show Copy Diagnostics button", isOn: $settings.copyDiagnosticsAlwaysVisible)
                Text("When on, the Copy Diagnostics button appears on every run — not just failed ones.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)

                Toggle("Anonymize paths in diagnostics", isOn: $settings.anonymizePathsInDiagnostics)
                Text("Replaces source / output paths with their basenames so bug reports don't leak filesystem layout.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Section("Bug reporting") {
                TextField("GitHub issues URL", text: $settings.githubIssuesUrl)
                    .textFieldStyle(.roundedBorder)
                    .disableAutocorrection(true)
                Toggle("Auto-open issue tracker on crash", isOn: $settings.autoOpenIssueTrackerOnCrash)
            }

            Section {
                Button {
                    openLogsDirectory()
                } label: {
                    Label("Open logs directory", systemImage: "folder")
                }
                Button {
                    copySystemInfo()
                } label: {
                    Label("Copy system info", systemImage: "info.circle")
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func openLogsDirectory() {
        let dir = settings.logFileOutputDir.isEmpty
            ? FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Logs/JANGStudio")
            : URL(fileURLWithPath: settings.logFileOutputDir)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        NSWorkspace.shared.open(dir)
    }

    private func copySystemInfo() {
        let version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let settingsSet = UserDefaults.standard.data(forKey: "JANGStudioSettings") != nil
        let info = """
            macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)
            RAM: \(ProcessInfo.processInfo.physicalMemory / 1_000_000_000) GB
            CPU cores: \(ProcessInfo.processInfo.processorCount)
            App version: \(version)
            Settings: \(settingsSet ? "user defaults" : "not set")
            """
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(info, forType: .string)
    }
}

// MARK: - Updates tab

private struct UpdatesTab: View {
    @Bindable var settings: AppSettings
    var body: some View {
        Form {
            Text("Updates settings — see commit 5")
                .foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Auto-persist helper

/// Drives persistence whenever any observed property of `settings` changes.
/// Uses `withObservationTracking` in a loop — reads every property once to
/// register tracking, then fires `persist()` on the main actor on each change.
@MainActor
private func observeAndPersist(_ settings: AppSettings) async {
    while !Task.isCancelled {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            withObservationTracking {
                _ = settings.defaultOutputParentPath
                _ = settings.defaultProfile
                _ = settings.defaultFamily
                _ = settings.defaultMethod
                _ = settings.defaultHadamardEnabled
                _ = settings.defaultCalibrationSamples
                _ = settings.outputNamingTemplate
                _ = settings.autoDeletePartialOnCancel
                _ = settings.revealInFinderOnFinish
                _ = settings.pythonOverridePath
                _ = settings.customJangToolsPath
                _ = settings.logVerbosity
                _ = settings.jsonlLogRetentionLines
                _ = settings.logFileOutputDir
                _ = settings.tickThrottleMs
                _ = settings.maxBundleSizeWarningMb
                _ = settings.mlxThreadCount
                _ = settings.metalPipelineCacheEnabled
                _ = settings.preAllocateRam
                _ = settings.preAllocateRamGb
                _ = settings.convertConcurrency
                _ = settings.copyDiagnosticsAlwaysVisible
                _ = settings.anonymizePathsInDiagnostics
                _ = settings.githubIssuesUrl
                _ = settings.autoOpenIssueTrackerOnCrash
                _ = settings.updateChannel
                _ = settings.autoCheckForUpdates
            } onChange: {
                Task { @MainActor in
                    settings.persist()
                    continuation.resume()
                }
            }
        }
    }
}
