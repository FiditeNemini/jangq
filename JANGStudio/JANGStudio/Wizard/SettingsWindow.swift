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

            // M200 (iter 137): "Calibration / Sample count" Stepper was
            // removed. The Swift setting existed for 100+ iters with a
            // UI Stepper that user could interact with — but the jang
            // convert CLI does NOT accept `--samples`/`-n` (the flag is
            // on the separate `profile` subcommand for TurboSmelt routing
            // profile collection, not convert). CLIArgsBuilder.args(for:)
            // never emitted a calibration-sample-count argument. Changing
            // the Stepper from 64 to 1024 had zero measurable effect.
            // That's a "don't lie to the user" violation
            // (feedback_dont_lie_to_user.md). If a future quant method
            // (AWQ, GPTQ with calibration data) is added, reintroduce
            // this Section AFTER: (1) adding `--samples` to p_convert's
            // argparse in jang-tools __main__.py, (2) threading it into
            // calibrate.py, (3) plumbing plan.calibrationSamples through
            // CLIArgsBuilder.args(for:). Prove end-to-end first; only
            // then add the Settings UI.

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

            Section("Publishing") {
                TextField("Default HuggingFace org", text: $settings.defaultHFOrg)
                    .textFieldStyle(.roundedBorder)
                    .disableAutocorrection(true)
                Text("Pre-fills the Publish sheet's repo field as {org}/{model-name}. Leave empty if you publish to multiple orgs.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
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
                // M62 (iter 108): logVerbosity is persisted but no emit
                // site currently consults it — enabling "Debug"/"Trace"
                // doesn't change log output. Signal the gap to the user.
                // When a JANG_LOG_LEVEL refactor lands across all stderr
                // emit sites, remove this label.
                Label("Not yet implemented — setting is preserved for when JANG_LOG_LEVEL lands.",
                      systemImage: "info.circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
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
                // M62 (iter 108): preAllocateRam + preAllocateRamGb persist
                // but there's no MLX buffer-pool env var to pipe them to
                // today. Signal the gap so a user enabling the toggle
                // doesn't expect it to actually reserve memory. Remove
                // this label when MLX exposes a pre-alloc knob.
                Label("Not yet implemented — awaits an MLX buffer-pool API to pipe the value into.",
                      systemImage: "info.circle")
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
        // M109 (iter 36): replace `.first!` force-unwrap with a safe fallback.
        // `.libraryDirectory` is essentially always present on macOS but a
        // sandboxed/MDM-restricted environment could return an empty array;
        // force-unwrap crashed the whole app. Fall back to NSHomeDirectory()
        // + Library so the user still gets SOMETHING to open.
        let dir: URL
        if settings.logFileOutputDir.isEmpty {
            let libraryRoot = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library")
            dir = libraryRoot.appendingPathComponent("Logs/JANGStudio")
        } else {
            dir = URL(fileURLWithPath: settings.logFileOutputDir)
        }
        // M157 (iter 80): surface createDirectory failures. Pre-fix, `try?`
        // silently swallowed permission-denied / read-only-volume / disk-full
        // errors — NSWorkspace.open against the nonexistent dir then silently
        // no-oped. User clicked "Open logs directory" → nothing happened,
        // zero feedback. Post-fix: log the failure to stderr (picked up by
        // Copy Diagnostics via iter-14 M22 pipeline) AND fall back to
        // opening the parent directory so the user gets SOMEWHERE useful.
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            NSWorkspace.shared.open(dir)
        } catch {
            FileHandle.standardError.write(
                Data("[SettingsWindow] could not create \(dir.path): \(error)\n".utf8))
            // Fall back to opening the parent, which should exist even if we
            // can't create the leaf.
            NSWorkspace.shared.open(dir.deletingLastPathComponent())
        }
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
            Section("Update channel") {
                Picker("Channel", selection: $settings.updateChannel) {
                    ForEach(UpdateChannel.allCases) { c in
                        Text(c.displayName).tag(c)
                    }
                }
                Toggle("Automatically check for updates", isOn: $settings.autoCheckForUpdates)
                Text("JANG Studio v1.0 ships with manual updates. Automatic updates via Sparkle are planned for v1.1.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                // M176b (iter 110): per-affordance "Not yet implemented"
                // marker to match iter-108 M62's pattern. The v1.0/v1.1
                // caption above explains the section; this label makes
                // the toggle's status visible at its own attention site.
                // Persisted value stays for when Sparkle lands in v1.1.
                Label("Not yet implemented — awaits Sparkle integration in v1.1.",
                      systemImage: "info.circle")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Section {
                Button {
                    if let url = URL(string: "https://github.com/jjang-ai/jangq/releases") {
                        NSWorkspace.shared.open(url)
                    }
                } label: {
                    Label("Check for updates (browser)", systemImage: "arrow.down.circle")
                }
            }
            Section("About") {
                LabeledContent("Version", value: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "—")
                LabeledContent("Build", value: Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "—")
                HStack {
                    Spacer()
                    Button("View release notes") {
                        if let url = URL(string: "https://github.com/jjang-ai/jangq/releases/latest") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                }
            }
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Auto-persist helper

/// Drives persistence whenever any observed property of `settings` changes.
/// Uses `withObservationTracking` in a loop — reads every property once to
/// register tracking, then fires `persist()` on the main actor on each change.
///
/// M64 (iter 98): verified the coalescing behavior is correct-by-design.
/// `withObservationTracking.onChange` is a ONE-SHOT callback — it fires
/// exactly once on the first mutation of any tracked property, after which
/// tracking is consumed. If multiple properties mutate in the same
/// synchronous `@MainActor` pass (e.g., Reset button setting `foo` then
/// `bar` in two successive lines), `onChange` fires only once — BUT
/// `persist()` runs AFTER both mutations have landed (Task hops to a new
/// main-actor execution), so the persisted Snapshot captures both. The
/// loop's next iteration re-registers tracking for the next batch.
///
/// Net behavior: ONE `persist()` call per batch of mutations that land in
/// the same main-actor transaction. Coalescing is a feature (fewer disk
/// writes, atomic multi-field updates) not a bug (no lost data — every
/// mutation is captured by the pending persist). The test pin below
/// guards against a future refactor that accidentally re-enters the loop
/// mid-batch and loses the coalescing.
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
                // M200 (iter 137): defaultCalibrationSamples removed —
                // field had no downstream consumer, was a Settings lie.
                _ = settings.outputNamingTemplate
                _ = settings.autoDeletePartialOnCancel
                _ = settings.revealInFinderOnFinish
                _ = settings.defaultHFOrg
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
