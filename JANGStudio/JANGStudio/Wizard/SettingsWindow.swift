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
            Text("Advanced settings — see commit 3")
                .foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
        .padding()
    }
}

// MARK: - Performance tab

private struct PerformanceTab: View {
    @Bindable var settings: AppSettings
    var body: some View {
        Form {
            Text("Performance settings — see commit 4")
                .foregroundStyle(.secondary)
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
            Text("Diagnostics settings — see commit 4")
                .foregroundStyle(.secondary)
        }
        .formStyle(.grouped)
        .padding()
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
