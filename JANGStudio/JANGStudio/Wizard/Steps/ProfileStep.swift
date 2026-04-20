// JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift
import SwiftUI

struct ProfileStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(ProfilesService.self) private var profilesSvc
    @Environment(CapabilitiesService.self) private var capsSvc
    @State private var preflight: [PreflightCheck] = []

    private var jangProfileNames: [String] {
        profilesSvc.profiles.jang.map { $0.name }
    }
    private var jangtqProfileNames: [String] {
        profilesSvc.profiles.jangtq.map { $0.name }
    }
    private var isJANGTQAllowed: Bool {
        coord.plan.isJANGTQAllowed(for: capsSvc.capabilities.jangtqWhitelist)
    }

    var body: some View {
        Form {
            Section("Family") {
                Picker("", selection: $coord.plan.family) {
                    Text("JANG").tag(Family.jang)
                    Text("JANGTQ").tag(Family.jangtq).disabled(!isJANGTQAllowed)
                }.pickerStyle(.segmented)
                if !isJANGTQAllowed {
                    Label("JANGTQ supports \(capsSvc.capabilities.jangtqWhitelist.joined(separator: ", ")) only.",
                          systemImage: "info.circle").font(.caption)
                }
            }
            Section("Profile") {
                Picker("", selection: $coord.plan.profile) {
                    ForEach(coord.plan.family == .jang ? jangProfileNames : jangtqProfileNames, id: \.self) { p in
                        Text(p).tag(p)
                    }
                }.pickerStyle(.menu)
            }
            Section("Output folder") {
                HStack {
                    Text(coord.plan.outputURL?.path ?? "—").foregroundStyle(.secondary)
                    Spacer()
                    Button("Choose…", action: pickOutput)
                }
            }
            Section("Options") {
                Picker("Method", selection: $coord.plan.method) {
                    Text("MSE").tag(QuantMethod.mse)
                    Text("RTN").tag(QuantMethod.rtn)
                    Text("MSE (all)").tag(QuantMethod.mseAll)
                }.pickerStyle(.segmented)
                Toggle("Hadamard rotation", isOn: $coord.plan.hadamard)
            }
            Section("Pre-flight") {
                ForEach(preflight) { check in
                    HStack {
                        Image(systemName: icon(check.status))
                            .foregroundStyle(color(check.status))
                        Text(check.title)
                        if let hint = check.hint {
                            Text(hint).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            Button("Start Conversion") { coord.active = .run }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!allMandatoryPass())
        }
        .formStyle(.grouped)
        .padding()
        .onChange(of: coord.plan.profile) { _, _ in refresh() }
        .onChange(of: coord.plan.family) { _, _ in refresh() }
        .onChange(of: coord.plan.outputURL) { _, _ in refresh() }
        .onChange(of: coord.plan.hadamard) { _, _ in refresh() }
        .onAppear {
            if coord.plan.outputURL == nil, let src = coord.plan.sourceURL {
                coord.plan.outputURL = src.deletingLastPathComponent().appendingPathComponent("\(src.lastPathComponent)-\(coord.plan.profile)")
            }
            refresh()
        }
    }

    private func refresh() {
        // M141 (iter 63): pass profiles so the disk-space check can do a
        // profile-aware size estimate instead of short-circuiting to .pass.
        preflight = PreflightRunner().run(
            plan: coord.plan,
            capabilities: capsSvc.capabilities,
            profiles: profilesSvc.profiles
        )
    }
    private func allMandatoryPass() -> Bool { !preflight.contains { $0.status == .fail } }

    private func pickOutput() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true; panel.canChooseFiles = false
        panel.allowsMultipleSelection = false; panel.canCreateDirectories = true
        panel.prompt = "Choose"
        if panel.runModal() == .OK { coord.plan.outputURL = panel.url }
    }

    private func icon(_ s: PreflightStatus) -> String {
        switch s { case .pass: "checkmark.circle.fill"; case .warn: "exclamationmark.triangle.fill"; case .fail: "xmark.circle.fill" }
    }
    private func color(_ s: PreflightStatus) -> Color {
        switch s { case .pass: .green; case .warn: .yellow; case .fail: .red }
    }
}
