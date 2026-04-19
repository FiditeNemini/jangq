// JANGStudio/JANGStudio/Wizard/Steps/ProfileStep.swift
import SwiftUI

private let JANG_PROFILES = [
    "JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L",
    "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L",
    "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L",
    "JANG_5K", "JANG_6K", "JANG_6M",
]
private let JANGTQ_PROFILES = ["JANGTQ2", "JANGTQ3", "JANGTQ4"]

struct ProfileStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var preflight: [PreflightCheck] = []

    var body: some View {
        Form {
            Section("Family") {
                Picker("", selection: $coord.plan.family) {
                    Text("JANG").tag(Family.jang)
                    Text("JANGTQ").tag(Family.jangtq).disabled(!coord.plan.isJANGTQAllowed)
                }.pickerStyle(.segmented)
                if !coord.plan.isJANGTQAllowed {
                    Label("JANGTQ supports Qwen 3.6 and MiniMax only (v1). GLM coming in v1.1.",
                          systemImage: "info.circle").font(.caption)
                }
            }
            Section("Profile") {
                Picker("", selection: $coord.plan.profile) {
                    ForEach(coord.plan.family == .jang ? JANG_PROFILES : JANGTQ_PROFILES, id: \.self) { p in
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

    private func refresh() { preflight = PreflightRunner().run(plan: coord.plan) }
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
