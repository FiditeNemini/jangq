// JANGStudio/JANGStudio/Wizard/Steps/VerifyStep.swift
import SwiftUI

struct VerifyStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(CapabilitiesService.self) private var capsSvc
    @State private var checks: [VerifyCheck] = []
    @State private var busy = true

    var body: some View {
        Form {
            Section("Output verification") {
                if busy { ProgressView() }
                ForEach(checks) { c in
                    HStack {
                        Image(systemName: icon(c.status)).foregroundStyle(color(c.status))
                        Text(c.title)
                        if !c.required { Text("(warn)").font(.caption).foregroundStyle(.secondary) }
                        if let h = c.hint { Text(h).font(.caption).foregroundStyle(.secondary) }
                    }
                }
            }
            if !busy, finishable() {
                Section {
                    if let url = coord.plan.outputURL {
                        LabeledContent("Ready at", value: url.path)
                    }
                    HStack {
                        Button("Reveal in Finder") { revealOutput() }
                        Button("Copy Path") { copyPath() }
                        Button("Convert another") { reset() }
                        Button("Finish") { finishApp() }.buttonStyle(.borderedProminent)
                    }
                }
            } else if !busy {
                Section {
                    Label("Output incomplete — cannot finish.", systemImage: "xmark.octagon.fill")
                        .foregroundStyle(.red)
                    Button("Retry conversion") { coord.active = .run }
                }
            }
        }
        .formStyle(.grouped).padding()
        .onAppear { Task { await refresh() } }
    }

    private func refresh() async {
        busy = true
        let c = await PostConvertVerifier().run(plan: coord.plan, capabilities: capsSvc.capabilities)
        await MainActor.run { checks = c; busy = false }
    }

    private func finishable() -> Bool { !checks.contains { $0.required && $0.status == .fail } }

    private func icon(_ s: PreflightStatus) -> String {
        switch s { case .pass: "checkmark.circle.fill"; case .warn: "exclamationmark.triangle.fill"; case .fail: "xmark.circle.fill" }
    }
    private func color(_ s: PreflightStatus) -> Color {
        switch s { case .pass: .green; case .warn: .yellow; case .fail: .red }
    }
    private func revealOutput() {
        if let url = coord.plan.outputURL { NSWorkspace.shared.activateFileViewerSelecting([url]) }
    }
    private func copyPath() {
        if let p = coord.plan.outputURL?.path {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(p, forType: .string)
        }
    }
    private func reset() {
        coord.plan = ConversionPlan()
        coord.active = .source
    }
    private func finishApp() { NSApp.windows.first?.close() }
}
