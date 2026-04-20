// JANGStudio/JANGStudio/Wizard/Steps/VerifyStep.swift
import SwiftUI

struct VerifyStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(CapabilitiesService.self) private var capsSvc
    @State private var checks: [VerifyCheck] = []
    @State private var busy = true
    @State private var showingInference = false
    @State private var showingExamples = false
    @State private var showingModelCard = false
    @State private var showingPublish = false

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

                    // Adoption actions row
                    HStack {
                        Button {
                            showingInference = true
                        } label: {
                            Label("Test Inference", systemImage: "bubble.left.and.bubble.right")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(coord.plan.outputURL == nil)

                        Button {
                            showingExamples = true
                        } label: {
                            Label("View Usage Examples", systemImage: "doc.text")
                        }
                        .disabled(coord.plan.outputURL == nil)

                        Button {
                            showingModelCard = true
                        } label: {
                            Label("Generate Model Card", systemImage: "doc.richtext")
                        }
                        .disabled(coord.plan.outputURL == nil)

                        Button {
                            showingPublish = true
                        } label: {
                            Label("Publish to HF", systemImage: "arrow.up.doc.on.clipboard")
                        }
                        .disabled(coord.plan.outputURL == nil)
                    }

                    HStack {
                        Button {
                            revealOutput()
                        } label: {
                            Label("Reveal in Finder", systemImage: "folder")
                        }
                        Button {
                            copyPath()
                        } label: {
                            Label("Copy Path", systemImage: "doc.on.doc")
                        }
                    }

                    HStack {
                        Button("Convert another") { reset() }
                        Spacer()
                        Button("Finish") { finishApp() }
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
        .sheet(isPresented: $showingInference) {
            if let url = coord.plan.outputURL {
                TestInferenceSheet(
                    modelPath: url,
                    isVL: coord.plan.detected?.isVL ?? false,
                    isVideoVL: coord.plan.detected?.isVideoVL ?? false,
                    modelType: coord.plan.detected?.modelType ?? "unknown",
                    profile: coord.plan.profile,
                    sizeGb: Double(coord.plan.detected?.totalBytes ?? 0) / 1_000_000_000.0
                )
            }
        }
        .sheet(isPresented: $showingExamples) {
            if let url = coord.plan.outputURL {
                UsageExamplesSheet(modelPath: url)
            }
        }
        .sheet(isPresented: $showingModelCard) {
            if let url = coord.plan.outputURL {
                GenerateModelCardSheet(modelPath: url)
            }
        }
        .sheet(isPresented: $showingPublish) {
            if let url = coord.plan.outputURL {
                PublishToHuggingFaceSheet(modelPath: url)
            }
        }
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
