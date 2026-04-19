// JANGStudio/JANGStudio/Wizard/Steps/ArchitectureStep.swift
import SwiftUI

struct ArchitectureStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var showOverrides = false

    var body: some View {
        Form {
            if let d = coord.plan.detected {
                Section("Detected architecture") {
                    LabeledContent("Model type", value: d.modelType)
                    LabeledContent("Layout", value: d.isMoE ? "MoE · \(d.numExperts) experts" : "Dense")
                    LabeledContent("Source dtype", value: d.dtype.rawValue.uppercased())
                    LabeledContent("Vision/Language", value: d.isVL ? "Yes" : "No")
                    if d.numExperts >= 256 {
                        Label("Large expert count — bfloat16 auto-forced to avoid float16 overflow.",
                              systemImage: "info.circle")
                    }
                }
            }
            DisclosureGroup("Advanced overrides", isExpanded: $showOverrides) {
                Picker("Force dtype", selection: Binding(
                    get: { coord.plan.overrides.forceDtype ?? .unknown },
                    set: { coord.plan.overrides.forceDtype = ($0 == .unknown) ? nil : $0 }
                )) {
                    Text("Auto").tag(SourceDtype.unknown)
                    Text("BF16").tag(SourceDtype.bf16)
                    Text("FP16").tag(SourceDtype.fp16)
                }
                Picker("Block size", selection: Binding(
                    get: { coord.plan.overrides.forceBlockSize ?? 0 },
                    set: { coord.plan.overrides.forceBlockSize = ($0 == 0) ? nil : $0 }
                )) {
                    Text("Auto").tag(0)
                    Text("32").tag(32)
                    Text("64").tag(64)
                    Text("128").tag(128)
                }
            }
            Button("Looks right → Profile") { coord.active = .profile }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
        }
        .formStyle(.grouped)
        .padding()
    }
}
