// JANGStudio/JANGStudio/Wizard/Steps/ArchitectureStep.swift
import SwiftUI

struct ArchitectureStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(CapabilitiesService.self) private var capsSvc
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
                    ForEach(capsSvc.capabilities.supportedSourceDtypes, id: \.name) { d in
                        Text(d.alias.uppercased()).tag(dtypeFromAlias(d.alias))
                    }
                }
                Picker("Block size", selection: Binding(
                    get: { coord.plan.overrides.forceBlockSize ?? 0 },
                    set: { coord.plan.overrides.forceBlockSize = ($0 == 0) ? nil : $0 }
                )) {
                    Text("Auto").tag(0)
                    ForEach(capsSvc.capabilities.blockSizes, id: \.self) { bs in
                        Text("\(bs)").tag(bs)
                    }
                }
            }
            // M134 (iter 56): peer-helper parity with SourceStep (wraps in
            // `if isStep1Complete`) and ProfileStep (`.disabled(!allMandatoryPass())`).
            // Pre-iter-56 this button had NO gate, so a user who navigated
            // to Architecture while detection was still in-flight (or after
            // repicking a source that failed detection → detected=nil) could
            // click through to Profile with an inconsistent plan state.
            // Downstream preflight would eventually block them at
            // ProfileStep's Start Conversion gate, but that's a late, noisy
            // failure path. Gate here for immediate, consistent feedback.
            Button("Looks right → Profile") { coord.active = .profile }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
                .disabled(!coord.plan.isStep2Complete)
        }
        .formStyle(.grouped)
        .padding()
    }

    private func dtypeFromAlias(_ alias: String) -> SourceDtype {
        switch alias {
        case "bf16": return .bf16
        case "fp16": return .fp16
        case "fp8": return .fp8
        case "fp8-e5m2": return .fp8
        default: return .unknown
        }
    }
}
