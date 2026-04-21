// JANGStudio/JANGStudio/Wizard/Steps/ArchitectureStep.swift
import SwiftUI

struct ArchitectureStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(CapabilitiesService.self) private var capsSvc
    @State private var showOverrides = false

    var body: some View {
        Form {
            if let d = coord.plan.detected {
                Section {
                    // M218 (iter 146): always-visible cold-start caption.
                    // Pre-M218 a stranger who reached Step 2 saw only
                    // technical field names (model type, MoE, dtype) with
                    // no explanation of what this step is for. Now a
                    // subtitle line names the purpose + the typical
                    // stranger action ("these usually need no change").
                    Text("JANG Studio inspected your model and detected the properties below. Review them — usually no change is needed. If you want to override a specific property (e.g. force bfloat16), expand Advanced overrides.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 4)
                    LabeledContent("Model type", value: d.modelType)
                    VStack(alignment: .leading, spacing: 2) {
                        LabeledContent("Layout", value: d.isMoE ? "MoE · \(d.numExperts) experts" : "Dense")
                        // M218 (iter 146): plain-English caption beneath
                        // the technical value. Strangers know "Dense"
                        // sounds like "thick" not "non-sparse"; and MoE
                        // is jargon. One-line explanation defuses both.
                        if d.isMoE {
                            Text("MoE = Mixture-of-Experts: \(d.numExperts) specialized sub-networks; only a few activate per token. Common in large recent models (Qwen3-MoE, MiniMax, DeepSeek-V3).")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        } else {
                            Text("Dense = all weights active for every token. Standard transformer layout.")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        LabeledContent("Source dtype", value: d.dtype.rawValue.uppercased())
                        // M218: explain what dtype shows up as, briefly.
                        Text(dtypeHint(d.dtype))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    LabeledContent("Vision/Language", value: d.isVL ? "Yes" : "No")
                    if d.numExperts >= 256 {
                        Label("Large expert count — bfloat16 auto-forced to avoid float16 overflow.",
                              systemImage: "info.circle")
                    }
                } header: {
                    Text("Detected architecture")
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

    /// M218 (iter 146): plain-English hint for the detected source
    /// dtype. Each string is one short sentence a stranger can
    /// understand without a numerics background. Keeps captions
    /// stable across layouts (no wrapping surprise).
    private func dtypeHint(_ d: SourceDtype) -> String {
        switch d {
        case .bf16:
            return "BF16 = brain-float 16-bit; dominant modern LLM weight format. Convert supports it natively."
        case .fp16:
            return "FP16 = IEEE half-float; older LLM format. Convert handles it fine but some 512+ expert models are auto-upgraded to BF16 to avoid overflow."
        case .fp8:
            return "FP8 = 8-bit float (E4M3 or E5M2); emerging format used by DeepSeek V3/V3.2. JANGTQ family supports it; JANG family dequantizes to BF16 first."
        case .jangV2:
            return "Already a converted JANG v2 model — you're about to re-quantize. Uncommon; usually avoid."
        case .unknown:
            return "Dtype could not be detected. If convert fails, expand Advanced overrides and set Force dtype to match your source."
        }
    }
}
