// JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift
import SwiftUI

struct SourceStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var isDetecting = false
    @State private var isRecommending = false
    @State private var errorText: String?
    @State private var recommendation: Recommendation?

    var body: some View {
        Form {
            // MARK: - Folder picker
            Section {
                HStack {
                    if let url = coord.plan.sourceURL {
                        Text(url.lastPathComponent).font(.headline)
                        Text(url.deletingLastPathComponent().path)
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("No folder selected").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Choose Folder…", action: pickFolder)
                }
            } header: {
                HStack(spacing: 4) {
                    Text("Source model folder")
                    InfoHint("Pick a HuggingFace model directory — one containing `config.json` and `.safetensors` shards. JANG Studio auto-detects the architecture and recommends a conversion plan.")
                }
            }

            // MARK: - Detected
            if let detected = coord.plan.detected {
                Section("Detected") {
                    LabeledContent("Model type", value: detected.modelType)
                    LabeledContent(
                        "Parameters",
                        value: detected.isMoE ? "MoE · \(detected.numExperts) experts" : "Dense"
                    )
                    LabeledContent("Source dtype", value: detected.dtype.rawValue.uppercased())
                    LabeledContent(
                        "Disk",
                        value: "\(detected.totalBytes / 1_000_000_000) GB (\(detected.shardCount) shards)"
                    )
                    if detected.isVideoVL {
                        Label("Vision + Video", systemImage: "film")
                            .foregroundStyle(.purple)
                    } else if detected.isVL {
                        Label("Vision (image)", systemImage: "eye")
                            .foregroundStyle(.blue)
                    }
                }
            }

            // MARK: - Recommendation (beginner-friendly)
            if let rec = recommendation {
                Section {
                    Text(rec.beginnerSummary)
                        .font(.callout)
                        .padding(.vertical, 4)

                    Divider()

                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow {
                            Text("Family").foregroundStyle(.secondary)
                            Text(rec.recommended.family.uppercased())
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.family)
                        }
                        GridRow {
                            Text("Profile").foregroundStyle(.secondary)
                            Text(rec.recommended.profile)
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.profile)
                        }
                        GridRow {
                            Text("Method").foregroundStyle(.secondary)
                            Text(rec.recommended.method.uppercased())
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.method)
                        }
                        GridRow {
                            Text("Hadamard").foregroundStyle(.secondary)
                            Text(rec.recommended.hadamard ? "On" : "Off")
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.hadamard)
                        }
                        if let forceDtype = rec.recommended.forceDtype {
                            GridRow {
                                Text("Force dtype").foregroundStyle(.secondary)
                                Text(forceDtype)
                                    .fontWeight(.medium)
                                InfoHint(rec.whyEachChoice.forceDtype)
                            }
                        }
                    }

                    ForEach(rec.warnings, id: \.self) { warning in
                        Label(warning, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .font(.caption)
                            .padding(.top, 2)
                    }

                    if !rec.recommended.alternatives.isEmpty {
                        DisclosureGroup("Other options") {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(rec.recommended.alternatives) { alt in
                                    VStack(alignment: .leading, spacing: 2) {
                                        HStack(spacing: 4) {
                                            if let fam = alt.family {
                                                Text(fam.uppercased())
                                                    .font(.caption2)
                                                    .padding(.horizontal, 5)
                                                    .padding(.vertical, 2)
                                                    .background(Color.secondary.opacity(0.2))
                                                    .cornerRadius(4)
                                            }
                                            Text(alt.profile).fontWeight(.medium)
                                        }
                                        Text(alt.useWhen)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                } header: {
                    HStack(spacing: 4) {
                        Text("Recommended for this model")
                        InfoHint("Smart defaults auto-filled based on what JANG Studio detected. Most beginners can skip Steps 2 and 3 and hit Start in Step 4.")
                    }
                }
            }

            // MARK: - Detection status
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            if isDetecting || isRecommending {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(isDetecting ? "Inspecting source…" : "Building recommendation…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // MARK: - Continue
            if coord.plan.isStep1Complete {
                Button("Continue →") { coord.active = .architecture }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            coord.plan.sourceURL = url
            coord.plan.detected = nil
            recommendation = nil
            errorText = nil
            Task { await detectAndRecommend(url: url) }
        }
    }

    private func detectAndRecommend(url: URL) async {
        // Step A: fast inspect-source call
        isDetecting = true
        do {
            let detected = try await SourceDetector.inspect(url: url)
            await MainActor.run { coord.plan.detected = detected }
        } catch {
            await MainActor.run {
                errorText = "Detection failed: \(error.localizedDescription)"
                isDetecting = false
            }
            return
        }
        await MainActor.run { isDetecting = false }

        // Step B: recommendation call (also fast, reads same config.json)
        isRecommending = true
        defer { Task { @MainActor in isRecommending = false } }
        do {
            let rec = try await RecommendationService.fetch(modelURL: url)
            await MainActor.run {
                self.recommendation = rec
                self.applyRecommendation(rec)
            }
        } catch {
            await MainActor.run {
                // Recommendation failure is soft — user can still convert manually.
                errorText = "Recommendation failed (conversion still works): \(error.localizedDescription)"
            }
        }
    }

    /// Fill in ConversionPlan defaults based on the recommendation. Only touches
    /// fields the user hasn't manually changed (heuristic: fields still at their
    /// initial defaults get replaced; anything else is preserved).
    private func applyRecommendation(_ rec: Recommendation) {
        let plan = coord.plan

        // Family — replace unconditionally since user hasn't visited Step 3 yet
        plan.family = (rec.recommended.family == "jangtq") ? .jangtq : .jang

        // Profile — replace if still at the app-level default (JANG_4K)
        if plan.profile == "JANG_4K" {
            plan.profile = rec.recommended.profile
        }

        // Method
        let recMethod: QuantMethod = switch rec.recommended.method {
        case "rtn": .rtn
        case "mse-all": .mseAll
        default: .mse
        }
        plan.method = recMethod

        // Hadamard
        plan.hadamard = rec.recommended.hadamard

        // Force dtype — only set if recommendation has one
        if let forceDtypeStr = rec.recommended.forceDtype {
            let forceDtype: SourceDtype? = switch forceDtypeStr {
            case "bfloat16": .bf16
            case "float16": .fp16
            case "float8_e4m3fn", "float8_e5m2": .fp8
            default: nil
            }
            if let ft = forceDtype {
                plan.overrides.forceDtype = ft
            }
        }

        // Block size
        if plan.overrides.forceBlockSize == nil {
            plan.overrides.forceBlockSize = rec.recommended.blockSize
        }
    }
}

enum SourceDetector {
    struct SourceInfo: Decodable {
        let model_type: String
        let is_moe: Bool
        let num_experts: Int
        let dtype: String
        let total_bytes: Int64
        let shard_count: Int
        let is_vl: Bool
        let is_video_vl: Bool
        let has_generation_config: Bool
        let jangtq_compatible: Bool
    }

    static func inspect(url: URL) async throws -> ArchitectureSummary {
        let proc = Process()
        proc.executableURL = BundleResolver.pythonExecutable
        proc.arguments = ["-m", "jang_tools", "inspect-source", "--json", url.path]
        let out = Pipe(); proc.standardOutput = out; proc.standardError = Pipe()
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else {
            throw NSError(domain: "SourceDetector", code: Int(proc.terminationStatus),
                          userInfo: [NSLocalizedDescriptionKey: "inspect-source exited \(proc.terminationStatus)"])
        }
        let data = out.fileHandleForReading.readDataToEndOfFile()
        let info = try JSONDecoder().decode(SourceInfo.self, from: data)
        let dtype: SourceDtype = switch info.dtype {
            case "bfloat16": .bf16
            case "float16": .fp16
            case "float8_e4m3fn", "float8_e5m2": .fp8
            default: .unknown
        }
        return .init(modelType: info.model_type, isMoE: info.is_moe, numExperts: info.num_experts,
                     isVL: info.is_vl, isVideoVL: info.is_video_vl,
                     hasGenerationConfig: info.has_generation_config,
                     dtype: dtype, totalBytes: info.total_bytes, shardCount: info.shard_count)
    }
}
