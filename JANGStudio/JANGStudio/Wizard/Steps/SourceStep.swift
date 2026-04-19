// JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift
import SwiftUI

struct SourceStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var isDetecting = false
    @State private var errorText: String?

    var body: some View {
        Form {
            Section("Source model folder") {
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
            }
            if let detected = coord.plan.detected {
                Section("Detected") {
                    LabeledContent("Model type", value: detected.modelType)
                    LabeledContent("Parameters", value: detected.isMoE ? "MoE · \(detected.numExperts) experts" : "Dense")
                    LabeledContent("Source dtype", value: detected.dtype.rawValue.uppercased())
                    LabeledContent("Disk", value: "\(detected.totalBytes / 1_000_000_000) GB (\(detected.shardCount) shards)")
                    if detected.isVL { Label("Vision/Language model", systemImage: "eye") }
                }
            }
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            if isDetecting {
                ProgressView().controlSize(.small)
            }
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
            errorText = nil
            Task { await detect(url: url) }
        }
    }

    private func detect(url: URL) async {
        isDetecting = true
        defer { isDetecting = false }
        do {
            let detected = try await SourceDetector.inspect(url: url)
            await MainActor.run { coord.plan.detected = detected }
        } catch {
            await MainActor.run { errorText = "Detection failed: \(error.localizedDescription)" }
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
