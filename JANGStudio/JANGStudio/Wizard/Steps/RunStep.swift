// JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift
import SwiftUI

struct RunStep: View {
    @Bindable var coord: WizardCoordinator
    @State private var phase: (n: Int, total: Int, name: String) = (0, 5, "idle")
    @State private var tick: (done: Int, total: Int, label: String)? = nil
    @State private var logs: [String] = []
    @State private var runner: PythonRunner?
    @State private var startedAt: Date?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Phase \(phase.n)/\(phase.total) · \(phase.name)").font(.headline)
                Spacer()
                if coord.plan.run == .running {
                    Button("Cancel", role: .destructive) { Task { await runner?.cancel() } }
                }
            }
            ProgressView(value: Double(phase.n), total: Double(phase.total))
            if let t = tick {
                ProgressView(value: Double(t.done), total: Double(t.total)) {
                    Text(t.label).font(.caption).lineLimit(1).truncationMode(.middle)
                }
            }
            ScrollView {
                Text(logs.suffix(500).joined(separator: "\n"))
                    .font(.system(.caption, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(minHeight: 240)
            .background(Color(.textBackgroundColor))
            .border(.separator)
            if coord.plan.run == .succeeded {
                Button("Continue → Verify") { coord.active = .verify }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
            } else if coord.plan.run == .failed {
                Label("Conversion failed — see log", systemImage: "xmark.octagon.fill").foregroundStyle(.red)
                Button("Retry") { Task { await start() } }
                Button("Copy Diagnostics") {
                    let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first!
                    let events = logs.filter { $0.hasPrefix("{") }
                    if let url = try? DiagnosticsBundle.write(plan: coord.plan, logLines: logs, eventLines: events,
                                                              verify: [], to: desktop) {
                        NSWorkspace.shared.activateFileViewerSelecting([url])
                    }
                }
            }
        }
        .padding()
        .onAppear { Task { await start() } }
    }

    private func start() async {
        guard coord.plan.run != .running else { return }
        coord.plan.run = .running
        logs.removeAll()
        startedAt = Date()
        let args = buildArgs()
        let r = PythonRunner(extraArgs: args)
        runner = r
        do {
            for try await ev in r.run() {
                await MainActor.run { apply(ev) }
            }
            await MainActor.run { coord.plan.run = .succeeded }
        } catch {
            await MainActor.run {
                coord.plan.run = .failed
                logs.append("[ERROR] \(error)")
            }
        }
    }

    private func apply(_ ev: ProgressEvent) {
        switch ev.payload {
        case .phase(let n, let total, let name):
            phase = (n, total, name); tick = nil
            logs.append("[\(n)/\(total)] \(name)")
        case .tick(let done, let total, let label):
            tick = (done, total, label ?? "")
        case .message(let level, let text):
            logs.append("[\(level)] \(text)")
        case .done(let ok, _, let err):
            if !ok, let err { logs.append("[done] error=\(err)") }
        case .versionMismatch(let v): logs.append("[error] protocol version \(v) unsupported")
        case .parseError(let s): logs.append("[parse-err] \(s)")
        }
    }

    private func buildArgs() -> [String] {
        let plan = coord.plan
        guard let src = plan.sourceURL?.path, let out = plan.outputURL?.path else { return [] }
        switch plan.family {
        case .jang:
            var args = ["-m", "jang_tools", "convert", src, "-o", out, "-p", plan.profile,
                        "-m", plan.method.rawValue, "--progress=json", "--quiet-text"]
            if plan.hadamard { args.append("--hadamard") }
            return args
        case .jangtq:
            let mod: String = switch plan.detected?.modelType ?? "" {
                case "qwen3_5_moe": "jang_tools.convert_qwen35_jangtq"
                case "minimax_m2":  "jang_tools.convert_minimax_jangtq"
                default: "jang_tools.convert_qwen35_jangtq"
            }
            return ["-m", mod, "--progress=json", "--quiet-text", src, out, plan.profile]
        }
    }
}
