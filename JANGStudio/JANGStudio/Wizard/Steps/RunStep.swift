// JANGStudio/JANGStudio/Wizard/Steps/RunStep.swift
import SwiftUI

struct RunStep: View {
    @Bindable var coord: WizardCoordinator
    @Environment(AppSettings.self) private var settings
    @State private var phase: (n: Int, total: Int, name: String) = (0, 5, "idle")
    @State private var tick: (done: Int, total: Int, label: String)? = nil
    @State private var logs: [String] = []
    @State private var runner: PythonRunner?
    @State private var startedAt: Date?
    @State private var cancelRequested: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Phase \(phase.n)/\(phase.total) · \(phase.name)").font(.headline)
                Spacer()
                if coord.plan.run == .running {
                    Button("Cancel", role: .destructive) {
                        cancelRequested = true
                        Task { await runner?.cancel() }
                    }
                    .disabled(cancelRequested)
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
            } else if coord.plan.run == .cancelled {
                Label("Cancelled — partial output left on disk at output folder.", systemImage: "stop.circle.fill")
                    .foregroundStyle(.orange)
                HStack {
                    Button("Retry") { cancelRequested = false; Task { await start() } }
                        .buttonStyle(.borderedProminent).keyboardShortcut(.defaultAction)
                    Button("Delete partial output", role: .destructive) {
                        if let out = coord.plan.outputURL {
                            // M107 (iter 35): surface delete failures via the
                            // existing log pane so the user doesn't assume the
                            // delete succeeded when it didn't (permission
                            // denied, file in use by another process, already
                            // gone from disk). Pre-iter-35 the `try?` silently
                            // swallowed every error; user walked away thinking
                            // the output was cleaned up when it wasn't.
                            do {
                                try FileManager.default.removeItem(at: out)
                                logs.append("[cleanup] deleted \(out.path)")
                            } catch {
                                logs.append("[cleanup] delete FAILED: \(error.localizedDescription)")
                            }
                        }
                    }
                    .disabled(coord.plan.outputURL == nil)
                }
            } else if coord.plan.run == .failed {
                Label("Conversion failed — see log", systemImage: "xmark.octagon.fill").foregroundStyle(.red)
                Button("Retry") { cancelRequested = false; Task { await start() } }
                Button("Copy Diagnostics") {
                    // M109 (iter 36): `.first!` would crash the app in sandboxed /
                    // MDM environments where `.desktopDirectory` isn't available.
                    // Fall back to the home directory so Copy Diagnostics always
                    // works — the user can move the zip afterward.
                    let desktop = FileManager.default.urls(for: .desktopDirectory, in: .userDomainMask).first
                        ?? URL(fileURLWithPath: NSHomeDirectory())
                    let events = logs.filter { $0.hasPrefix("{") }
                    // M62-anonymize: honor Settings → Diagnostics →
                    // "Anonymize paths in diagnostics". Otherwise a bug report
                    // zip leaks the user's filesystem layout.
                    // M107 (iter 35): surface write failure via the existing log
                    // pane instead of silently dismissing.
                    // M106 (iter 42): switched to writeAsync so the ditto
                    // subprocess runs off MainActor. Pre-fix, a large diag
                    // bundle (50+ MB of tick events + stderr) could beach-ball
                    // the UI for several seconds during zip creation.
                    Task {
                        do {
                            let url = try await DiagnosticsBundle.writeAsync(
                                plan: coord.plan, logLines: logs, eventLines: events,
                                verify: [], to: desktop,
                                anonymizePaths: settings.anonymizePathsInDiagnostics)
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        } catch {
                            logs.append("[diagnostics] FAILED to write zip: \(error.localizedDescription)")
                        }
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
        cancelRequested = false
        logs.removeAll()
        startedAt = Date()
        let args = buildArgs()
        let r = PythonRunner(extraArgs: args)
        runner = r
        do {
            for try await ev in r.run() {
                await MainActor.run { apply(ev) }
            }
            // Stream finished without throwing — distinguish cancel vs natural success.
            await MainActor.run {
                coord.plan.run = cancelRequested ? .cancelled : .succeeded
                if cancelRequested {
                    logs.append("[cancelled] SIGTERM acknowledged, process exited")
                    // M62: honor Settings → General → Behavior →
                    // "Auto-delete partial output on cancel". Previously inert.
                    if settings.autoDeletePartialOnCancel, let out = coord.plan.outputURL {
                        do {
                            try FileManager.default.removeItem(at: out)
                            logs.append("[cancelled] deleted partial output at \(out.path) (auto-delete setting on)")
                        } catch {
                            logs.append("[cancelled] auto-delete failed: \(error.localizedDescription)")
                        }
                    }
                }
            }
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

    private func buildArgs() -> [String] { CLIArgsBuilder.args(for: coord.plan) }
}
