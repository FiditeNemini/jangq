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
    /// M138 (iter 60): authoritative success marker. Pre-iter-60, RunStep
    /// used `cancelRequested ? .cancelled : .succeeded` to decide outcome
    /// after the stream exited. But PythonRunner treats a cancelled AND a
    /// successful subprocess THE SAME WAY: `continuation.finish()` clean,
    /// no throw. So a user hitting Cancel at the same microsecond the
    /// conversion completed with exit 0 would get run=.cancelled even
    /// though the output is fully written. Worse: if the user had
    /// "Auto-delete partial output on cancel" enabled, the successful
    /// output folder was DELETED. Same race class as iter-59 M137
    /// (Publish), but with data-loss stakes.
    ///
    /// Track whether we received a final `.done(ok: true, …)` event —
    /// THAT is the authoritative "conversion completed successfully"
    /// signal. A cancel that preempted the final write won't emit
    /// ok=true, so `sawSuccessfulDone` stays false and we correctly
    /// report .cancelled.
    @State private var sawSuccessfulDone: Bool = false

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
        // M136 (iter 58): only auto-start on first entry (.idle). Without
        // the run-state check, SwiftUI's .onAppear fires every time the view
        // reappears — e.g., when the user nav-backs from VerifyStep via the
        // sidebar to inspect logs. `start()`'s only guard was
        // `run != .running`, so a completed / failed / cancelled run got
        // restarted on nav-back, wiping logs + overwriting the finished
        // output folder. Retry buttons below still call `start()` directly
        // (they rely on the weaker guard inside start()); only the
        // auto-start path needs the tighter gate.
        .onAppear {
            if coord.plan.run == .idle {
                Task { await start() }
            }
        }
    }

    private func start() async {
        guard coord.plan.run != .running else { return }
        coord.plan.run = .running
        cancelRequested = false
        sawSuccessfulDone = false   // M138: reset for the new run.
        logs.removeAll()
        startedAt = Date()
        let args = buildArgs()
        let r = PythonRunner(extraArgs: args)
        runner = r
        do {
            for try await ev in r.run() {
                await MainActor.run { apply(ev) }
            }
            // Stream finished without throwing. M138 (iter 60): use the
            // authoritative success signal (final `.done(ok: true, …)`
            // event) rather than the user's cancel-intent flag. A late
            // cancel click that landed AFTER the subprocess naturally
            // completed would otherwise set run=.cancelled and —
            // catastrophically, when autoDeletePartialOnCancel=true —
            // delete the successfully-written output folder.
            await MainActor.run {
                if sawSuccessfulDone {
                    coord.plan.run = .succeeded
                    if cancelRequested {
                        // Document the race outcome so the user who hit
                        // Cancel understands the subprocess beat them.
                        logs.append("[note] Cancel click landed after the final write — output is complete.")
                    }
                } else {
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
            if ok {
                // M138 (iter 60): authoritative success marker. Python emits
                // exactly one .done event at end-of-run; ok=true means the
                // subprocess completed its final write without error.
                sawSuccessfulDone = true
            } else if let err {
                logs.append("[done] error=\(err)")
            }
        case .versionMismatch(let v): logs.append("[error] protocol version \(v) unsupported")
        case .parseError(let s): logs.append("[parse-err] \(s)")
        }
    }

    private func buildArgs() -> [String] { CLIArgsBuilder.args(for: coord.plan) }
}
