import SwiftUI

struct PublishToHuggingFaceSheet: View {
    let modelPath: URL
    @Environment(\.dismiss) private var dismiss
    @Environment(AppSettings.self) private var settings

    @State private var repoName: String = ""
    @State private var isPrivate: Bool = false
    @State private var token: String = ""
    @State private var isDryRunning: Bool = false
    @State private var isPublishing: Bool = false
    @State private var dryRunResult: PublishResult?
    @State private var publishResult: PublishResult?
    @State private var errorMessage: String?
    @State private var orgPrefixApplied: Bool = false   // M48: idempotent one-shot flag

    // M43 (iter 24): live progress from the streaming publish.
    // `progressPhase` = current 3-phase phase name (scan/upload/finalize).
    // `progressBytes` = (uploaded, total) for the UI bar.
    // `progressLabel` = latest per-file label (e.g. filename being uploaded).
    @State private var progressPhase: String = ""
    @State private var progressBytes: (done: Int64, total: Int64)? = nil
    @State private var progressLabel: String = ""
    @State private var progressLog: [String] = []
    // M96 (iter 30): handle to the running publish Task so a user-initiated
    // Cancel can tear down the subprocess via continuation.onTermination.
    @State private var publishTask: Task<Void, Never>? = nil
    @State private var wasCancelled: Bool = false
    // M171 (iter 94): handle for the Preview button's dry-run Task. Pre-M171
    // the button did `Task { await runDryRun() }` fire-and-forget, so a
    // user who clicked Preview then dismissed the sheet left a ~few-second
    // Python subprocess orphaned. iter-85 M162's .onDisappear only cancels
    // publishTask; this closes the matching gap for dryRun. Same class as
    // iter-86 M163 (retry-task consistency fix for read-only sheets).
    @State private var dryRunTask: Task<Void, Never>? = nil

    init(modelPath: URL, defaultRepoName: String = "") {
        self.modelPath = modelPath
        self._repoName = State(initialValue: defaultRepoName.isEmpty
                               ? modelPath.lastPathComponent
                               : defaultRepoName)
        self._token = State(initialValue: ProcessInfo.processInfo.environment["HF_HUB_TOKEN"]
                            ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"]
                            ?? "")
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            form
            Divider()
            footer
        }
        .frame(minWidth: 640, minHeight: 460)
        .task {
            // M48 (iter 25): if the user has configured a default HF org in
            // Settings, prefix it on the repoName so the field lands at
            // `org/my-model-JANG_4K` instead of the always-invalid
            // `my-model-JANG_4K`. Guard with `orgPrefixApplied` so clicking
            // into the field and back out doesn't re-apply after the user
            // has started editing.
            guard !orgPrefixApplied else { return }
            orgPrefixApplied = true
            applyOrgPrefixIfNeeded()
        }
        .onDisappear {
            // M162 (iter 85): dismissing the sheet mid-publish (user clicks
            // Close, cmd-W, or the in-header Close button) MUST tear down
            // the upload. Before this hook, the publishTask kept running
            // even after the sheet was gone — the Python subprocess
            // continued uploading files to HuggingFace for the remaining
            // ~30 minutes with no UI, no way to cancel, no visibility.
            // User who hit Close thinking they were cancelling would
            // unknowingly complete the upload to the wrong repo. Real
            // data-exfiltration vector: picks the wrong org/name, hits
            // Close to try again, and the files still ship.
            //
            // Cancelling the Task triggers onTermination on the
            // AsyncThrowingStream, which triggers ProcessHandle.cancel()
            // via the iter-30 M96 wiring — SIGTERM + 3 s SIGKILL
            // escalation. Partial-repo cleanup is still M97 (deferred).
            publishTask?.cancel()
            // M171 (iter 94): also cancel dry-run. Closes the sibling
            // orphan gap iter-85 M162 missed (Preview button spawned a
            // handle-less Task).
            dryRunTask?.cancel()
        }
    }

    /// Prefix the default HF org onto a basename-only repo field. No-ops when:
    /// - settings.defaultHFOrg is empty (user hasn't configured an org yet)
    /// - repoName already contains a `/` (user already entered an org)
    /// - repoName doesn't match the basename-only default
    private func applyOrgPrefixIfNeeded() {
        let org = settings.defaultHFOrg.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !org.isEmpty else { return }
        guard !repoName.contains("/") else { return }
        // Only replace the basename default — don't stomp user-typed text.
        guard repoName == modelPath.lastPathComponent else { return }
        repoName = "\(org)/\(modelPath.lastPathComponent)"
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Publish to HuggingFace")
                    .font(.headline)
                Text(modelPath.lastPathComponent)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { dismiss() }
        }
        .padding(12)
    }

    private var form: some View {
        Form {
            Section("Repository") {
                TextField("org/model-name", text: $repoName)
                    .textFieldStyle(.roundedBorder)
                    .disableAutocorrection(true)
                Toggle("Private repository", isOn: $isPrivate)
            }
            Section("Authentication") {
                SecureField("HuggingFace token", text: $token)
                    .textFieldStyle(.roundedBorder)
                if token.isEmpty {
                    Label("Get a token at huggingface.co/settings/tokens",
                          systemImage: "link")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if let r = dryRunResult {
                Section("Preview") {
                    LabeledContent("Files", value: "\(r.filesCount ?? 0)")
                    let sizeGb = Double(r.totalSizeBytes ?? 0) / 1_000_000_000
                    LabeledContent("Total size", value: String(format: "%.2f GB", sizeGb))
                    LabeledContent("Repo", value: r.repo)
                    LabeledContent("Private", value: isPrivate ? "Yes" : "No")
                }
            }
            if isPublishing {
                // M43 (iter 24): live upload progress. Shown only while the
                // streaming publish is running. Replaces the former
                // spinner-only UX for what used to be a 30-min silent window.
                Section("Uploading") {
                    if !progressPhase.isEmpty {
                        LabeledContent("Phase", value: progressPhase)
                    }
                    if let b = progressBytes, b.total > 0 {
                        let done = Double(b.done)
                        let total = Double(b.total)
                        ProgressView(value: done, total: total) {
                            let doneGb = done / 1_000_000_000
                            let totalGb = total / 1_000_000_000
                            Text(String(format: "%.2f / %.2f GB (%.0f%%)", doneGb, totalGb,
                                        100 * done / total))
                                .font(.caption)
                        }
                    } else {
                        ProgressView().controlSize(.small)
                    }
                    if !progressLabel.isEmpty {
                        Text(progressLabel)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
            if let r = publishResult, let url = r.url {
                Section("Published") {
                    LabeledContent("Repo URL") {
                        HStack {
                            Text(url).font(.caption).textSelection(.enabled)
                            Button("Open") {
                                if let u = URL(string: url) { NSWorkspace.shared.open(u) }
                            }
                        }
                    }
                    // M44: show the commit URL the upload actually produced.
                    // This is the immediate confirmation that the upload landed
                    // — just showing the repo URL doesn't prove the commit
                    // went through (it could point at an existing older commit).
                    if let commit = r.commitUrl, !commit.isEmpty, commit != url {
                        LabeledContent("Commit") {
                            HStack {
                                Text(commit).font(.caption).textSelection(.enabled)
                                Button("Open") {
                                    if let u = URL(string: commit) { NSWorkspace.shared.open(u) }
                                }
                            }
                        }
                    }
                }
            }
            if let err = errorMessage {
                Section {
                    Label(err, systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                        .font(.caption)
                }
            }
        }
        .formStyle(.grouped)
        .padding(12)
    }

    private var footer: some View {
        HStack {
            Spacer()
            Button {
                dryRunTask?.cancel()
                dryRunTask = Task { await runDryRun() }
            } label: {
                if isDryRunning {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Preview", systemImage: "eye")
                }
            }
            .disabled(isDryRunning || isPublishing || repoName.isEmpty || token.isEmpty)

            Button {
                publishTask = Task { await runPublish() }
            } label: {
                if isPublishing {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Publish", systemImage: "arrow.up.circle.fill")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isDryRunning || isPublishing || repoName.isEmpty || token.isEmpty || dryRunResult == nil)

            // M96 (iter 30): Cancel button visible only during upload.
            // Cancelling the consuming Task → stream onTermination → Python
            // subprocess SIGTERM + 3s SIGKILL escalation. Partial HF repo
            // cleanup is future work (M97).
            if isPublishing {
                Button(role: .destructive) {
                    wasCancelled = true
                    publishTask?.cancel()
                } label: {
                    Label("Cancel upload", systemImage: "stop.circle")
                }
                .keyboardShortcut(.cancelAction)
            }
        }
        .padding(12)
    }

    private func runDryRun() async {
        // M46: validate the repo id BEFORE dispatching — otherwise a typo
        // like "my model" (space) or "justname" (no slash) only surfaces
        // as a cryptic HfHubHTTPError ~30 seconds into the upload.
        if let validationMsg = HFRepoValidator.validationError(repoName) {
            errorMessage = validationMsg
            return
        }
        isDryRunning = true
        errorMessage = nil
        publishResult = nil
        do {
            let r = try await PublishService.dryRun(modelPath: modelPath, repo: repoName, isPrivate: isPrivate, token: token)
            dryRunResult = r
        } catch {
            errorMessage = error.localizedDescription
        }
        isDryRunning = false
    }

    private func runPublish() async {
        if let validationMsg = HFRepoValidator.validationError(repoName) {
            errorMessage = validationMsg
            return
        }
        isPublishing = true
        errorMessage = nil
        progressPhase = ""
        progressBytes = nil
        progressLabel = ""
        progressLog = []
        wasCancelled = false
        // M43 (iter 24): use the streaming variant so the UI gets live
        // progress during the 30+ min upload instead of a dead spinner.
        do {
            for try await event in PublishService.publishWithProgress(
                modelPath: modelPath, repo: repoName,
                isPrivate: isPrivate, token: token) {
                apply(event: event)
            }
            // M137 (iter 59): reaching here means the stream completed
            // without throwing — upload succeeded on the HF side. Even if
            // `wasCancelled == true` (user pressed Cancel very late, after
            // the final chunk was acknowledged), the repo now HAS the
            // files. Showing "Upload cancelled" would be a false negative
            // that makes the user think they need to re-upload.
            //
            // The pre-iter-59 check `if wasCancelled { error } else { success }`
            // had a real race: click Cancel at the same microsecond as the
            // final event lands → loop exits normally → wasCancelled=true
            // from the button handler → user sees "Upload cancelled" despite
            // the upload having completed. CancellationError dispatch is the
            // authoritative "we stopped before the work finished" signal.
            publishResult = PublishResult(
                dryRun: false,
                repo: repoName,
                url: "https://huggingface.co/\(repoName)",
                commitUrl: "https://huggingface.co/\(repoName)",
                filesCount: nil,
                totalSizeBytes: Int(progressBytes?.total ?? 0))
            // M15 (iter 17): wipe the token from @State after a successful
            // publish. If the user leaves this sheet open on their screen,
            // a passerby can't see / copy the token from the SecureField's
            // buffer. On failure we KEEP the token — the user needs to retry
            // and retyping it is worse UX than a ~30-second exposure window.
            token = ""
            if wasCancelled {
                // Document the race outcome in the progress log so the user
                // who hit Cancel understands the upload beat them.
                progressLog.append("[note] Cancel click landed after the final upload event — HF repo is complete.")
            }
        } catch is CancellationError {
            // M137: user-initiated cancel that landed before the stream
            // completed. This is the authoritative "cancelled" branch.
            errorMessage = "Upload cancelled. The HuggingFace repo may contain partial files — delete or overwrite before retrying."
        } catch {
            errorMessage = error.localizedDescription
        }
        isPublishing = false
    }

    private func apply(event: ProgressEvent) {
        switch event.payload {
        case .phase(_, _, let name):
            progressPhase = name
            progressLog.append("phase: \(name)")
        case .tick(let done, let total, let label):
            progressBytes = (Int64(done), Int64(total))
            if let label { progressLabel = label }
        case .message(let level, let text):
            progressLog.append("[\(level)] \(text)")
        case .done:
            // Terminal event — no-op here; stream completion handles the success path.
            break
        case .versionMismatch, .parseError:
            // Non-fatal telemetry — keep a breadcrumb for diagnostics.
            break
        }
    }
}
