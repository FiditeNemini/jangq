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
                Task { await runDryRun() }
            } label: {
                if isDryRunning {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Preview", systemImage: "eye")
                }
            }
            .disabled(isDryRunning || isPublishing || repoName.isEmpty || token.isEmpty)

            Button {
                Task { await runPublish() }
            } label: {
                if isPublishing {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Publish", systemImage: "arrow.up.circle.fill")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isDryRunning || isPublishing || repoName.isEmpty || token.isEmpty || dryRunResult == nil)
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
        // M43 (iter 24): use the streaming variant so the UI gets live
        // progress during the 30+ min upload instead of a dead spinner.
        do {
            for try await event in PublishService.publishWithProgress(
                modelPath: modelPath, repo: repoName,
                isPrivate: isPrivate, token: token) {
                apply(event: event)
            }
            // Stream finished without throwing — upload succeeded. The Python
            // side printed the final PublishResult JSON to stdout; in the
            // streaming path we reconstruct it from what we know.
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
