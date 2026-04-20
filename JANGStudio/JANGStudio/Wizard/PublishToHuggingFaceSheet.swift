import SwiftUI

struct PublishToHuggingFaceSheet: View {
    let modelPath: URL
    @Environment(\.dismiss) private var dismiss

    @State private var repoName: String = ""
    @State private var isPrivate: Bool = false
    @State private var token: String = ""
    @State private var isDryRunning: Bool = false
    @State private var isPublishing: Bool = false
    @State private var dryRunResult: PublishResult?
    @State private var publishResult: PublishResult?
    @State private var errorMessage: String?

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
        do {
            let r = try await PublishService.publish(modelPath: modelPath, repo: repoName, isPrivate: isPrivate, token: token)
            publishResult = r
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
}
