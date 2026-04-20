import SwiftUI
import UniformTypeIdentifiers

struct TestInferenceSheet: View {
    let modelPath: URL
    let isVL: Bool
    let isVideoVL: Bool
    let modelType: String
    let profile: String
    let sizeGb: Double

    @Environment(\.dismiss) private var dismiss
    @State private var vm: TestInferenceViewModel
    @State private var showingSettings = false
    @State private var exportErrorMessage: String? = nil   // M107: surface save failures

    init(modelPath: URL,
         isVL: Bool = false,
         isVideoVL: Bool = false,
         modelType: String = "unknown",
         profile: String = "JANG_4K",
         sizeGb: Double = 0) {
        self.modelPath = modelPath
        self.isVL = isVL
        self.isVideoVL = isVideoVL
        self.modelType = modelType
        self.profile = profile
        self.sizeGb = sizeGb
        self._vm = State(initialValue: TestInferenceViewModel(modelPath: modelPath))
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messagesView
            Divider()
            footer
        }
        .frame(minWidth: 680, minHeight: 540)
        .popover(isPresented: $showingSettings, arrowEdge: .bottom) {
            settingsPopover
        }
        // M107 (iter 35): surface Export Transcript save failures instead of
        // silently swallowing them.
        .alert("Export failed",
               isPresented: Binding(get: { exportErrorMessage != nil },
                                    set: { if !$0 { exportErrorMessage = nil } })) {
            Button("OK") { exportErrorMessage = nil }
        } message: {
            Text(exportErrorMessage ?? "")
        }
    }

    // MARK: - Header
    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Test Inference")
                    .font(.headline)
                HStack(spacing: 8) {
                    Badge(text: modelType)
                    Badge(text: profile)
                    Badge(text: "\(String(format: "%.2f", sizeGb)) GB")
                    if isVideoVL {
                        Badge(text: "video-VL", color: .purple)
                    } else if isVL {
                        Badge(text: "VL", color: .blue)
                    }
                }
                Text(modelPath.path)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            Spacer()
            Button(action: { showingSettings.toggle() }) {
                Image(systemName: "gearshape")
            }
            .buttonStyle(.borderless)
            .help("Inference settings")
            Button("Clear") { vm.clear() }
                .buttonStyle(.borderless)
                .disabled(vm.messages.isEmpty)
            Button("Close") { dismiss() }
        }
        .padding(12)
        .background(Color(.windowBackgroundColor))
    }

    // MARK: - Messages
    private var messagesView: some View {
        ScrollViewReader { scroller in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if vm.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(vm.messages) { msg in
                            ChatBubble(msg: msg).id(msg.id)
                        }
                    }
                    if vm.isGenerating {
                        HStack {
                            ProgressView().controlSize(.small)
                            Text("Generating...").foregroundStyle(.secondary)
                            Spacer()
                        }.padding(.horizontal, 8)
                    }
                    if let err = vm.lastError {
                        Label(err, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.red)
                            .padding(8)
                            .background(Color.red.opacity(0.08))
                            .cornerRadius(6)
                    }
                }
                .padding(16)
            }
            .onChange(of: vm.messages.count) { _, _ in
                if let last = vm.messages.last {
                    withAnimation { scroller.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 36))
                .foregroundStyle(.tertiary)
            Text("No messages yet")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text(isVL ? "Type a prompt or drop an image to get started."
                     : "Type a prompt to see how your converted model responds.")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 40)
    }

    // MARK: - Footer
    private var footer: some View {
        VStack(spacing: 8) {
            if isVL || isVideoVL {
                HStack {
                    if isVL {
                        dropTargetView(
                            label: vm.pendingImagePath?.lastPathComponent ?? "Drop image here",
                            systemImage: "photo",
                            types: [.image]
                        ) { url in
                            vm.pendingImagePath = url
                        }
                    }
                    if isVideoVL {
                        dropTargetView(
                            label: vm.pendingVideoPath?.lastPathComponent ?? "Drop video here",
                            systemImage: "film",
                            types: [.movie, .video]
                        ) { url in
                            vm.pendingVideoPath = url
                        }
                    }
                }
                .padding(.horizontal, 12)
            }

            HStack(spacing: 8) {
                TextField("Prompt", text: $vm.promptText, axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await vm.send() } }
                    .disabled(vm.isGenerating)
                if vm.isGenerating {
                    Button("Cancel", role: .destructive) {
                        Task { await vm.cancel() }
                    }
                } else {
                    Button("Send") {
                        Task { await vm.send() }
                    }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
                    .disabled(vm.promptText.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .padding(.horizontal, 12)

            HStack(spacing: 12) {
                if vm.lastTokensPerSec > 0 {
                    StatView(label: "tok/s", value: String(format: "%.1f", vm.lastTokensPerSec))
                }
                if vm.lastPeakRssMb > 0 {
                    StatView(label: "peak RAM", value: String(format: "%.0f MB", vm.lastPeakRssMb))
                }
                Spacer()
                Button("Export...") { exportTranscript() }
                    .buttonStyle(.borderless)
                    .font(.caption)
                    .disabled(vm.messages.isEmpty)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 10)
        }
        .padding(.top, 8)
        .background(Color(.windowBackgroundColor))
    }

    // MARK: - Settings popover
    private var settingsPopover: some View {
        Form {
            TextField("System prompt", text: $vm.systemPrompt, axis: .vertical)
                .lineLimit(2...4)
            Slider(value: $vm.temperature, in: 0.0...2.0, step: 0.05) {
                Text("Temperature")
            } minimumValueLabel: {
                Text("0.0").font(.caption2)
            } maximumValueLabel: {
                Text("2.0").font(.caption2)
            }
            LabeledContent("Temperature", value: String(format: "%.2f", vm.temperature))
            Stepper(value: $vm.maxTokens, in: 8...4096, step: 32) {
                LabeledContent("Max tokens", value: "\(vm.maxTokens)")
            }
        }
        .padding(16)
        .frame(width: 340)
    }

    // MARK: - Helpers

    private func dropTargetView(label: String,
                                systemImage: String,
                                types: [UTType],
                                onDrop onDropCallback: @escaping (URL) -> Void) -> some View {
        HStack(spacing: 4) {
            Image(systemName: systemImage)
            Text(label).font(.caption).lineLimit(1)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(6)
        .onDrop(of: types, isTargeted: nil) { providers in
            for prov in providers {
                _ = prov.loadObject(ofClass: URL.self) { url, _ in
                    if let url {
                        Task { @MainActor in onDropCallback(url) }
                    }
                }
            }
            return true
        }
    }

    private func exportTranscript() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "chat-transcript.json"
        panel.canCreateDirectories = true
        panel.allowedContentTypes = [.json]
        if panel.runModal() == .OK, let url = panel.url {
            // M107 (iter 35): previously silent-swallowed the error via try?.
            // Disk-full / permission-denied / read-only-volume failures left
            // the user thinking they saved a file that doesn't exist. Now we
            // surface the error as an alert. Successful saves stay quiet —
            // Finder open is the user's feedback that the file landed.
            do {
                try vm.exportTranscript(to: url)
            } catch {
                exportErrorMessage = "Couldn't save transcript to \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }
}

// MARK: - Subviews

private struct Badge: View {
    let text: String
    var color: Color = .secondary
    var body: some View {
        Text(text)
            .font(.caption2)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .cornerRadius(4)
    }
}

private struct StatView: View {
    let label: String
    let value: String
    var body: some View {
        HStack(spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(.tertiary)
            Text(value).font(.caption).monospacedDigit()
        }
    }
}

private struct ChatBubble: View {
    let msg: ChatMessage
    var body: some View {
        HStack {
            if msg.role == .user { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(msg.role.rawValue.uppercased())
                        .font(.caption2)
                        .foregroundStyle(roleColor.opacity(0.8))
                    if let tps = msg.tokensPerSec {
                        Text("\(String(format: "%.1f", tps)) tok/s")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                Text(msg.text)
                    .textSelection(.enabled)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(roleColor.opacity(0.12))
                    .cornerRadius(8)
            }
            if msg.role == .assistant { Spacer(minLength: 40) }
        }
    }
    private var roleColor: Color {
        switch msg.role {
        case .user: return .accentColor
        case .assistant: return .green
        case .system: return .secondary
        }
    }
}
