import SwiftUI

struct GenerateModelCardSheet: View {
    let modelPath: URL
    @Environment(\.dismiss) private var dismiss

    @State private var card: ModelCardResult?
    @State private var loading = true
    @State private var errorMessage: String?
    @State private var saveStatus: String?
    /// M163 (iter 86): retry-Button task handle. The initial `.task { await
    /// generate() }` is auto-cancelled by SwiftUI on sheet dismount, but the
    /// Retry-button-spawned Task is NOT bound to the view lifecycle — it
    /// would orphan the Python subprocess on close. Mirrors iter-85 M162's
    /// sheet-dismiss cancel pattern.
    @State private var retryTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            if loading {
                ProgressView("Generating model card…")
                    .padding(40)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else if let err = errorMessage {
                errorView(err)
            } else if let card = card {
                skeletonWarning
                metadataView(card)
                Divider()
                cardPreview(card)
                Divider()
                footer(card)
            }
        }
        .frame(minWidth: 700, minHeight: 560)
        .task { await generate() }
        .onDisappear { retryTask?.cancel() }
    }

    /// M91 (iter 28): `feedback_readme_standards.md` lists 12 hard
    /// requirements for HF uploads. The auto-generated card is a skeleton
    /// — it ONLY covers the automatable rules (YAML tags, license, quant
    /// config, Python snippet). The non-automatable rules (per-subject
    /// MMLU scores, JANG-vs-MLX side-by-side, speed comparison, Korean
    /// section, per-subject comparison table) require live evals + manual
    /// curation and are not in the skeleton. Users publishing the
    /// skeleton as-is would violate the project standard, silently.
    ///
    /// This banner makes the gap visible at card-preview time so the user
    /// either fills it in before saving, or knowingly accepts a partial
    /// upload.
    private var skeletonWarning: some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text("Skeleton only — not upload-ready")
                    .font(.caption)
                    .fontWeight(.medium)
                Text("Before publishing to HuggingFace, add per-subject MMLU scores, JANG-vs-MLX comparison tables (speed + size + per-subject), and a Korean section per the project's README standards. The generated card covers metadata + Python snippet only.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer()
        }
        .padding(10)
        .background(Color.orange.opacity(0.12))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.orange.opacity(0.4), lineWidth: 1)
        )
        .padding(.horizontal, 12)
        .padding(.top, 8)
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Model Card")
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

    private func metadataView(_ card: ModelCardResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            LabeledContent("License", value: card.license)
            LabeledContent("Base model", value: card.baseModel)
            LabeledContent("Family", value: card.quantizationConfig.family)
            LabeledContent("Profile", value: card.quantizationConfig.profile)
            LabeledContent("Avg bits", value: String(format: "%.2f", card.quantizationConfig.actualBits))
            if let sizeGb = card.quantizationConfig.sizeGb {
                LabeledContent("Size", value: "\(String(format: "%.2f", sizeGb)) GB")
            }
        }
        .padding(12)
    }

    private func cardPreview(_ card: ModelCardResult) -> some View {
        ScrollView {
            Text(card.cardMarkdown)
                .font(.system(.body, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
        }
    }

    private func footer(_ card: ModelCardResult) -> some View {
        HStack(spacing: 8) {
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(card.cardMarkdown, forType: .string)
                saveStatus = "Copied to clipboard."
            } label: {
                Label("Copy markdown", systemImage: "doc.on.doc")
            }
            Button {
                do {
                    try ModelCardService.writeReadme(modelPath: modelPath, content: card.cardMarkdown)
                    saveStatus = "Saved as \(modelPath.appendingPathComponent("README.md").path)"
                } catch {
                    saveStatus = "Save failed: \(error.localizedDescription)"
                }
            } label: {
                Label("Save as README.md in model dir", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.borderedProminent)
            Spacer()
            if let status = saveStatus {
                Text(status).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(12)
    }

    private func errorView(_ err: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 36))
                .foregroundStyle(.red)
            Text("Failed to generate model card")
                .font(.headline)
            Text(err)
                .font(.caption)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Button("Retry") {
                retryTask?.cancel()
                retryTask = Task { await generate() }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(24)
    }

    private func generate() async {
        loading = true
        errorMessage = nil
        card = nil
        do {
            let result = try await ModelCardService.generate(modelPath: modelPath)
            card = result
        } catch {
            errorMessage = error.localizedDescription
        }
        loading = false
    }
}
