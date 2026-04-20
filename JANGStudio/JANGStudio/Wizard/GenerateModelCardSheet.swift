import SwiftUI

struct GenerateModelCardSheet: View {
    let modelPath: URL
    @Environment(\.dismiss) private var dismiss

    @State private var card: ModelCardResult?
    @State private var loading = true
    @State private var errorMessage: String?
    @State private var saveStatus: String?

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
                metadataView(card)
                Divider()
                cardPreview(card)
                Divider()
                footer(card)
            }
        }
        .frame(minWidth: 700, minHeight: 560)
        .task { await generate() }
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
            Button("Retry") { Task { await generate() } }
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
