import SwiftUI

struct UsageExamplesSheet: View {
    let modelPath: URL
    @Environment(\.dismiss) private var dismiss

    @State private var selectedLang: ExampleLanguage = .python
    @State private var snippets: [ExampleLanguage: String] = [:]
    @State private var loadingLangs: Set<ExampleLanguage> = []
    @State private var errorByLang: [ExampleLanguage: String] = [:]
    @State private var saveErrorMessage: String? = nil   // M107: surface save failures

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            tabBar
            Divider()
            snippetArea
            Divider()
            footer
        }
        .frame(minWidth: 720, minHeight: 540)
        // M107 (iter 35): surface Save-to-file failures instead of swallowing.
        .alert("Save failed",
               isPresented: Binding(get: { saveErrorMessage != nil },
                                    set: { if !$0 { saveErrorMessage = nil } })) {
            Button("OK") { saveErrorMessage = nil }
        } message: {
            Text(saveErrorMessage ?? "")
        }
        .task {
            // Pre-fetch all four snippets in parallel so tab switches are instant.
            await withTaskGroup(of: Void.self) { group in
                for lang in ExampleLanguage.allCases {
                    group.addTask { await fetchSnippet(lang) }
                }
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Usage Examples")
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

    private var tabBar: some View {
        HStack(spacing: 0) {
            ForEach(ExampleLanguage.allCases) { lang in
                Button {
                    selectedLang = lang
                } label: {
                    VStack(spacing: 4) {
                        Text(lang.displayName)
                            .font(.subheadline)
                            .fontWeight(selectedLang == lang ? .semibold : .regular)
                        Rectangle()
                            .fill(selectedLang == lang ? Color.accentColor : Color.clear)
                            .frame(height: 2)
                    }
                    .frame(maxWidth: .infinity)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.top, 4)
    }

    private var snippetArea: some View {
        ScrollView {
            if loadingLangs.contains(selectedLang) {
                ProgressView().padding(40)
            } else if let err = errorByLang[selectedLang] {
                VStack(alignment: .leading, spacing: 8) {
                    Label("Failed to load snippet", systemImage: "exclamationmark.triangle.fill")
                        .foregroundStyle(.red)
                    Text(err)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                    Button("Retry") { Task { await fetchSnippet(selectedLang) } }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(16)
            } else if let text = snippets[selectedLang] {
                Text(text)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            } else {
                Text("")
            }
        }
    }

    private var footer: some View {
        HStack {
            if let text = snippets[selectedLang] {
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc")
                }
                Button {
                    saveToFile(text: text)
                } label: {
                    Label("Save to file…", systemImage: "square.and.arrow.down")
                }
            }
            Spacer()
            Text("Language: \(selectedLang.displayName)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(12)
    }

    private func fetchSnippet(_ lang: ExampleLanguage) async {
        await MainActor.run {
            loadingLangs.insert(lang)
            errorByLang.removeValue(forKey: lang)
        }
        do {
            let s = try await ExamplesService.fetch(modelPath: modelPath, lang: lang)
            await MainActor.run {
                snippets[lang] = s.snippet
                loadingLangs.remove(lang)
            }
        } catch {
            await MainActor.run {
                errorByLang[lang] = error.localizedDescription
                loadingLangs.remove(lang)
            }
        }
    }

    private func saveToFile(text: String) {
        let panel = NSSavePanel()
        let base = modelPath.lastPathComponent
        panel.nameFieldStringValue = "\(base)-usage.\(selectedLang.fileExtension)"
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            // M107 (iter 35): silent-swallowed the failure via `try?` →
            // user's disk-full / permission-denied / bad-path error was
            // invisible. Now we surface it as an alert.
            guard let data = text.data(using: .utf8) else {
                saveErrorMessage = "Internal error: snippet is not valid UTF-8"
                return
            }
            do {
                try data.write(to: url)
            } catch {
                saveErrorMessage = "Couldn't save to \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
    }
}
