import SwiftUI

struct SettingsView: View {
    @Environment(APIClient.self) var api
    @State private var health: HealthResponse?
    // M185 (iter 120): pre-fix the "Check Connection" button silently
    // dropped errors (`} catch { health = nil }`). User clicked, saw
    // nothing happen, no idea what failed (wrong URL? bad token? server
    // down?). Surface the error string so the user has actionable info.
    // iter-35 M107 / iter-90 M167 pattern applied to a fresh file.
    @State private var lastError: String?

    var body: some View {
        @Bindable var client = api

        VStack(alignment: .leading, spacing: 0) {
            Text("Settings")
                .font(.system(.title, design: .serif))
                .fontWeight(.medium)
                .padding(.bottom, 20)

            VStack(alignment: .leading, spacing: 16) {
                // Server URL
                VStack(alignment: .leading, spacing: 6) {
                    Text("Server URL")
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(Theme.dim)
                    TextField("http://localhost:8420", text: $client.baseURL)
                        .textFieldStyle(.plain)
                        .font(.system(size: 14, design: .monospaced))
                        .padding(10)
                        .background(Theme.surface)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.border))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                // API Key
                VStack(alignment: .leading, spacing: 6) {
                    Text("API Key (optional)")
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(Theme.dim)
                    SecureField("Leave empty if auth is disabled", text: $client.apiKey)
                        .textFieldStyle(.plain)
                        .font(.system(size: 14, design: .monospaced))
                        .padding(10)
                        .background(Theme.surface)
                        .overlay(RoundedRectangle(cornerRadius: 6).stroke(Theme.border))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }

                Divider()

                // Health
                if let h = health {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("SERVER STATUS")
                            .font(.system(size: 10, weight: .bold, design: .monospaced))
                            .foregroundStyle(Theme.muted)

                        HStack(spacing: 20) {
                            HStack(spacing: 6) {
                                Circle()
                                    .fill(h.status == "ok" ? Color.green : Color.red)
                                    .frame(width: 8, height: 8)
                                Text(h.status.uppercased())
                                    .font(.system(size: 12, weight: .bold, design: .monospaced))
                            }
                            StatBox("Jobs", "\(h.totalJobs)", mono: true)
                            StatBox("Queue", "\(h.queueLength)", mono: true)
                            StatBox("Disk", "\(h.diskFreeGb) / \(h.diskTotalGb) GB", mono: true)
                        }

                        if let model = h.processingModel {
                            HStack(spacing: 6) {
                                Text("Processing:")
                                    .font(.system(size: 11))
                                    .foregroundStyle(Theme.dim)
                                Text(model)
                                    .font(.system(size: 11, design: .monospaced))
                            }
                        }
                    }
                }

                Button("Check Connection") {
                    Task {
                        do {
                            health = try await api.getHealth()
                            lastError = nil
                        } catch {
                            health = nil
                            lastError = "Connection failed: \(error.localizedDescription)"
                        }
                    }
                }
                .buttonStyle(.bordered)

                if let err = lastError {
                    Text(err)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.red)
                        .padding(.top, 4)
                }
            }
        }
        .padding(24)
        .task {
            // Initial health check on view appear. Errors silently null the
            // health stat (no banner) — the visible-on-mount banner would
            // be jarring if the user just hasn't started the server yet.
            // The "Check Connection" button below surfaces explicit errors
            // when the user actively requests a check.
            do { health = try await api.getHealth() } catch { health = nil }
        }
    }
}
