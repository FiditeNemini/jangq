import SwiftUI

struct QueueView: View {
    @Environment(APIClient.self) var api
    @State private var queue: QueueResponse?
    @State private var isLoading = false
    @State private var timer: Timer?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Header
            HStack {
                Text("Queue")
                    .font(.system(.title, design: .serif))
                    .fontWeight(.medium)
                Spacer()
                if let q = queue {
                    Text("\(q.queueLength) waiting")
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundStyle(Theme.dim)
                }
                Button(action: refresh) {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 12))
                }
                .buttonStyle(.plain)
                .disabled(isLoading)
            }
            .padding(.bottom, 20)

            if let q = queue {
                // Active job
                if let active = q.active {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("NOW PROCESSING")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundStyle(Theme.phaseColor(active.phase))
                            Spacer()
                        }
                        JobCard(job: active, expanded: true)
                    }
                    .padding(.bottom, 20)
                } else {
                    HStack {
                        Image(systemName: "moon.zzz")
                            .foregroundStyle(Theme.muted)
                        Text("No job running")
                            .font(.system(size: 13))
                            .foregroundStyle(Theme.dim)
                    }
                    .padding(.vertical, 16)
                    .frame(maxWidth: .infinity)
                    .background(Theme.surface)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .padding(.bottom, 20)
                }

                // Queued jobs
                if !q.queued.isEmpty {
                    Text("WAITING")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .padding(.bottom, 8)

                    VStack(spacing: 8) {
                        ForEach(q.queued) { job in
                            JobCard(job: job, expanded: false)
                        }
                    }
                    .padding(.bottom, 20)
                }

                // Recent completed
                if !q.recentCompleted.isEmpty {
                    Text("RECENT")
                        .font(.system(size: 10, weight: .bold, design: .monospaced))
                        .foregroundStyle(Theme.muted)
                        .padding(.bottom, 8)

                    VStack(spacing: 8) {
                        ForEach(q.recentCompleted) { job in
                            JobCard(job: job, expanded: false)
                        }
                    }
                }
            } else if isLoading {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                Text("Failed to load queue")
                    .foregroundStyle(Theme.dim)
                    .frame(maxWidth: .infinity)
            }
        }
        .padding(24)
        .onAppear {
            refresh()
            timer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in refresh() }
        }
        .onDisappear {
            timer?.invalidate()
        }
    }

    func refresh() {
        Task {
            isLoading = queue == nil
            do {
                queue = try await api.getQueue()
            } catch {
                // Keep existing data on refresh failure
            }
            isLoading = false
        }
    }
}

// MARK: - Job card

struct JobCard: View {
    let job: JobResponse
    let expanded: Bool
    @Environment(APIClient.self) var api
    @State private var showLogs = false
    /// M186 (iter 121): pre-fix the Cancel/Retry buttons silently
    /// swallowed network errors via `Task { try? await api.X(...) }`.
    /// User clicked, fails (server down, auth expired, network glitch),
    /// nothing visible. iter-35 M107 / iter-90 M167 / iter-120 M185
    /// pattern in another fresh file. Now we surface the error inline
    /// on the card so the user sees what failed without leaving the
    /// queue list.
    @State private var actionError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Top row: model name + phase
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(job.modelId)
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                        .lineLimit(1)
                    HStack(spacing: 8) {
                        Badge(text: job.profile, color: Theme.bitColor(profileBits))
                        if job.queuePosition > 0 {
                            Text("#\(job.queuePosition)")
                                .font(.system(size: 10, weight: .bold, design: .monospaced))
                                .foregroundStyle(Theme.muted)
                        }
                        if !job.user.isEmpty {
                            Text(job.user)
                                .font(.system(size: 10))
                                .foregroundStyle(Theme.muted)
                        }
                    }
                }
                Spacer()
                PhaseIndicator(phase: job.phase, progress: job.progressPct)
            }

            // Progress bar (when active)
            if isActive {
                VStack(alignment: .leading, spacing: 4) {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Theme.border)
                                .frame(height: 6)
                            RoundedRectangle(cornerRadius: 3)
                                .fill(Theme.phaseColor(job.phase))
                                .frame(width: max(0, geo.size.width * job.progressPct / 100), height: 6)
                                .animation(.easeInOut(duration: 0.5), value: job.progressPct)
                        }
                    }
                    .frame(height: 6)

                    HStack {
                        Text(job.phaseDetail)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundStyle(Theme.dim)
                            .lineLimit(1)
                        Spacer()
                        Text("\(Int(job.progressPct))%")
                            .font(.system(size: 11, weight: .bold, design: .monospaced))
                            .foregroundStyle(Theme.phaseColor(job.phase))
                    }
                }
            }

            // Expanded details
            if expanded {
                // Architecture badges
                if !job.architecture.archType.isEmpty {
                    ArchBadges(arch: job.architecture)
                }

                // Phase-specific stats
                if job.phase == "downloading" {
                    HStack(spacing: 16) {
                        StatBox("Downloaded", formatBytes(job.download.bytesDone), mono: true)
                        StatBox("Total", formatBytes(job.download.bytesTotal), mono: true)
                        StatBox("Speed", "\(job.download.speedMbps) MB/s", mono: true)
                        if job.download.etaSeconds > 0 {
                            StatBox("ETA", formatDuration(Double(job.download.etaSeconds)), mono: true)
                        }
                    }
                }

                if job.phase == "quantizing" && job.quantization.tensorsTotal > 0 {
                    HStack(spacing: 16) {
                        StatBox("Tensors", "\(job.quantization.tensorsDone)/\(job.quantization.tensorsTotal)", mono: true)
                        StatBox("Actual bits", String(format: "%.2f", job.quantization.actualBits), mono: true)
                        if !job.quantization.bitHistogram.isEmpty {
                            StatBox("Bit widths", job.quantization.bitHistogram.sorted(by: { $0.key < $1.key }).map { "\($0.key):\($0.value)" }.joined(separator: " "), mono: true)
                        }
                    }
                }

                if job.phase == "uploading" {
                    HStack(spacing: 16) {
                        StatBox("Uploaded", formatBytes(job.upload.bytesDone), mono: true)
                        StatBox("Total", formatBytes(job.upload.bytesTotal), mono: true)
                    }
                }

                if job.phase == "completed" {
                    HStack(spacing: 16) {
                        StatBox("Output", job.result.outputRepo, mono: true)
                        StatBox("Size", "\(job.result.totalSizeGb) GB", mono: true)
                        StatBox("Bits", String(format: "%.2f", job.result.actualBits), mono: true)
                        StatBox("Time", formatDuration(job.result.durationSeconds), mono: true)
                        if job.result.vlReady {
                            Badge(text: "VL Ready", color: .green)
                        }
                    }
                }

                if job.phase == "failed" && !job.error.isEmpty {
                    Text(job.error.components(separatedBy: "\n").first ?? job.error)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }

                // Action buttons
                HStack(spacing: 8) {
                    if isActive || job.phase == "queued" {
                        Button("Cancel") {
                            Task {
                                do {
                                    try await api.cancelJob(job.jobId)
                                    actionError = nil
                                } catch {
                                    actionError = "Cancel failed: \(error.localizedDescription)"
                                }
                            }
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .foregroundStyle(.red)
                    }
                    if job.phase == "failed" || job.phase == "cancelled" {
                        Button("Retry") {
                            Task {
                                do {
                                    _ = try await api.retryJob(job.jobId)
                                    actionError = nil
                                } catch {
                                    actionError = "Retry failed: \(error.localizedDescription)"
                                }
                            }
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .foregroundStyle(.blue)
                    }
                    if job.phase == "completed", let url = URL(string: job.result.outputUrl), !job.result.outputUrl.isEmpty {
                        Button("Open on HF") {
                            NSWorkspace.shared.open(url)
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 11))
                        .foregroundStyle(.blue)
                    }
                    Button(showLogs ? "Hide Logs" : "Logs") {
                        showLogs.toggle()
                    }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundStyle(Theme.dim)
                }

                // M186 (iter 121): surface Cancel/Retry errors here so the
                // user sees what failed without leaving the queue list.
                if let err = actionError {
                    Text(err)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.red)
                        .lineLimit(2)
                        .padding(.top, 2)
                }

                // Logs
                if showLogs {
                    LogView(jobId: job.jobId)
                }
            }
        }
        .padding(12)
        .background(job.isActive ? Theme.phaseColor(job.phase).opacity(0.03) : Color.clear)
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(job.isActive ? Theme.phaseColor(job.phase).opacity(0.2) : Theme.border, lineWidth: 1)
        )
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    var isActive: Bool {
        !["queued", "completed", "failed", "cancelled"].contains(job.phase)
    }

    var profileBits: Int {
        allProfiles.first(where: { $0.name == job.profile })?.bits ?? 4
    }
}

// MARK: - Log view

struct LogView: View {
    let jobId: String
    @Environment(APIClient.self) var api
    @State private var lines: [String] = []

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 1) {
                ForEach(Array(lines.enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(Theme.dim)
                        .textSelection(.enabled)
                }
            }
            .padding(8)
        }
        .frame(maxHeight: 200)
        .background(Color.black.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .task {
            do {
                lines = try await api.getJobLogs(jobId)
            } catch {
                lines = ["Failed to load logs: \(error.localizedDescription)"]
            }
        }
    }
}
