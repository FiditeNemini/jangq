// JANGStudio/JANGStudio/Wizard/Steps/SourceStep.swift
import SwiftUI

struct SourceStep: View {
    @Bindable var coord: WizardCoordinator
    // M143 (iter 65): access AppSettings so applyRecommendation can tell
    // whether the current profile is still at the user's configured
    // default (safe to overwrite) vs. a value the user manually picked in
    // ProfileStep (must preserve). Pre-iter-65 the check was hardcoded
    // `== "JANG_4K"` which misfired for any user whose
    // settings.defaultProfile wasn't JANG_4K.
    @Environment(AppSettings.self) private var settings
    @State private var isDetecting = false
    @State private var isRecommending = false
    @State private var errorText: String?
    @State private var recommendation: Recommendation?
    /// M135 (iter 57): stale-task handle tracking. User picks folder A →
    /// detection starts (Task A, ~5s) → user changes mind, picks folder B →
    /// detection starts (Task B, ~1s). Without this handle, Task A continues
    /// running after Task B finishes and eventually stomps
    /// `coord.plan.detected` / `self.recommendation` with folder A's result.
    /// The user sees A's metadata while sourceURL points at B — the
    /// conversion later uses wrong metadata → misdetected architecture →
    /// wrong quantization profile applied. Subprocess kill propagates
    /// through SourceDetector's iter-34 M105 withTaskCancellationHandler
    /// wrap and RecommendationService's iter-33 M101 wrap.
    @State private var detectionTask: Task<Void, Never>?

    var body: some View {
        Form {
            // MARK: - Folder picker
            Section {
                HStack {
                    if let url = coord.plan.sourceURL {
                        Text(url.lastPathComponent).font(.headline)
                        Text(url.deletingLastPathComponent().path)
                            .font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("No folder selected").foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button("Choose Folder…", action: pickFolder)
                }
            } header: {
                HStack(spacing: 4) {
                    Text("Source model folder")
                    InfoHint("Pick a HuggingFace model directory — one containing `config.json` and `.safetensors` shards. JANG Studio auto-detects the architecture and recommends a conversion plan.")
                }
            }

            // MARK: - Detected
            if let detected = coord.plan.detected {
                Section("Detected") {
                    LabeledContent("Model type", value: detected.modelType)
                    LabeledContent(
                        "Parameters",
                        value: detected.isMoE ? "MoE · \(detected.numExperts) experts" : "Dense"
                    )
                    LabeledContent("Source dtype", value: detected.dtype.rawValue.uppercased())
                    LabeledContent(
                        "Disk",
                        value: "\(detected.totalBytes / 1_000_000_000) GB (\(detected.shardCount) shards)"
                    )
                    if detected.isVideoVL {
                        Label("Vision + Video", systemImage: "film")
                            .foregroundStyle(.purple)
                    } else if detected.isVL {
                        Label("Vision (image)", systemImage: "eye")
                            .foregroundStyle(.blue)
                    }
                    // No safetensors → hard-fail with a specific hint.
                    if detected.shardCount == 0 {
                        Label(
                            "No .safetensors files found in this folder. You likely picked a parent folder, a docs folder, or a download that didn't complete. Pick the actual model directory.",
                            systemImage: "xmark.octagon.fill"
                        )
                        .foregroundStyle(.red)
                        .font(.callout)
                    }
                }
            }

            // MARK: - Recommendation (beginner-friendly)
            if let rec = recommendation {
                Section {
                    Text(rec.beginnerSummary)
                        .font(.callout)
                        .padding(.vertical, 4)

                    Divider()

                    Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
                        GridRow {
                            Text("Family").foregroundStyle(.secondary)
                            Text(rec.recommended.family.uppercased())
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.family)
                        }
                        GridRow {
                            Text("Profile").foregroundStyle(.secondary)
                            Text(rec.recommended.profile)
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.profile)
                        }
                        GridRow {
                            Text("Method").foregroundStyle(.secondary)
                            Text(rec.recommended.method.uppercased())
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.method)
                        }
                        GridRow {
                            Text("Hadamard").foregroundStyle(.secondary)
                            Text(rec.recommended.hadamard ? "On" : "Off")
                                .fontWeight(.medium)
                            InfoHint(rec.whyEachChoice.hadamard)
                        }
                        if let forceDtype = rec.recommended.forceDtype {
                            GridRow {
                                Text("Force dtype").foregroundStyle(.secondary)
                                Text(forceDtype)
                                    .fontWeight(.medium)
                                InfoHint(rec.whyEachChoice.forceDtype)
                            }
                        }
                    }

                    ForEach(rec.warnings, id: \.self) { warning in
                        Label(warning, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .font(.caption)
                            .padding(.top, 2)
                    }

                    if !rec.recommended.alternatives.isEmpty {
                        DisclosureGroup("Other options") {
                            VStack(alignment: .leading, spacing: 8) {
                                ForEach(rec.recommended.alternatives) { alt in
                                    VStack(alignment: .leading, spacing: 2) {
                                        HStack(spacing: 4) {
                                            if let fam = alt.family {
                                                Text(fam.uppercased())
                                                    .font(.caption2)
                                                    .padding(.horizontal, 5)
                                                    .padding(.vertical, 2)
                                                    .background(Color.secondary.opacity(0.2))
                                                    .cornerRadius(4)
                                            }
                                            Text(alt.profile).fontWeight(.medium)
                                        }
                                        Text(alt.useWhen)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                } header: {
                    HStack(spacing: 4) {
                        Text("Recommended for this model")
                        InfoHint("Smart defaults auto-filled based on what JANG Studio detected. Most beginners can skip Steps 2 and 3 and hit Start in Step 4.")
                    }
                }
            }

            // MARK: - Detection status
            if let errorText {
                Label(errorText, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
            }
            if isDetecting || isRecommending {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(isDetecting ? "Inspecting source…" : "Building recommendation…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            // MARK: - Continue
            if coord.plan.isStep1Complete {
                Button("Continue →") { coord.active = .architecture }
                    .buttonStyle(.borderedProminent)
                    .keyboardShortcut(.defaultAction)
            }
        }
        .formStyle(.grouped)
        .padding()
    }

    private func pickFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        if panel.runModal() == .OK, let url = panel.url {
            coord.plan.sourceURL = url
            coord.plan.detected = nil
            recommendation = nil
            errorText = nil
            // M135 (iter 57): cancel any previous detection task before
            // starting a new one. Without this, a slow previous detection
            // can stomp the new URL's state after it finishes.
            detectionTask?.cancel()
            detectionTask = Task { await detectAndRecommend(url: url) }
        }
    }

    private func detectAndRecommend(url: URL) async {
        // Step A: fast inspect-source call
        isDetecting = true
        do {
            let detected = try await SourceDetector.inspect(url: url)
            // M135: guard against stale-task overwrite. If this task was
            // cancelled while waiting on the subprocess (user picked a
            // different folder), don't mutate state — the newer task owns it.
            guard !Task.isCancelled else { return }
            await MainActor.run { coord.plan.detected = detected }
        } catch {
            await MainActor.run {
                // Also guard the error-path — a cancelled task's subprocess
                // kill shouldn't surface as a user-facing "Detection failed".
                guard !Task.isCancelled else { return }
                errorText = "Detection failed: \(error.localizedDescription)"
                isDetecting = false
            }
            return
        }
        guard !Task.isCancelled else { return }
        await MainActor.run { isDetecting = false }

        // Step B: recommendation call (also fast, reads same config.json)
        isRecommending = true
        defer { Task { @MainActor in isRecommending = false } }
        do {
            let rec = try await RecommendationService.fetch(modelURL: url)
            guard !Task.isCancelled else { return }
            await MainActor.run {
                self.recommendation = rec
                self.applyRecommendation(rec)
            }
        } catch {
            await MainActor.run {
                guard !Task.isCancelled else { return }
                // Recommendation failure is soft — user can still convert manually.
                errorText = "Recommendation failed (conversion still works): \(error.localizedDescription)"
            }
        }
    }

    /// Fill in ConversionPlan defaults based on the recommendation. Only touches
    /// fields the user hasn't manually changed (heuristic: fields still at their
    /// initial defaults get replaced; anything else is preserved).
    private func applyRecommendation(_ rec: Recommendation) {
        let plan = coord.plan

        // M144 (iter 66): family was previously overwritten
        // unconditionally every time the user picked a new source. That
        // created an INCONSISTENT-STATE bug: user picks source A, goes to
        // ProfileStep, manually switches to JANGTQ2 (family=.jangtq), then
        // re-picks source A again (or a similar source). Recommendation
        // comes back with family=jang → unconditional overwrite sets
        // family=.jang. Profile preserved as "JANGTQ2" (iter-65 M143 fix
        // does the right thing). Result: family=.jang but profile=JANGTQ2
        // — an invalid pair. Pre-M144, the user ended up stuck until they
        // manually re-synced ProfileStep.
        //
        // Fix: couple family + profile. If profile was preserved (user
        // manually set it), family stays preserved too. If profile was
        // overwritten, derive family from the new profile name so the
        // pair is always consistent.
        let seedDefault = settings.defaultProfile.isEmpty ? "JANG_4K" : settings.defaultProfile
        if plan.profile == seedDefault {
            // User hasn't touched profile → both profile AND family get
            // overwritten together, derived from the new profile so they
            // can't disagree.
            plan.profile = rec.recommended.profile
            plan.family = plan.profile.hasPrefix("JANGTQ") ? .jangtq : .jang
        }
        // else: user manually picked a profile in ProfileStep. Preserve
        // BOTH profile and family to keep them in sync.

        // M145 (iter 67): extend iter-66 M144's "user hasn't touched"
        // preservation to method, hadamard, and forceDtype. Pre-iter-67
        // these three fields were ALL unconditionally overwritten every
        // time the user re-picked a source — silently wiping any manual
        // ProfileStep choices the user made (e.g., toggling Hadamard off
        // for a 2-bit profile, switching to RTN method for speed, forcing
        // bfloat16). Same UX pathology as the pre-iter-65 profile bug.
        //
        // "User hasn't touched" signal = field still matches what
        // applyDefaults seeded it with (from settings.default*).

        // Method: preserve if user manually picked something different
        // from their Settings default.
        let recMethod: QuantMethod = switch rec.recommended.method {
        case "rtn": .rtn
        case "mse-all": .mseAll
        default: .mse
        }
        let seedMethod: QuantMethod = switch settings.defaultMethod.lowercased() {
        case "rtn": .rtn
        case "mse-all", "mseall", "mse_all": .mseAll
        case "mse": .mse
        default: .mse   // applyDefaults leaves method untouched on unknown; seed with init default
        }
        if plan.method == seedMethod {
            plan.method = recMethod
        }

        // Hadamard: preserve if user manually toggled from their Settings
        // default. Pre-iter-67 a user who turned hadamard off for a 2-bit
        // convert got it silently re-enabled on source re-pick.
        if plan.hadamard == settings.defaultHadamardEnabled {
            plan.hadamard = rec.recommended.hadamard
        }

        // Force dtype — only set if recommendation has one AND the user
        // hasn't already chosen an override. Pre-iter-67 this overwrote
        // any user-set forceDtype when rec supplied one (which it does
        // for 512+ expert MoE models — so a user who manually forced fp16
        // for speed on a smaller variant got bfloat16 slammed back on
        // re-pick).
        if plan.overrides.forceDtype == nil,
           let forceDtypeStr = rec.recommended.forceDtype {
            let forceDtype: SourceDtype? = switch forceDtypeStr {
            case "bfloat16": .bf16
            case "float16": .fp16
            case "float8_e4m3fn", "float8_e5m2": .fp8
            default: nil
            }
            if let ft = forceDtype {
                plan.overrides.forceDtype = ft
            }
        }

        // Block size
        if plan.overrides.forceBlockSize == nil {
            plan.overrides.forceBlockSize = rec.recommended.blockSize
        }
    }
}

enum SourceDetector {
    struct SourceInfo: Decodable {
        let model_type: String
        let is_moe: Bool
        let num_experts: Int
        let dtype: String
        let total_bytes: Int64
        let shard_count: Int
        let is_vl: Bool
        let is_video_vl: Bool
        let has_generation_config: Bool
        let jangtq_compatible: Bool
    }

    static func inspect(url: URL) async throws -> ArchitectureSummary {
        // M105 (iter 34): previously used synchronous `proc.waitUntilExit()`
        // inside this `async` function, blocking whatever thread the async
        // context was running on (often the main actor via SourceStep's
        // .task) AND missing Task-cancel propagation. A user picking folder
        // A then quickly picking folder B would orphan the A-subprocess
        // AND momentarily freeze the UI. Same fix template as iter 33's
        // service-sweep: DispatchQueue for the subprocess thread +
        // withTaskCancellationHandler + ProcessHandle for SIGTERM on cancel.
        let handle = ProcessHandle()
        let data: Data = try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                DispatchQueue.global().async {
                    do {
                        let proc = Process()
                        proc.executableURL = BundleResolver.pythonExecutable
                        proc.arguments = ["-m", "jang_tools", "inspect-source", "--json", url.path]
                        let out = Pipe()
                        let err = Pipe()
                        proc.standardOutput = out
                        proc.standardError = err
                        try proc.run()
                        handle.set(process: proc)
                        proc.waitUntilExit()
                        if proc.terminationStatus != 0 {
                            // M120 (iter 43): include stderr in the surfaced
                            // error so SourceStep's errorText banner tells the
                            // user WHY inspect-source failed (e.g. "config.json
                            // at … is not valid JSON (line 1, col 3)"). Pre-fix,
                            // a malformed config.json produced a useless
                            // "inspect-source exited 1" with the real reason
                            // discarded on the floor.
                            let stderr = String(
                                data: err.fileHandleForReading.readDataToEndOfFile(),
                                encoding: .utf8
                            )?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                            let desc = stderr.isEmpty
                                ? "inspect-source exited \(proc.terminationStatus)"
                                : "inspect-source exited \(proc.terminationStatus): \(stderr)"
                            cont.resume(throwing: NSError(
                                domain: "SourceDetector",
                                code: Int(proc.terminationStatus),
                                userInfo: [NSLocalizedDescriptionKey: desc]))
                            return
                        }
                        cont.resume(returning: out.fileHandleForReading.readDataToEndOfFile())
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            handle.cancel()
        }

        let info = try JSONDecoder().decode(SourceInfo.self, from: data)
        let dtype: SourceDtype = switch info.dtype {
            case "bfloat16": .bf16
            case "float16": .fp16
            case "float8_e4m3fn", "float8_e5m2": .fp8
            default: .unknown
        }
        return .init(modelType: info.model_type, isMoE: info.is_moe, numExperts: info.num_experts,
                     isVL: info.is_vl, isVideoVL: info.is_video_vl,
                     hasGenerationConfig: info.has_generation_config,
                     dtype: dtype, totalBytes: info.total_bytes, shardCount: info.shard_count)
    }
}
