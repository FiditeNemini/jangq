// JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift
import Foundation

struct PostConvertVerifier {
    @MainActor func run(plan: ConversionPlan, capabilities: Capabilities = .frozen, skipPythonValidate: Bool = false) async -> [VerifyCheck] {
        guard let out = plan.outputURL else {
            return [.init(id: .jangConfigExists, title: "jang_config.json exists",
                          status: .fail, required: true, hint: "No output dir")]
        }
        var checks: [VerifyCheck] = []
        let jangCfgURL = out.appendingPathComponent("jang_config.json")

        // #1 jang_config exists + JSON valid
        let jangCfg = (try? JSONSerialization.jsonObject(with: Data(contentsOf: jangCfgURL)) as? [String: Any]) ?? [:]
        checks.append(.init(id: .jangConfigExists, title: "jang_config.json exists",
                            status: jangCfg.isEmpty ? .fail : .pass, required: true,
                            hint: jangCfg.isEmpty ? "Missing or unparseable jang_config.json" : nil))

        // #2 format + format_version
        let fmt = (jangCfg["format"] as? String) ?? ""
        let ver = (jangCfg["format_version"] as? String) ?? ""
        let okFmt = fmt == "jang" && (ver.hasPrefix("2.") || ver.hasPrefix("3."))
        checks.append(.init(id: .jangConfigFormat, title: "jang format v2+", status: okFmt ? .pass : .fail,
                            required: true, hint: okFmt ? nil : "format=\(fmt) version=\(ver)"))

        // #3 schema via python (skipped in unit tests)
        if !skipPythonValidate {
            let ok = await Self.runJangValidate(outputDir: out)
            checks.append(.init(id: .schemaValid, title: "jang validate passes", status: ok ? .pass : .fail,
                                required: true, hint: ok ? nil : "Run `jang validate` for details"))
        } else {
            checks.append(.init(id: .schemaValid, title: "jang validate passes", status: .pass, required: true, hint: "skipped in test"))
        }

        // #4 capabilities
        let caps = (jangCfg["capabilities"] as? [String: Any]) ?? [:]
        checks.append(.init(id: .capabilitiesPresent, title: "capabilities stamp present",
                            status: caps.isEmpty ? .fail : .pass, required: true,
                            hint: caps.isEmpty ? "jang_config.capabilities missing" : nil))

        // #5 chat template (inline | .jinja | .json all accepted)
        let hasJinja = FileManager.default.fileExists(atPath: out.appendingPathComponent("chat_template.jinja").path)
        let hasChatJSON = FileManager.default.fileExists(atPath: out.appendingPathComponent("chat_template.json").path)
        let tokCfgData = try? Data(contentsOf: out.appendingPathComponent("tokenizer_config.json"))
        let tokCfg = (tokCfgData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
        let hasInline = !((tokCfg["chat_template"] as? String) ?? "").isEmpty
        let hasChat = hasJinja || hasChatJSON || hasInline
        checks.append(.init(id: .chatTemplate, title: "Chat template present",
                            status: hasChat ? .pass : .fail, required: true,
                            hint: hasChat ? nil : "No chat_template inline / .jinja / .json file"))

        // #6 tokenizer files
        let hasJSON = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer.json").path)
        let hasModel = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer.model").path)
        let hasCfg = FileManager.default.fileExists(atPath: out.appendingPathComponent("tokenizer_config.json").path)
        let hasSpecial = FileManager.default.fileExists(atPath: out.appendingPathComponent("special_tokens_map.json").path)
        let okTok = (hasJSON || hasModel) && hasCfg && hasSpecial
        checks.append(.init(id: .tokenizerFiles, title: "Tokenizer files complete",
                            status: okTok ? .pass : .fail, required: true,
                            hint: okTok ? nil : "Missing tokenizer.json|.model, tokenizer_config, or special_tokens_map"))

        // #7 shards match index
        let idxURL = out.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: idxURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = obj["weight_map"] as? [String: String] {
            let shards = Set(map.values)
            let onDisk = Set(shards.filter { FileManager.default.fileExists(atPath: out.appendingPathComponent($0).path) })
            let ok = shards == onDisk
            checks.append(.init(id: .shardsMatchIndex, title: "Shards match index",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Index references \(shards.count) shards, \(onDisk.count) on disk"))
        } else {
            checks.append(.init(id: .shardsMatchIndex, title: "Shards match index",
                                status: .fail, required: true, hint: "model.safetensors.index.json missing"))
        }

        // #8 VL preprocessors
        if plan.detected?.isVL == true {
            let ok = FileManager.default.fileExists(atPath: out.appendingPathComponent("preprocessor_config.json").path)
            checks.append(.init(id: .vlPreprocessors, title: "VL preprocessor configs",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Missing preprocessor_config.json for VL model"))
        }

        // #8b Video VL preprocessor — only required when detected.isVideoVL
        if plan.detected?.isVideoVL == true {
            let ok = FileManager.default.fileExists(atPath: out.appendingPathComponent("video_preprocessor_config.json").path)
            checks.append(.init(id: .videoPreprocessors, title: "Video VL preprocessor config",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "Missing video_preprocessor_config.json for video-VL model"))
        }

        // #9 MiniMax custom .py
        if plan.detected?.modelType == "minimax_m2" {
            let files = (try? FileManager.default.contentsOfDirectory(atPath: out.path)) ?? []
            let hasPyModel = files.contains { $0.hasPrefix("modeling_") && $0.hasSuffix(".py") }
            let hasPyCfg = files.contains { $0.hasPrefix("configuration_") && $0.hasSuffix(".py") }
            let ok = hasPyModel && hasPyCfg
            checks.append(.init(id: .miniMaxCustomPy, title: "MiniMax modeling_*.py + configuration_*.py",
                                status: ok ? .pass : .fail, required: true,
                                hint: ok ? nil : "HF trust_remote_code will fail without these"))
        }

        // #10 tokenizer class concrete.
        // Memory ref `feedback_jang_studio_audit_coverage.md` makes this a hard
        // requirement: swift-transformers (Osaurus, vmlx-swift-lm) throws
        // `unsupportedTokenizer("TokenizersBackend")` and the model won't load.
        // Upgraded from warn-only to required=true in Ralph iter 5; the Python
        // side (convert.py Osaurus fix) now auto-remaps, so this verifier row
        // catches sources that slipped past the remap (e.g. an unmapped
        // model_type would leave the blocklist value intact).
        let cls = (tokCfg["tokenizer_class"] as? String) ?? ""
        let classOK = !cls.isEmpty && !capabilities.tokenizerClassBlocklist.contains(cls)
        checks.append(.init(id: .tokenizerClassConcrete, title: "Tokenizer class concrete",
                            status: classOK ? .pass : .fail, required: true,
                            hint: classOK ? nil : "tokenizer_class=\(cls) is in blocklist — Osaurus/vmlx-swift-lm will fail to load. Re-run convert; if it persists, add your model_type to the Osaurus remap in convert.py."))

        // #11 generation_config.json — HF consumers expect this. Warn only (HF will fall back to defaults).
        let hasGenCfg = FileManager.default.fileExists(atPath: out.appendingPathComponent("generation_config.json").path)
        checks.append(.init(id: .generationConfig, title: "generation_config.json present",
                            status: hasGenCfg ? .pass : .warn, required: false,
                            hint: hasGenCfg ? nil : "HF downstream loaders may fall back to unexpected defaults"))

        // #12 layer count sanity — config.json must have num_hidden_layers > 0
        let cfgData = try? Data(contentsOf: out.appendingPathComponent("config.json"))
        let cfgObj = (cfgData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
        let layerCount = (cfgObj["num_hidden_layers"] as? Int)
            ?? ((cfgObj["text_config"] as? [String: Any])?["num_hidden_layers"] as? Int)
            ?? 0
        checks.append(.init(id: .layerCountSane, title: "num_hidden_layers > 0",
                            status: layerCount > 0 ? .pass : .fail, required: true,
                            hint: layerCount > 0 ? "\(layerCount) layers" : "config.json missing or has num_hidden_layers=0"))

        return checks
    }

    /// Default wall-time budget for `jang validate`. Validation is file-inspection
    /// only (no model load, no inference) — it should complete in ≤5 seconds
    /// on any reasonable machine. 60s is a 10× safety margin that still caps
    /// the worst case so a hung Python subprocess can't stall VerifyStep
    /// indefinitely if the user leaves the wizard open in the background.
    /// Exposed as a parameter for tests; prod callers use the default.
    static let defaultValidateTimeoutSeconds: Double = 60

    /// Run `jang validate` and return whether it exited 0 within the timeout.
    /// M42 (iter 19): previously used `proc.waitUntilExit()` which blocks the
    /// calling thread indefinitely if the subprocess hangs. Now uses the same
    /// actor-friendly pattern as PythonRunner/InferenceRunner (iter 3): a
    /// CheckedContinuation tied to `terminationHandler`, plus a Task.sleep
    /// timeout race that SIGTERMs on expiry.
    static func runJangValidate(outputDir: URL,
                                timeoutSeconds: Double = defaultValidateTimeoutSeconds) async -> Bool {
        let proc = Process()
        proc.executableURL = BundleResolver.pythonExecutable
        proc.arguments = ["-m", "jang_tools", "validate", outputDir.path]
        proc.standardOutput = Pipe(); proc.standardError = Pipe()
        do { try proc.run() } catch { return false }

        // Race the natural exit against a timeout sleep. First winner resolves
        // the continuation; if the timeout wins, SIGTERM the subprocess + a
        // 3-second SIGKILL escalation so a truly deadlocked child still dies.
        //
        // M101 (iter 33): wrap in withTaskCancellationHandler so a cancelled
        // consumer Task (e.g., user navigating away from VerifyStep mid-run)
        // also tears down the subprocess instead of waiting for the 60s
        // default timeout. See iter-32 cross-layer cancel sweep.
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
                // Atomic-ish guard — Process APIs don't expose a built-in "already
                // resolved" flag, and a double-resume on CheckedContinuation is a
                // fatal error. Wrap with a dispatch queue so exit + timeout +
                // cancel can't resume in parallel.
                let lock = DispatchQueue(label: "PostConvertVerifier.runJangValidate")
                var resolved = false

                proc.terminationHandler = { p in
                    lock.sync {
                        if resolved { return }
                        resolved = true
                        cont.resume(returning: p.terminationStatus == 0)
                    }
                }

                Task.detached {
                    try? await Task.sleep(for: .seconds(timeoutSeconds))
                    lock.sync {
                        if resolved { return }
                        resolved = true
                        // SIGTERM + 3s SIGKILL escalation, same pattern as PythonRunner.
                        if proc.isRunning { proc.terminate() }
                        Task.detached {
                            try? await Task.sleep(for: .seconds(3))
                            if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
                        }
                        cont.resume(returning: false)
                    }
                }
            }
        } onCancel: {
            // On consumer-Task cancel, SIGTERM the subprocess. The
            // terminationHandler will then resolve the continuation with
            // the terminated exit code (→ returns false, which is fine:
            // cancelled verifications are treated as "did not succeed").
            if proc.isRunning { proc.terminate() }
        }
    }
}
