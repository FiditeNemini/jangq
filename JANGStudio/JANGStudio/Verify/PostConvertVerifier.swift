// JANGStudio/JANGStudio/Verify/PostConvertVerifier.swift
import Foundation

struct PostConvertVerifier {
    func run(plan: ConversionPlan, skipPythonValidate: Bool = false) async -> [VerifyCheck] {
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

        // #5 chat template
        let hasJinja = FileManager.default.fileExists(atPath: out.appendingPathComponent("chat_template.jinja").path)
        let tokCfgData = try? Data(contentsOf: out.appendingPathComponent("tokenizer_config.json"))
        let tokCfg = (tokCfgData.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }) ?? [:]
        let hasInline = !((tokCfg["chat_template"] as? String) ?? "").isEmpty
        let hasChat = hasJinja || hasInline
        checks.append(.init(id: .chatTemplate, title: "Chat template present",
                            status: hasChat ? .pass : .fail, required: true,
                            hint: hasChat ? nil : "No chat_template inline or .jinja file"))

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

        // #10 tokenizer class concrete
        let cls = (tokCfg["tokenizer_class"] as? String) ?? ""
        let classOK = !cls.isEmpty && cls != "TokenizersBackend"
        checks.append(.init(id: .tokenizerClassConcrete, title: "Tokenizer class concrete",
                            status: classOK ? .pass : .warn, required: false,
                            hint: classOK ? nil : "tokenizer_class=\(cls) — Osaurus serving may fail"))

        return checks
    }

    private static func runJangValidate(outputDir: URL) async -> Bool {
        let proc = Process()
        proc.executableURL = BundleResolver.pythonExecutable
        proc.arguments = ["-m", "jang_tools", "validate", outputDir.path]
        proc.standardOutput = Pipe(); proc.standardError = Pipe()
        do { try proc.run() } catch { return false }
        proc.waitUntilExit()
        return proc.terminationStatus == 0
    }
}
