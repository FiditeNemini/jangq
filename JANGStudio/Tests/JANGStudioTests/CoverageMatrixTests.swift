// JANGStudio/Tests/JANGStudioTests/CoverageMatrixTests.swift
import XCTest
@testable import JANGStudio

/// Full cartesian audit of (profile × architecture class × dtype) combinations.
/// Ensures no combination silently breaks preflight, verifier gating, or arg routing.
@MainActor
final class CoverageMatrixTests: XCTestCase {
    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("cov-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: tmp) }

    // MARK: - Architecture class definitions

    private static let archClasses: [(model: String, experts: Int, isVL: Bool, dtype: SourceDtype, label: String)] = [
        ("llama",           0,   false, .bf16, "dense llama BF16"),
        ("qwen3_5_moe",     256, false, .bf16, "qwen3_5_moe BF16"),
        ("qwen3_5_moe",     256, true,  .bf16, "qwen3_5_moe VL BF16"),
        ("qwen3_5_moe",     256, false, .fp8,  "qwen3_5_moe FP8"),
        ("minimax_m2",      256, false, .fp8,  "minimax FP8"),
        ("minimax_m2",      256, false, .bf16, "minimax BF16"),
        ("glm_moe_dsa",     256, false, .fp8,  "glm FP8 (JANGTQ unsupported in v1)"),
        ("deepseek_v32",    256, false, .bf16, "deepseek BF16"),
    ]

    private func makePlan(_ c: (model: String, experts: Int, isVL: Bool, dtype: SourceDtype, label: String),
                          family: Family, profile: String) throws -> ConversionPlan {
        let src = tmp.appendingPathComponent("src-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"\#(c.model)"}"#.write(
            to: src.appendingPathComponent("config.json"),
            atomically: true, encoding: .utf8)

        let p = ConversionPlan()
        p.sourceURL = src
        p.outputURL = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        p.family = family
        p.profile = profile
        p.detected = .init(modelType: c.model, isMoE: c.experts > 1, numExperts: c.experts,
                            isVL: c.isVL, isVideoVL: false, hasGenerationConfig: true,
                            dtype: c.dtype, totalBytes: 0, shardCount: 1)
        return p
    }

    // MARK: - Preflight matrix

    func test_preflight_acceptsAllJANGProfiles_forSupportedArchClass() throws {
        // For each arch class, ALL JANG profiles should at least not get rejected on
        // the jangtq-arch-support check (since family == .jang). Some may warn
        // (hadamard/bf16), but `jangtqArchSupported` must be .pass for all JANG.
        let jangProfiles = ["JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L",
                            "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L",
                            "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L",
                            "JANG_5K", "JANG_6K", "JANG_6M"]
        for arch in Self.archClasses {
            for prof in jangProfiles {
                let p = try makePlan(arch, family: .jang, profile: prof)
                let checks = PreflightRunner().run(plan: p)
                let jangtqCheck = checks.first { $0.id == .jangtqArchSupported }!
                XCTAssertEqual(jangtqCheck.status, .pass,
                    "JANG family on \(arch.label) profile=\(prof) should bypass JANGTQ arch check")
            }
        }
    }

    func test_preflight_rejectsJANGTQOnNonWhitelistedArchs() throws {
        let nonWhitelisted = Self.archClasses.filter { $0.model != "qwen3_5_moe" && $0.model != "minimax_m2" }
        for arch in nonWhitelisted {
            for prof in ["JANGTQ2", "JANGTQ3", "JANGTQ4"] {
                let p = try makePlan(arch, family: .jangtq, profile: prof)
                let checks = PreflightRunner().run(plan: p)
                let jangtqCheck = checks.first { $0.id == .jangtqArchSupported }!
                XCTAssertEqual(jangtqCheck.status, .fail,
                    "JANGTQ on \(arch.label) profile=\(prof) should be REJECTED by preflight")
            }
        }
    }

    func test_preflight_acceptsJANGTQOnWhitelistedArchsWithCorrectDtype() throws {
        let whitelisted = Self.archClasses.filter {
            ($0.model == "qwen3_5_moe" || $0.model == "minimax_m2") &&
            ($0.dtype == .bf16 || $0.dtype == .fp8)
        }
        for arch in whitelisted {
            for prof in ["JANGTQ2", "JANGTQ3", "JANGTQ4"] {
                let p = try makePlan(arch, family: .jangtq, profile: prof)
                let checks = PreflightRunner().run(plan: p)
                let archCheck = checks.first { $0.id == .jangtqArchSupported }!
                let dtypeCheck = checks.first { $0.id == .jangtqSourceDtype }!
                XCTAssertEqual(archCheck.status, .pass,
                    "JANGTQ on \(arch.label) profile=\(prof) arch check should pass")
                XCTAssertEqual(dtypeCheck.status, .pass,
                    "JANGTQ on \(arch.label) profile=\(prof) dtype check should pass")
            }
        }
    }

    // MARK: - CLI args routing × all profiles

    func test_cliArgs_everyProfileEveryArchProducesValidInvocation() throws {
        let jang = ["JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L",
                    "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L",
                    "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L",
                    "JANG_5K", "JANG_6K", "JANG_6M"]
        let jangtq = ["JANGTQ2", "JANGTQ3", "JANGTQ4"]

        for arch in Self.archClasses {
            for prof in jang {
                let p = try makePlan(arch, family: .jang, profile: prof)
                let args = CLIArgsBuilder.args(for: p)
                XCTAssertEqual(args[0], "-m", "\(arch.label)/\(prof): args[0] != -m")
                XCTAssertEqual(args[1], "jang_tools", "\(arch.label)/\(prof): args[1] != jang_tools")
                XCTAssertEqual(args[2], "convert", "\(arch.label)/\(prof): args[2] != convert")
                XCTAssertTrue(args.contains("--progress=json"), "\(arch.label)/\(prof): missing --progress=json")
                XCTAssertTrue(args.contains("--quiet-text"), "\(arch.label)/\(prof): missing --quiet-text")
                XCTAssertTrue(args.contains(prof), "\(arch.label)/\(prof): profile not in args")
            }
            if arch.model == "qwen3_5_moe" || arch.model == "minimax_m2" {
                for prof in jangtq {
                    let p = try makePlan(arch, family: .jangtq, profile: prof)
                    let args = CLIArgsBuilder.args(for: p)
                    XCTAssertTrue(args[1].contains("jangtq"),
                        "\(arch.label) profile=\(prof) should route to a JANGTQ converter, got \(args[1])")
                    XCTAssertTrue(args.contains(prof), "\(arch.label)/\(prof): profile not in args")
                }
            }
        }
    }

    // MARK: - Verifier: missing-file matrix

    /// Build a minimal output dir missing ONE required file and confirm the corresponding
    /// check fails. Ensures every tokenizer/chat/config/shard requirement is actually enforced.
    func test_verifier_flagsEachMissingRequiredFile() async throws {
        let required: [(remove: String, expectedFailID: VerifyID, label: String)] = [
            ("jang_config.json",             .jangConfigExists, "missing jang_config"),
            ("tokenizer.json",               .tokenizerFiles,   "missing tokenizer.json"),
            ("tokenizer_config.json",        .tokenizerFiles,   "missing tokenizer_config.json"),
            ("special_tokens_map.json",      .tokenizerFiles,   "missing special_tokens_map.json"),
            ("model.safetensors.index.json", .shardsMatchIndex, "missing shard index"),
        ]
        for scenario in required {
            let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
            try Self.writeGoodFixture(at: out)
            try FileManager.default.removeItem(at: out.appendingPathComponent(scenario.remove))

            let p = ConversionPlan()
            p.outputURL = out
            p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256,
                               isVL: false, isVideoVL: false, hasGenerationConfig: true,
                               dtype: .bf16, totalBytes: 0, shardCount: 1)
            let checks = await PostConvertVerifier().run(plan: p, skipPythonValidate: true)
            XCTAssertTrue(
                checks.contains { $0.id == scenario.expectedFailID && $0.status == .fail },
                "scenario `\(scenario.label)` did not trigger \(scenario.expectedFailID) fail"
            )
        }
    }

    func test_verifier_chatTemplateRequiredEitherInline_OR_Jinja() async throws {
        // Case A: neither inline nor .jinja — fail
        let noneURL = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: noneURL)
        // Overwrite tokenizer_config.json to remove the chat_template key
        try #"{"tokenizer_class":"Qwen2Tokenizer"}"#.write(
            to: noneURL.appendingPathComponent("tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        let planA = ConversionPlan()
        planA.outputURL = noneURL
        planA.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256,
                               isVL: false, isVideoVL: false, hasGenerationConfig: true,
                               dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checksA = await PostConvertVerifier().run(plan: planA, skipPythonValidate: true)
        XCTAssertTrue(checksA.contains { $0.id == .chatTemplate && $0.status == .fail },
            "no chat template inline/jinja should fail .chatTemplate")

        // Case B: .jinja file present, tokenizer_config.json has no chat_template → still passes
        let jinjaURL = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: jinjaURL)
        try #"{"tokenizer_class":"Qwen2Tokenizer"}"#.write(
            to: jinjaURL.appendingPathComponent("tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        try "{% for m in messages %}{{m.content}}{% endfor %}".write(
            to: jinjaURL.appendingPathComponent("chat_template.jinja"),
            atomically: true, encoding: .utf8)
        let planB = ConversionPlan()
        planB.outputURL = jinjaURL
        planB.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256,
                               isVL: false, isVideoVL: false, hasGenerationConfig: true,
                               dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checksB = await PostConvertVerifier().run(plan: planB, skipPythonValidate: true)
        XCTAssertTrue(checksB.contains { $0.id == .chatTemplate && $0.status == .pass },
            ".jinja file should satisfy chatTemplate even without inline template")
    }

    func test_verifier_VLPreprocessorRequiredForVLModels() async throws {
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        // Good fixture has no preprocessor_config.json — writes deliberately omit it
        let p = ConversionPlan()
        p.outputURL = out
        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256,
                           isVL: true, isVideoVL: false, hasGenerationConfig: true,
                           dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: p, skipPythonValidate: true)
        XCTAssertTrue(checks.contains { $0.id == .vlPreprocessors && $0.status == .fail },
            "VL model without preprocessor_config.json should fail vlPreprocessors")
    }

    func test_verifier_minimaxCustomPyRequiredWhenMiniMaxDetected() async throws {
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        // Good fixture has no modeling_*.py / configuration_*.py — expected fail for minimax
        let p = ConversionPlan()
        p.outputURL = out
        p.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256,
                           isVL: false, isVideoVL: false, hasGenerationConfig: true,
                           dtype: .fp8, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: p, skipPythonValidate: true)
        XCTAssertTrue(checks.contains { $0.id == .miniMaxCustomPy && $0.status == .fail })

        // Now add the files and re-run — should pass
        try "# stub".write(to: out.appendingPathComponent("modeling_minimax.py"),
                           atomically: true, encoding: .utf8)
        try "# stub".write(to: out.appendingPathComponent("configuration_minimax.py"),
                           atomically: true, encoding: .utf8)
        let checks2 = await PostConvertVerifier().run(plan: p, skipPythonValidate: true)
        XCTAssertTrue(checks2.contains { $0.id == .miniMaxCustomPy && $0.status == .pass })
    }

    func test_verifier_tokenizerClassConcreteIsWarnNotFail() async throws {
        // Write fixture then overwrite tokenizer_config.json to use TokenizersBackend
        let out = tmp.appendingPathComponent("out-\(UUID().uuidString)")
        try Self.writeGoodFixture(at: out)
        try #"{"chat_template":"{% for m in messages %}{{m.content}}{% endfor %}","tokenizer_class":"TokenizersBackend"}"#.write(
            to: out.appendingPathComponent("tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        let p = ConversionPlan()
        p.outputURL = out
        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256,
                           isVL: false, isVideoVL: false, hasGenerationConfig: true,
                           dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: p, skipPythonValidate: true)
        let tokClass = checks.first { $0.id == .tokenizerClassConcrete }!
        XCTAssertEqual(tokClass.status, .warn, "TokenizersBackend should warn, not fail")
        XCTAssertFalse(tokClass.required, "tokenizerClassConcrete is warn-only (required: false)")
    }

    // MARK: - Helpers

    /// Writes a minimal valid JANG output directory with: config.json, jang_config.json,
    /// tokenizer.json, tokenizer_config.json (inline chat_template + concrete class),
    /// special_tokens_map.json, model.safetensors.index.json (1 shard), and the shard file.
    /// Deliberately omits preprocessor_config.json so VL tests can assert its absence.
    private static func writeGoodFixture(at url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe","torch_dtype":"bfloat16"}"#.write(
            to: url.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try #"{"format":"jang","format_version":"2.0","capabilities":{"arch":"qwen3_5_moe"},"quantization":{"bit_widths_used":[4],"block_size":64}}"#.write(
            to: url.appendingPathComponent("jang_config.json"), atomically: true, encoding: .utf8)
        try #"{"model":{"type":"BPE"}}"#.write(
            to: url.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try #"{"chat_template":"{% for m in messages %}{{m.content}}{% endfor %}","tokenizer_class":"Qwen2Tokenizer"}"#.write(
            to: url.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)
        try #"{"bos_token":"<s>","eos_token":"</s>"}"#.write(
            to: url.appendingPathComponent("special_tokens_map.json"), atomically: true, encoding: .utf8)
        try #"{"weight_map":{"a":"model-00001-of-00001.safetensors"}}"#.write(
            to: url.appendingPathComponent("model.safetensors.index.json"), atomically: true, encoding: .utf8)
        FileManager.default.createFile(
            atPath: url.appendingPathComponent("model-00001-of-00001.safetensors").path,
            contents: Data())
    }
}
