// JANGStudio/Tests/JANGStudioTests/PreflightRunnerTests.swift
import XCTest
@testable import JANGStudio

final class PreflightRunnerTests: XCTestCase {
    private var tmp: URL!

    override func setUpWithError() throws {
        tmp = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("pf-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
    }
    override func tearDownWithError() throws { try? FileManager.default.removeItem(at: tmp) }

    func test_missingSourceDirFails() {
        let plan = ConversionPlan()
        plan.sourceURL = URL(fileURLWithPath: "/tmp/definitely_missing_xyz")
        plan.outputURL = tmp
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .sourceReadable && $0.status == .fail })
    }

    func test_jangtqOnLlamaFails() throws {
        let src = tmp.appendingPathComponent("src"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.family = .jangtq
        plan.profile = "JANGTQ2"
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .jangtqArchSupported && $0.status == .fail })
    }

    func test_outputSameAsSourceFails() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src   // same!
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .outputUsable && $0.status == .fail })
    }

    // MARK: - M139 (iter 61): reject nested src/dst
    //
    // Pre-iter-61 outputUsable only rejected `dst == src`. A user picking
    // output INSIDE the source tree (e.g., src=/models/foo, dst=/models/foo/out)
    // passed preflight, let convert write shards into a subdir of the source.
    // Confusing + risky if any future cleanup pass rglobs.

    func test_outputInsideSourceFails() throws {
        let src = tmp.appendingPathComponent("foo")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src.appendingPathComponent("out")   // nested inside src
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let failure = checks.first { $0.id == .outputUsable && $0.status == .fail }
        XCTAssertNotNil(failure, "outputUsable should fail when output is inside source")
        XCTAssertEqual(failure?.hint, "Output cannot be inside the source folder")
    }

    func test_sourceInsideOutputFails() throws {
        // Symmetric: user picks output as the parent of source.
        let parent = tmp.appendingPathComponent("workspace")
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let src = parent.appendingPathComponent("hf-model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = parent   // source nested inside output
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let failure = checks.first { $0.id == .outputUsable && $0.status == .fail }
        XCTAssertNotNil(failure, "outputUsable should fail when source is inside output")
        XCTAssertEqual(failure?.hint, "Source cannot be inside the output folder")
    }

    func test_siblingPrefixPathsDoNotTrigger() throws {
        // Regression: `/a/b` must NOT be rejected as "inside /a/bc". The
        // check uses path + "/" specifically to prevent this.
        let srcParent = tmp.appendingPathComponent("abc")
        try FileManager.default.createDirectory(at: srcParent, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: srcParent.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = srcParent
        plan.outputURL = tmp.appendingPathComponent("abcd")   // sibling with shared prefix
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let found = checks.first { $0.id == .outputUsable }
        XCTAssertEqual(found?.status, .pass,
            "Sibling directories with a shared string prefix (abc vs abcd) must not trigger the nested-path check. Got \(found?.hint ?? "nil")")
    }

    // MARK: - M140 (iter 62): symmetric to M131 — preflight must detect
    // 512+ expert MoE dynamically, not just via a hardcoded name list.
    //
    // Pre-iter-62 check passed on ANY model_type not in the capabilities
    // `knownExpert512Types` hardcoded list. A future 512-expert
    // qwen3_5_moe or deepseek_v3 variant would skip the warning while the
    // underlying recommend.py-side already forces bfloat16 for the same
    // dynamic reason. Symmetric bug across the boundary; fix the preflight
    // to match.

    func test_bf16Warning_fires_on_dynamic_512_experts() throws {
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        // 512-expert qwen3_5_moe — NOT in the default .frozen whitelist
        // `knownExpert512Types: ["minimax_m2", "glm_moe_dsa"]`, but the
        // num_experts count should trigger the dynamic check.
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 512, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16   // user forced fp16 — should warn
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .warn,
            "512-expert qwen3_5_moe + forced fp16 must warn about bfloat16. Got status=\(String(describing: bf?.status)).")
        XCTAssertNotNil(bf?.hint)
        XCTAssertTrue(bf?.hint?.contains("512 experts") ?? false,
            "Hint must include the dynamic expert count so the user understands what triggered the warning. Got hint=\(bf?.hint ?? "nil")")
    }

    func test_bf16Warning_still_fires_for_named_whitelist_types() throws {
        // Regression guard: the named-family path must keep working after
        // iter-62's dynamic extension. minimax_m2 is in the frozen
        // whitelist — should warn on fp16 override.
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"minimax_m2"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        // Named whitelist entry, but with numExperts unknown (0).
        plan.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 0, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .fp8, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .warn)
    }

    func test_bf16Warning_skips_small_moe() throws {
        // Regression guard: 256-expert qwen3_5_moe must stay passing — no
        // over-warn on smaller MoEs.
        let src = tmp.appendingPathComponent("model")
        try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.overrides.forceDtype = .fp16
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        let bf = checks.first { $0.id == .bf16For512Experts }
        XCTAssertEqual(bf?.status, .pass,
            "256-expert qwen3_5_moe must not fire the 512+ warning.")
    }

    func test_hadamardAt2bitWarns() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false,
                              isVideoVL: false, hasGenerationConfig: true, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.profile = "JANG_2S"
        plan.hadamard = true
        let checks = PreflightRunner().run(plan: plan, capabilities: .frozen)
        XCTAssertTrue(checks.contains { $0.id == .hadamardVsLowBits && $0.status == .warn })
    }
}
