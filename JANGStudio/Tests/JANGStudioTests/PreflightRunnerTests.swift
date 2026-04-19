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
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .sourceReadable && $0.status == .fail })
    }

    func test_jangtqOnLlamaFails() throws {
        let src = tmp.appendingPathComponent("src"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"llama"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.family = .jangtq
        plan.profile = "JANGTQ2"
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .jangtqArchSupported && $0.status == .fail })
    }

    func test_outputSameAsSourceFails() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = src   // same!
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 0)
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .outputUsable && $0.status == .fail })
    }

    func test_hadamardAt2bitWarns() throws {
        let src = tmp.appendingPathComponent("model"); try FileManager.default.createDirectory(at: src, withIntermediateDirectories: true)
        try #"{"model_type":"qwen3_5_moe"}"#.write(to: src.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let plan = ConversionPlan()
        plan.sourceURL = src
        plan.outputURL = tmp.appendingPathComponent("out")
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 0)
        plan.profile = "JANG_2S"
        plan.hadamard = true
        let checks = PreflightRunner().run(plan: plan)
        XCTAssertTrue(checks.contains { $0.id == .hadamardVsLowBits && $0.status == .warn })
    }
}
