// JANGStudio/Tests/JANGStudioTests/ConversionPlanTests.swift
import XCTest
@testable import JANGStudio

final class ConversionPlanTests: XCTestCase {
    func test_defaults() {
        let plan = ConversionPlan()
        XCTAssertNil(plan.sourceURL)
        XCTAssertEqual(plan.family, .jang)
        XCTAssertEqual(plan.profile, "JANG_4K")
        XCTAssertEqual(plan.method, .mse)
        XCTAssertFalse(plan.hadamard)
        XCTAssertEqual(plan.run, .idle)
    }

    func test_isStep1Complete_requiresSourceAndDetection() {
        let p = ConversionPlan()
        XCTAssertFalse(p.isStep1Complete)
        p.sourceURL = URL(fileURLWithPath: "/tmp/x")
        XCTAssertFalse(p.isStep1Complete)
        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isStep1Complete)
    }

    func test_isJANGTQAllowed_matrix() {
        let p = ConversionPlan()
        p.detected = .init(modelType: "llama", isMoE: false, numExperts: 0, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertFalse(p.isJANGTQAllowed)

        p.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed)

        p.detected = .init(modelType: "minimax_m2", isMoE: true, numExperts: 256, isVL: false, dtype: .fp8, totalBytes: 0, shardCount: 1)
        XCTAssertTrue(p.isJANGTQAllowed)

        // GLM deferred to v1.1
        p.detected = .init(modelType: "glm_moe_dsa", isMoE: true, numExperts: 256, isVL: false, dtype: .fp8, totalBytes: 0, shardCount: 1)
        XCTAssertFalse(p.isJANGTQAllowed)
    }

    func test_persistRestore_roundtrip() throws {
        let p = ConversionPlan()
        p.sourceURL = URL(fileURLWithPath: "/tmp/model")
        p.outputURL = URL(fileURLWithPath: "/tmp/out")
        p.profile = "JANG_2L"
        p.family = .jang
        let data = try p.encodeForDefaults()
        let r = try ConversionPlan.decodeFromDefaults(data)
        XCTAssertEqual(r.sourceURL, p.sourceURL)
        XCTAssertEqual(r.outputURL, p.outputURL)
        XCTAssertEqual(r.profile, "JANG_2L")
    }
}
