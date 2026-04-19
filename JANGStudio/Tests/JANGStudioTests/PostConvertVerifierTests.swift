// JANGStudio/Tests/JANGStudioTests/PostConvertVerifierTests.swift
import XCTest
@testable import JANGStudio

final class PostConvertVerifierTests: XCTestCase {
    private func fixture(_ name: String) -> URL {
        Bundle(for: Self.self).url(forResource: name, withExtension: nil, subdirectory: nil)
            ?? Bundle(for: Self.self).bundleURL.appendingPathComponent("Fixtures/\(name)")
    }

    func test_goodOutputAllRequiredPass() async throws {
        let url = fixture("good_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 1)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let requiredFails = checks.filter { $0.required && $0.status == .fail }
        XCTAssertTrue(requiredFails.isEmpty, "unexpected required fails: \(requiredFails.map(\.id))")
    }

    func test_brokenOutputFlagsChatTemplateAndShardMismatch() async throws {
        let url = fixture("broken_output")
        let plan = ConversionPlan()
        plan.outputURL = url
        plan.detected = .init(modelType: "qwen3_5_moe", isMoE: true, numExperts: 256, isVL: false, dtype: .bf16, totalBytes: 0, shardCount: 2)
        let checks = await PostConvertVerifier().run(plan: plan, skipPythonValidate: true)
        let failedIDs = checks.filter { $0.status == .fail }.map { $0.id }
        XCTAssertTrue(failedIDs.contains(.chatTemplate))
        XCTAssertTrue(failedIDs.contains(.shardsMatchIndex))
    }
}
