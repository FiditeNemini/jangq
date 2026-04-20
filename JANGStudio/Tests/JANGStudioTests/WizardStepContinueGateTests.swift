// JANGStudio/Tests/JANGStudioTests/WizardStepContinueGateTests.swift
//
// M134 (iter 56): peer-helper parity on wizard step "Continue" buttons.
//
// Each wizard step has a forward-navigation button (e.g. "Continue →",
// "Looks right → Profile", "Start Conversion"). Pre-iter-56 the 5 steps
// drifted:
//   - SourceStep:       wraps the button in `if isStep1Complete { Button… }`
//   - ArchitectureStep: NO gate (could advance with stale/nil detected state)
//   - ProfileStep:      `.disabled(!allMandatoryPass())` (preflight-based)
//   - RunStep:          wraps in `if run == .succeeded { Button… }`
//   - VerifyStep:       "Convert another" resets state, always safe
//
// Iter 56 adds `.disabled(!coord.plan.isStep2Complete)` to Architecture.
// This test pins the gating invariant via source-inspection so a future
// refactor can't silently remove it.
//
// We test code-SHAPE rather than runtime behavior because SwiftUI's
// `.disabled` state isn't trivially inspectable outside a ViewInspector /
// XCUITest harness. If/when the JANGStudio test suite adopts ViewInspector,
// this test can migrate to a behavioral check.

import XCTest

final class WizardStepContinueGateTests: XCTestCase {

    private static let stepsRoot: URL = {
        // Test bundle is at .../DerivedData/.../Build/Products/Debug/JANGStudio.app/Contents/PlugIns/JANGStudioTests.xctest
        // Source tree lives in the repo — we look it up relative to the test runner's working directory.
        // SRCROOT isn't exposed to XCTest at runtime, so fall back to the common repo path.
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // JANGStudioTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // JANGStudio (xcodeproj root)
            .appendingPathComponent("JANGStudio/Wizard/Steps")
    }()

    private func stepSource(_ stepFile: String) throws -> String {
        let url = Self.stepsRoot.appendingPathComponent(stepFile)
        return try String(contentsOf: url, encoding: .utf8)
    }

    func test_architectureStep_continue_is_gated() throws {
        let src = try stepSource("ArchitectureStep.swift")
        // The "Looks right → Profile" button must be followed (within ~400
        // chars) by a `.disabled(` modifier on the step-completeness
        // predicate OR wrapped in an `if coord.plan.isStep` block.
        guard let btnRange = src.range(of: #"Button("Looks right → Profile")"#) else {
            XCTFail("could not locate Architecture step Continue button")
            return
        }
        let after = String(src[btnRange.upperBound...].prefix(400))
        let hasDisabled = after.contains(".disabled(!coord.plan.isStep2Complete)")
                       || after.contains(".disabled(!coord.plan.isStep1Complete)")
        XCTAssertTrue(hasDisabled, """
            M134 regression: ArchitectureStep's "Looks right → Profile"
            button must be gated by `.disabled(!coord.plan.isStepNComplete)`.
            Pre-iter-56 it was unconditional, allowing step-advance with
            stale/nil detected state. Context after the button:
            \(after)
            """)
    }

    func test_sourceStep_continue_is_gated() throws {
        let src = try stepSource("SourceStep.swift")
        // SourceStep uses the `if coord.plan.isStep1Complete { Button… }` style.
        XCTAssertTrue(
            src.contains("if coord.plan.isStep1Complete"),
            "SourceStep Continue button must remain gated via `if isStep1Complete`."
        )
    }

    func test_profileStep_continue_is_gated() throws {
        let src = try stepSource("ProfileStep.swift")
        XCTAssertTrue(
            src.contains(".disabled(!allMandatoryPass())"),
            "ProfileStep Start Conversion button must remain gated via preflight."
        )
    }

    func test_runStep_continue_is_gated() throws {
        let src = try stepSource("RunStep.swift")
        XCTAssertTrue(
            src.contains("if coord.plan.run == .succeeded"),
            "RunStep Continue → Verify button must remain gated by run status."
        )
    }
}
