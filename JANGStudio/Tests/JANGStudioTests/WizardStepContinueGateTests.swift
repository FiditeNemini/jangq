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

    // MARK: - M135 (iter 57): SourceStep stale-task cancellation
    //
    // When the user picks folder A, a detection task starts. If they
    // immediately pick folder B before A's task finishes, A must be
    // cancelled — otherwise A's result eventually stomps B's detected
    // state and the user sees A's metadata while sourceURL points at B.

    func test_sourceStep_tracks_detection_task_handle() throws {
        let src = try stepSource("SourceStep.swift")
        XCTAssertTrue(
            src.contains("@State private var detectionTask: Task<Void, Never>?"),
            """
            SourceStep must store the detection Task handle so it can be
            cancelled when the user picks a new folder. Without this,
            concurrent detections race and the slower one stomps state.
            """
        )
    }

    func test_sourceStep_pickFolder_cancels_previous_task() throws {
        let src = try stepSource("SourceStep.swift")
        // The pickFolder body must call `detectionTask?.cancel()` BEFORE
        // starting the new task. Locate the panel.runModal() block and
        // check the cancel call is present AND precedes the new Task.
        guard let pickRange = src.range(of: "if panel.runModal() == .OK, let url = panel.url {") else {
            XCTFail("could not locate pickFolder's panel.runModal block")
            return
        }
        let body = String(src[pickRange.upperBound...].prefix(800))
        XCTAssertTrue(
            body.contains("detectionTask?.cancel()"),
            """
            pickFolder must call detectionTask?.cancel() to tear down any
            previous detection. Body after panel.runModal:
            \(body)
            """
        )
        // Cancel must come before the new Task assignment.
        if let cancelIdx = body.range(of: "detectionTask?.cancel()")?.lowerBound,
           let newTaskIdx = body.range(of: "detectionTask = Task")?.lowerBound {
            XCTAssertLessThan(cancelIdx, newTaskIdx,
                "cancel() must precede the new Task assignment")
        }
    }

    // MARK: - M136 (iter 58): RunStep auto-start must only fire on .idle
    //
    // Pre-iter-58 RunStep's .onAppear was:
    //     .onAppear { Task { await start() } }
    // and `start()` only guarded `run != .running`. A user nav-backing
    // from VerifyStep via the sidebar (to inspect logs of a finished
    // conversion) would re-trigger start(), wipe logs, overwrite the
    // already-written output folder, and start a fresh conversion the
    // user didn't ask for. The fix gates the onAppear on run == .idle.
    // Retry buttons still call start() directly — they're after a failure/
    // cancel so the weaker `!= .running` guard there is correct.

    func test_runStep_onAppear_only_auto_starts_when_idle() throws {
        let src = try stepSource("RunStep.swift")
        // The onAppear block must check `coord.plan.run == .idle` before
        // calling start(). A bare `.onAppear { Task { await start() } }`
        // is the bug.
        //
        // Strip comment lines first — `.onAppear` appears in the iter-58
        // M136 rationale comment, which would make a naive "first .onAppear
        // + look nearby" test miss the actual modifier.
        let codeOnly = src.split(separator: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
            .joined(separator: "\n")

        guard codeOnly.contains(".onAppear") else {
            XCTFail("could not locate RunStep onAppear in code")
            return
        }
        // Pattern that must appear in code: the onAppear body must gate on
        // `coord.plan.run == .idle`.
        XCTAssertTrue(
            codeOnly.contains("coord.plan.run == .idle"),
            """
            RunStep .onAppear must check `coord.plan.run == .idle` before
            calling start(). Pre-iter-58 a nav-back from VerifyStep would
            restart a completed conversion, wiping logs and overwriting
            the already-written output folder.
            """
        )
    }

    func test_sourceStep_guards_mutations_with_isCancelled() throws {
        let src = try stepSource("SourceStep.swift")
        // detectAndRecommend must have `guard !Task.isCancelled else { return }`
        // before mutating coord.plan.detected / recommendation / errorText.
        // Count the guards — expect at least 3 (one per mutation site after
        // a suspension point).
        let guardCount = src.components(separatedBy: "guard !Task.isCancelled else { return }").count - 1
        XCTAssertGreaterThanOrEqual(
            guardCount, 3,
            """
            detectAndRecommend must guard state mutations with
            `guard !Task.isCancelled else { return }` after every await
            suspension point. Found \(guardCount) guards; need at least 3
            (Step A success, Step A error, Step B success/error).
            """
        )
    }
}
