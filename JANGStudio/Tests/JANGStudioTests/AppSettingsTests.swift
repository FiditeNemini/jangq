import XCTest
@testable import JANGStudio

@MainActor
final class AppSettingsTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Wipe defaults before each test to avoid cross-contamination
        UserDefaults.standard.removeObject(forKey: "JANGStudioSettings")
    }

    func test_defaults_match_spec() {
        let s = AppSettings()
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
        XCTAssertEqual(s.defaultFamily, "jang")
        XCTAssertEqual(s.defaultMethod, "mse")
        XCTAssertEqual(s.defaultCalibrationSamples, 256)
        XCTAssertEqual(s.outputNamingTemplate, "{basename}-{profile}")
        XCTAssertEqual(s.logVerbosity, .normal)
        XCTAssertEqual(s.tickThrottleMs, 100)
        XCTAssertTrue(s.revealInFinderOnFinish)
        XCTAssertTrue(s.metalPipelineCacheEnabled)
        XCTAssertTrue(s.copyDiagnosticsAlwaysVisible)
    }

    func test_render_output_name_substitutes_tokens() {
        let s = AppSettings()
        s.outputNamingTemplate = "{basename}-{profile}-{family}"
        let result = s.renderOutputName(basename: "Qwen3-0.6B", profile: "JANG_4K", family: "jang")
        XCTAssertEqual(result, "Qwen3-0.6B-JANG_4K-jang")
    }

    func test_render_output_name_with_date_token() {
        let s = AppSettings()
        s.outputNamingTemplate = "{basename}-{date}"
        let result = s.renderOutputName(basename: "model", profile: "x", family: "y")
        // Result should start with "model-" and have a date following
        XCTAssertTrue(result.hasPrefix("model-20"))   // 2026-
    }

    func test_persist_and_reload() {
        let s1 = AppSettings()
        s1.defaultProfile = "JANG_2L"
        s1.tickThrottleMs = 50
        s1.logVerbosity = .verbose
        s1.persist()

        let s2 = AppSettings()   // loads from UserDefaults
        XCTAssertEqual(s2.defaultProfile, "JANG_2L")
        XCTAssertEqual(s2.tickThrottleMs, 50)
        XCTAssertEqual(s2.logVerbosity, .verbose)
    }

    func test_reset_restores_defaults() {
        let s = AppSettings()
        s.defaultProfile = "JANG_2L"
        s.tickThrottleMs = 999
        s.reset()
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
        XCTAssertEqual(s.tickThrottleMs, 100)
    }

    func test_update_channel_roundtrip() {
        let s1 = AppSettings()
        s1.updateChannel = .beta
        s1.persist()
        let s2 = AppSettings()
        XCTAssertEqual(s2.updateChannel, .beta)
    }
}
