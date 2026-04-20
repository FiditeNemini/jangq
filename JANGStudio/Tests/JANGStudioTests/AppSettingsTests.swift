import XCTest
@testable import JANGStudio

@MainActor
final class AppSettingsTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Wipe defaults before each test to avoid cross-contamination
        UserDefaults.standard.removeObject(forKey: "JANGStudioSettings")
        UserDefaults.standard.removeObject(forKey: BundleResolver.pythonOverrideDefaultsKey)
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: "JANGStudioSettings")
        UserDefaults.standard.removeObject(forKey: BundleResolver.pythonOverrideDefaultsKey)
        super.tearDown()
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

    // MARK: - Iter 9: M61 — pythonOverridePath now actually flows to BundleResolver

    func test_python_override_persists_to_leaf_consumer_key() {
        let s = AppSettings()
        s.pythonOverridePath = "/opt/homebrew/bin/python3.11"
        s.persist()
        XCTAssertEqual(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "/opt/homebrew/bin/python3.11",
            "persist() must mirror pythonOverridePath to the leaf-consumer UserDefaults key"
        )
    }

    func test_clearing_python_override_removes_leaf_key() {
        // Set and persist first
        let s = AppSettings()
        s.pythonOverridePath = "/some/path"
        s.persist()
        XCTAssertNotNil(UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey))

        // Now clear
        s.pythonOverridePath = ""
        s.persist()
        XCTAssertNil(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "clearing pythonOverridePath must REMOVE the leaf key (so env/bundled fallbacks take over)"
        )
    }

    func test_bundle_resolver_reads_user_default() {
        UserDefaults.standard.set("/usr/bin/env", forKey: BundleResolver.pythonOverrideDefaultsKey)
        XCTAssertEqual(BundleResolver.pythonExecutable.path, "/usr/bin/env")
    }

    func test_bundle_resolver_ignores_empty_user_default() {
        // Empty string should NOT be treated as a valid override — fall through to env/bundled.
        UserDefaults.standard.set("", forKey: BundleResolver.pythonOverrideDefaultsKey)
        // Exact return depends on env but it should NOT be the empty string we set.
        XCTAssertNotEqual(BundleResolver.pythonExecutable.path, "")
    }

    func test_load_resyncs_leaf_consumer_key_on_fresh_process() {
        // Simulate: previous process persisted a python override. Current process
        // starts with a DIFFERENT leaf-key value (or no leaf-key at all).
        // load() on init must re-sync the leaf key from the Snapshot so consumers
        // don't get stale data.
        let s1 = AppSettings()
        s1.pythonOverridePath = "/path/from/prior/session"
        s1.persist()
        // Simulate leaf-key drift (someone else wrote a different value)
        UserDefaults.standard.set("/wrong/path", forKey: BundleResolver.pythonOverrideDefaultsKey)
        // Fresh AppSettings() calls load(), which should re-mirror the Snapshot value
        let s2 = AppSettings()
        XCTAssertEqual(s2.pythonOverridePath, "/path/from/prior/session")
        XCTAssertEqual(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "/path/from/prior/session",
            "load() must re-sync leaf key from the authoritative Snapshot"
        )
    }

    func test_reset_clears_leaf_consumer_key() {
        let s = AppSettings()
        s.pythonOverridePath = "/some/path"
        s.persist()
        XCTAssertNotNil(UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey))
        s.reset()
        XCTAssertNil(
            UserDefaults.standard.string(forKey: BundleResolver.pythonOverrideDefaultsKey),
            "reset() must also clear leaf-consumer mirrors (reset() calls persist internally)"
        )
    }
}
