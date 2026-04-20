import XCTest
@testable import JANGStudio

@MainActor
final class AppSettingsTests: XCTestCase {
    private static let allLeafKeys: [String] = [
        BundleResolver.pythonOverrideDefaultsKey,
        BundleResolver.tickThrottleMsDefaultsKey,
        BundleResolver.mlxThreadCountDefaultsKey,
        BundleResolver.customJangToolsPathDefaultsKey,
    ]

    override func setUp() {
        super.setUp()
        // Wipe defaults before each test to avoid cross-contamination.
        UserDefaults.standard.removeObject(forKey: "JANGStudioSettings")
        for k in Self.allLeafKeys { UserDefaults.standard.removeObject(forKey: k) }
    }

    override func tearDown() {
        UserDefaults.standard.removeObject(forKey: "JANGStudioSettings")
        for k in Self.allLeafKeys { UserDefaults.standard.removeObject(forKey: k) }
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

    // MARK: - Iter 11: M62 env passthrough mirroring

    func test_tick_throttle_mirrors_only_non_default() {
        let s = AppSettings()
        // Default is 100 ms — mirrored value should be absent (fall through to
        // Python default) so the env var is never set for users who never
        // touched the slider.
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 0,
                       "default 100 ms → leaf key absent (0 from .integer() means absent)")

        s.tickThrottleMs = 250
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 250)

        // Returning to default must REMOVE the key.
        s.tickThrottleMs = 100
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 0,
                       "returning to default must remove the mirror key")
    }

    func test_mlx_thread_count_mirrors_only_non_zero() {
        let s = AppSettings()
        s.persist()  // defaults: mlxThreadCount=0 (auto)
        XCTAssertNil(UserDefaults.standard.object(forKey: BundleResolver.mlxThreadCountDefaultsKey),
                     "0 means auto → leaf key must be absent")

        s.mlxThreadCount = 8
        s.persist()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.mlxThreadCountDefaultsKey), 8)

        s.mlxThreadCount = 0
        s.persist()
        XCTAssertNil(UserDefaults.standard.object(forKey: BundleResolver.mlxThreadCountDefaultsKey),
                     "returning to auto must remove the mirror key")
    }

    func test_custom_jang_tools_path_mirrors_only_non_empty() {
        let s = AppSettings()
        s.customJangToolsPath = "/Users/eric/custom"
        s.persist()
        XCTAssertEqual(UserDefaults.standard.string(forKey: BundleResolver.customJangToolsPathDefaultsKey),
                       "/Users/eric/custom")

        s.customJangToolsPath = ""
        s.persist()
        XCTAssertNil(UserDefaults.standard.string(forKey: BundleResolver.customJangToolsPathDefaultsKey))
    }

    // MARK: - Iter 25: M48 — defaultHFOrg persists + seeds Publish sheet

    func test_default_hf_org_default_is_empty() {
        let s = AppSettings()
        XCTAssertEqual(s.defaultHFOrg, "",
                       "default must be empty so we don't prefix a wrong org on users who haven't configured it")
    }

    func test_default_hf_org_persists_across_process() {
        let s1 = AppSettings()
        s1.defaultHFOrg = "dealignai"
        s1.persist()
        let s2 = AppSettings()
        XCTAssertEqual(s2.defaultHFOrg, "dealignai")
    }

    func test_reset_clears_default_hf_org() {
        let s = AppSettings()
        s.defaultHFOrg = "dealignai"
        s.reset()
        XCTAssertEqual(s.defaultHFOrg, "")
    }

    func test_pre_iter25_snapshot_defaults_hf_org_to_empty() throws {
        // Older snapshots persisted before iter 25 won't have a defaultHFOrg
        // field. The JSON decoder must still accept them (field has a default)
        // and apply the empty string. Without the Snapshot default, old
        // UserDefaults would fail to decode after an app update.
        let oldSnapshot = """
        {
            "defaultOutputParentPath": "",
            "defaultProfile": "JANG_4K",
            "defaultFamily": "jang",
            "defaultMethod": "mse",
            "defaultHadamardEnabled": false,
            "defaultCalibrationSamples": 256,
            "outputNamingTemplate": "{basename}-{profile}",
            "autoDeletePartialOnCancel": false,
            "revealInFinderOnFinish": true,
            "pythonOverridePath": "",
            "customJangToolsPath": "",
            "logVerbosity": "normal",
            "jsonlLogRetentionLines": 10000,
            "logFileOutputDir": "",
            "tickThrottleMs": 100,
            "maxBundleSizeWarningMb": 450,
            "mlxThreadCount": 0,
            "metalPipelineCacheEnabled": true,
            "preAllocateRam": false,
            "preAllocateRamGb": 4,
            "convertConcurrency": 1,
            "copyDiagnosticsAlwaysVisible": true,
            "anonymizePathsInDiagnostics": false,
            "githubIssuesUrl": "https://github.com/jjang-ai/jangq/issues",
            "autoOpenIssueTrackerOnCrash": false,
            "updateChannel": "stable",
            "autoCheckForUpdates": true
        }
        """.data(using: .utf8)!
        UserDefaults.standard.set(oldSnapshot, forKey: "JANGStudioSettings")
        // New AppSettings() calls load() which decodes the old snapshot.
        let s = AppSettings()
        // Other fields should round-trip unchanged.
        XCTAssertEqual(s.defaultProfile, "JANG_4K")
        // New field defaults to empty since the old snapshot didn't carry it.
        XCTAssertEqual(s.defaultHFOrg, "",
                       "pre-iter-25 UserDefaults snapshot must not fail to decode; defaultHFOrg should default to empty")
    }

    func test_load_resyncs_env_passthrough_keys_on_fresh_process() {
        let s1 = AppSettings()
        s1.tickThrottleMs = 200
        s1.mlxThreadCount = 4
        s1.customJangToolsPath = "/prior/session"
        s1.persist()

        // Simulate drift
        UserDefaults.standard.set(999, forKey: BundleResolver.tickThrottleMsDefaultsKey)
        UserDefaults.standard.set(999, forKey: BundleResolver.mlxThreadCountDefaultsKey)
        UserDefaults.standard.set("/wrong", forKey: BundleResolver.customJangToolsPathDefaultsKey)

        // Fresh AppSettings() → load() → mirrorLeafConsumerKeys() re-syncs.
        _ = AppSettings()
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.tickThrottleMsDefaultsKey), 200)
        XCTAssertEqual(UserDefaults.standard.integer(forKey: BundleResolver.mlxThreadCountDefaultsKey), 4)
        XCTAssertEqual(UserDefaults.standard.string(forKey: BundleResolver.customJangToolsPathDefaultsKey),
                       "/prior/session")
    }
}
