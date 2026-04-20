// JANGStudio/Tests/JANGStudioTests/BundleResolverTests.swift
import XCTest
@testable import JANGStudio

final class BundleResolverTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Clean slate for iter-11 leaf keys.
        for key in [
            BundleResolver.tickThrottleMsDefaultsKey,
            BundleResolver.mlxThreadCountDefaultsKey,
            BundleResolver.customJangToolsPathDefaultsKey,
        ] {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    override func tearDown() {
        for key in [
            BundleResolver.tickThrottleMsDefaultsKey,
            BundleResolver.mlxThreadCountDefaultsKey,
            BundleResolver.customJangToolsPathDefaultsKey,
        ] {
            UserDefaults.standard.removeObject(forKey: key)
        }
        super.tearDown()
    }

    func test_pythonExecutablePath_isUnderBundleResources() {
        let url = BundleResolver.pythonExecutable
        XCTAssertTrue(url.path.hasSuffix("Contents/Resources/python/bin/python3"), url.path)
    }

    func test_debugOverrideFromEnvironment() {
        setenv("JANGSTUDIO_PYTHON_OVERRIDE", "/opt/homebrew/bin/python3", 1)
        defer { unsetenv("JANGSTUDIO_PYTHON_OVERRIDE") }
        let url = BundleResolver.pythonExecutable
        XCTAssertEqual(url.path, "/opt/homebrew/bin/python3")
    }

    // MARK: - Iter 11: M62 env passthrough

    func test_childProcessEnvAdditions_empty_when_no_overrides() {
        let additions = BundleResolver.childProcessEnvAdditions(inherited: [:])
        XCTAssertTrue(additions.isEmpty, "no settings set → no env additions")
    }

    func test_childProcessEnvAdditions_wires_tick_throttle() {
        UserDefaults.standard.set(250, forKey: BundleResolver.tickThrottleMsDefaultsKey)
        let additions = BundleResolver.childProcessEnvAdditions(inherited: [:])
        XCTAssertEqual(additions["JANG_TICK_THROTTLE_MS"], "250")
    }

    func test_childProcessEnvAdditions_wires_thread_count_both_keys() {
        UserDefaults.standard.set(8, forKey: BundleResolver.mlxThreadCountDefaultsKey)
        let additions = BundleResolver.childProcessEnvAdditions(inherited: [:])
        // Both OMP_NUM_THREADS (BLAS layer) and MLX_NUM_THREADS (MLX layer)
        // should be set so the user doesn't have to know which layer consumes which.
        XCTAssertEqual(additions["OMP_NUM_THREADS"], "8")
        XCTAssertEqual(additions["MLX_NUM_THREADS"], "8")
    }

    func test_childProcessEnvAdditions_zero_thread_count_no_env_var() {
        // 0 means "auto" in the UI; must NOT wedge the child with OMP_NUM_THREADS=0.
        UserDefaults.standard.set(0, forKey: BundleResolver.mlxThreadCountDefaultsKey)
        let additions = BundleResolver.childProcessEnvAdditions(inherited: [:])
        XCTAssertNil(additions["OMP_NUM_THREADS"])
        XCTAssertNil(additions["MLX_NUM_THREADS"])
    }

    func test_childProcessEnvAdditions_prepends_custom_python_path() {
        UserDefaults.standard.set("/Users/eric/custom-jang-tools",
                                  forKey: BundleResolver.customJangToolsPathDefaultsKey)
        let additions = BundleResolver.childProcessEnvAdditions(
            inherited: ["PYTHONPATH": "/existing/path:/another"])
        XCTAssertEqual(additions["PYTHONPATH"],
                       "/Users/eric/custom-jang-tools:/existing/path:/another",
                       "custom path must be PREPENDED so the bundled jang_tools still resolves for things the custom path doesn't override")
    }

    func test_childProcessEnvAdditions_custom_python_path_empty_inherited_pythonpath() {
        UserDefaults.standard.set("/Users/eric/custom",
                                  forKey: BundleResolver.customJangToolsPathDefaultsKey)
        let additions = BundleResolver.childProcessEnvAdditions(inherited: [:])
        XCTAssertEqual(additions["PYTHONPATH"], "/Users/eric/custom")
    }
}
