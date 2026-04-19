// JANGStudio/Tests/JANGStudioTests/BundleResolverTests.swift
import XCTest
@testable import JANGStudio

final class BundleResolverTests: XCTestCase {
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
}
