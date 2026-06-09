// JANGStudio/Tests/JANGStudioUITests/WizardFlowTests.swift
import XCTest

final class WizardFlowTests: XCTestCase {
    @MainActor
    func test_sidebarListsFiveSteps() {
        let app = XCUIApplication()
        app.launchEnvironment["JANGSTUDIO_PYTHON_OVERRIDE"] =
            Bundle(for: Self.self).path(forResource: "fake_convert", ofType: "sh")!
        app.launch()
        let sourceExists = app.staticTexts["1 · Source Model"].exists
        let architectureExists = app.staticTexts["2 · Architecture"].exists
        let profileExists = app.staticTexts["3 · Profile"].exists
        let runExists = app.staticTexts["4 · Run"].exists
        let verifyExists = app.staticTexts["5 · Verify & Finish"].exists
        XCTAssertTrue(sourceExists)
        XCTAssertTrue(architectureExists)
        XCTAssertTrue(profileExists)
        XCTAssertTrue(runExists)
        XCTAssertTrue(verifyExists)
    }
}
