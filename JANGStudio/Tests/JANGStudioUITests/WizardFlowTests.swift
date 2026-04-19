// JANGStudio/Tests/JANGStudioUITests/WizardFlowTests.swift
import XCTest

final class WizardFlowTests: XCTestCase {
    func test_sidebarListsFiveSteps() {
        let app = XCUIApplication()
        app.launchEnvironment["JANGSTUDIO_PYTHON_OVERRIDE"] =
            Bundle(for: Self.self).path(forResource: "fake_convert", ofType: "sh")!
        app.launch()
        XCTAssertTrue(app.staticTexts["1 · Source Model"].exists)
        XCTAssertTrue(app.staticTexts["2 · Architecture"].exists)
        XCTAssertTrue(app.staticTexts["3 · Profile"].exists)
        XCTAssertTrue(app.staticTexts["4 · Run"].exists)
        XCTAssertTrue(app.staticTexts["5 · Verify & Finish"].exists)
    }
}
