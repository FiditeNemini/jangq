// JANGStudio/JANGStudio/App/JANGStudioApp.swift
import SwiftUI

@main
struct JANGStudioApp: App {
    var body: some Scene {
        WindowGroup("JANG Studio") {
            WizardView()
                .frame(minWidth: 960, minHeight: 640)
        }
        .windowResizability(.contentSize)
    }
}
