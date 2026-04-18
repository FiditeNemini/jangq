// JANGStudio/JANGStudio/App/JANGStudioApp.swift
import SwiftUI

@main
struct JANGStudioApp: App {
    var body: some Scene {
        WindowGroup("JANG Studio") {
            Text("JANG Studio — scaffolding")
                .frame(minWidth: 800, minHeight: 600)
        }
        .windowResizability(.contentSize)
    }
}
