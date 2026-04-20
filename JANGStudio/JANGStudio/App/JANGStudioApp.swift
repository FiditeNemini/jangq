// JANGStudio/JANGStudio/App/JANGStudioApp.swift
import SwiftUI

@main
struct JANGStudioApp: App {
    @State private var capabilities = CapabilitiesService()
    @State private var profiles = ProfilesService()

    var body: some Scene {
        WindowGroup("JANG Studio") {
            WizardView()
                .frame(minWidth: 960, minHeight: 640)
                .environment(capabilities)
                .environment(profiles)
                .task {
                    await capabilities.refresh()
                    await profiles.refresh()
                }
        }
        .windowResizability(.contentSize)
    }
}
