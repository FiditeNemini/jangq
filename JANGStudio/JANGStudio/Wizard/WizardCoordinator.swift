// JANGStudio/JANGStudio/Wizard/WizardCoordinator.swift
import SwiftUI

enum WizardStep: Int, CaseIterable, Identifiable {
    case source = 1, architecture, profile, run, verify
    var id: Int { rawValue }
    var title: String {
        switch self {
        case .source:       "1 · Source Model"
        case .architecture: "2 · Architecture"
        case .profile:      "3 · Profile"
        case .run:          "4 · Run"
        case .verify:       "5 · Verify & Finish"
        }
    }
}

@Observable
final class WizardCoordinator {
    var plan = ConversionPlan()
    var active: WizardStep = .source

    func canActivate(_ step: WizardStep) -> Bool {
        switch step {
        case .source:       return true
        case .architecture: return plan.isStep1Complete
        case .profile:      return plan.isStep2Complete
        case .run:          return plan.isStep3Complete
        case .verify:       return plan.isStep4Complete
        }
    }
}

struct WizardView: View {
    @State private var coord = WizardCoordinator()

    var body: some View {
        NavigationSplitView {
            List(WizardStep.allCases, selection: Binding(
                get: { coord.active },
                set: { coord.active = $0 ?? .source }
            )) { step in
                HStack {
                    Image(systemName: stepIcon(step))
                    Text(step.title)
                }
                .foregroundStyle(coord.canActivate(step) ? .primary : .secondary)
                .tag(step)
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 220, ideal: 240)
        } detail: {
            switch coord.active {
            case .source:       SourceStep(coord: coord)
            case .architecture: ArchitectureStep(coord: coord)
            case .profile:      ProfileStep(coord: coord)
            case .run:          RunStep(coord: coord)
            case .verify:       VerifyStep(coord: coord)
            }
        }
    }

    private func stepIcon(_ s: WizardStep) -> String {
        if !coord.canActivate(s) { return "lock" }
        if s == coord.active { return "circle.fill" }
        switch s {
        case .source:       return coord.plan.isStep1Complete ? "checkmark.circle.fill" : "circle"
        case .architecture: return coord.plan.isStep2Complete ? "checkmark.circle.fill" : "circle"
        case .profile:      return coord.plan.isStep3Complete ? "checkmark.circle.fill" : "circle"
        case .run:          return coord.plan.isStep4Complete ? "checkmark.circle.fill" : "circle"
        case .verify:       return "flag.checkered"
        }
    }
}
