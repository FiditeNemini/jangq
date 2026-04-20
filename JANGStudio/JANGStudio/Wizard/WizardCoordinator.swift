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
    @Environment(AppSettings.self) private var settings
    @State private var defaultsApplied = false

    var body: some View {
        NavigationSplitView {
            List(WizardStep.allCases, selection: Binding(
                get: { coord.active },
                // M176 (iter 109): gate sidebar navigation on canActivate.
                // Pre-M176 the set: binding accepted any click — user who
                // hadn't completed Source could click Architecture and land
                // in a dead-end (Continue button disabled, no signal why).
                // iter-81 flagged the mixed signal: visual lock icon +
                // `.secondary` foreground suggested "locked" but behavior
                // didn't match. Now we reject unreachable jumps by ignoring
                // the new value and keeping current `active`. User still
                // sees the click register (SwiftUI highlights the row
                // momentarily) but navigation doesn't happen for locked
                // steps. Forward navigation through the Continue button
                // path is unaffected.
                set: { newValue in
                    guard let step = newValue else { return }
                    if coord.canActivate(step) {
                        coord.active = step
                    }
                    // else: ignore — sidebar click on a locked step is a no-op.
                }
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
        .task {
            // Apply settings-configured defaults once on first wizard entry.
            // After reset() (VerifyStep → Convert another), we re-apply there.
            guard !defaultsApplied else { return }
            coord.plan.applyDefaults(from: settings)
            defaultsApplied = true
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
