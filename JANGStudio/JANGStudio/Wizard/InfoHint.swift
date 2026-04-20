// JANGStudio/JANGStudio/Wizard/InfoHint.swift
//
// Small `(i)` button that shows a plain-English explanation popover.
// Used next to form field labels so beginners can learn what each option
// means without leaving the wizard.

import SwiftUI

struct InfoHint: View {
    let text: String
    @State private var showing = false

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Button {
            showing = true
        } label: {
            Image(systemName: "info.circle")
                .foregroundStyle(.secondary)
                .font(.caption)
        }
        .buttonStyle(.borderless)
        .popover(isPresented: $showing, arrowEdge: .top) {
            Text(text)
                .font(.callout)
                .textSelection(.enabled)
                .padding(14)
                .frame(maxWidth: 360, alignment: .leading)
                .fixedSize(horizontal: false, vertical: true)
        }
        .help(text)
    }
}
