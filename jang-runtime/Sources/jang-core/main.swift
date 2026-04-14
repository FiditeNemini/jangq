import ArgumentParser
import Foundation
import JANGCore

@main
struct JangCore: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "jang-core",
        abstract: "Load and inspect .jangspec bundles in pure Swift.",
        version: JANGCore.version,
        subcommands: [Inspect.self]
    )
}

struct Inspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Print a bundle's manifest summary."
    )

    @Argument(help: "Path to a .jangspec directory.")
    var bundle: String

    func run() async throws {
        let url = URL(fileURLWithPath: (bundle as NSString).expandingTildeInPath).resolvingSymlinksInPath()
        let spec = try JangSpecBundle.open(at: url)
        let m = spec.manifest

        let hotGB = Double(m.hotCoreBytes) / 1e9
        let expGB = Double(m.expertBytes) / 1e9

        print("  bundle:        \(url.path)")
        print("  source jang:   \(m.sourceJang)")
        print("  arch:          \(m.targetArch)")
        print("  n_layers:      \(m.nLayers)")
        print("  experts/layer: \(m.nExpertsPerLayer)")
        print("  top_k:         \(m.targetTopK)")
        print(String(format: "  hot_core:      %.2f GB", hotGB))
        print(String(format: "  expert_bytes:  %.2f GB", expGB))
        print("  draft:         \(m.hasDraft)")
        print("  router_prior:  \(m.hasRouterPrior)")
        print("  bundle_version: \(m.bundleVersion)")
        print("  tool_version:  \(m.toolVersion)")
    }
}
