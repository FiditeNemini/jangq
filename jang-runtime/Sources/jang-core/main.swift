import ArgumentParser
import Foundation
import JANGCore

@main
struct JangCore: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "jang-core",
        abstract: "Load and inspect .jangspec bundles in pure Swift.",
        version: JANGCore.version,
        subcommands: [Inspect.self, HotCore.self]
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

struct HotCore: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "hot-core",
        abstract: "Print a per-tensor summary of a bundle's hot core."
    )

    @Argument(help: "Path to a .jangspec directory.")
    var bundle: String

    @Option(name: .long, help: "Quant group size (default 64).")
    var groupSize: Int = 64

    func run() async throws {
        let url = URL(fileURLWithPath: (bundle as NSString).expandingTildeInPath).resolvingSymlinksInPath()
        let spec = try JangSpecBundle.open(at: url)
        let hot = try HotCoreLoader.load(bundle: spec, groupSize: groupSize)

        print("  bundle:     \(url.path)")
        print("  quantized:  \(hot.quantized.count) base tensors")
        print("  raw:        \(hot.raw.count) tensors")

        // Histogram of bit widths.
        var bitCounts: [Int: Int] = [:]
        var bitBytes: [Int: Int] = [:]
        for q in hot.quantized.values {
            bitCounts[q.bits, default: 0] += 1
            bitBytes[q.bits, default: 0] += q.qweight.count + q.scales.count + q.biases.count
        }
        print("")
        print("  bit distribution:")
        for b in bitCounts.keys.sorted() {
            let cnt = bitCounts[b]!
            let gb = Double(bitBytes[b]!) / 1e9
            print(String(format: "    %d-bit: %5d tensors, %.2f GB", b, cnt, gb))
        }

        // Totals.
        let totalQBytes = hot.quantized.values.reduce(0) { $0 + $1.qweight.count + $1.scales.count + $1.biases.count }
        let totalRBytes = hot.raw.values.reduce(0) { $0 + $1.bytes.count }
        print("")
        print(String(format: "  total quantized: %.2f GB", Double(totalQBytes) / 1e9))
        print(String(format: "  total raw:       %.2f GB", Double(totalRBytes) / 1e9))
        print(String(format: "  total hot core:  %.2f GB", Double(totalQBytes + totalRBytes) / 1e9))
    }
}
