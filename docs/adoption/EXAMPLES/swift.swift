// Minimal example: load a JANG model directory in Swift via JANGCore + JANGCoreMetal.
//
// Requirements:
//   - macOS 15+ (Apple Silicon)
//   - The JANGRuntime Swift package from https://github.com/jjang-ai/jangq
//     added to your project via SwiftPM.
//
// Status (as of v1):
//   JANGCore handles the on-disk format: manifest parsing, expert index,
//   and tensor loading from .safetensors shards.
//   Full text-generation inference is provided via JANGCoreMetal + JANG.
//   See Package.swift for the product structure.
//
// Author: Jinho Jang (eric@jangq.ai)

import Foundation
import JANGCore

// MARK: - Load a JANG model bundle

func loadJANGModel(at modelURL: URL) throws {
    // JANGCore parses jang_config.json + model.safetensors.index.json
    // and exposes the shard manifest and expert index.
    let bundle = try JangSpecBundle(directory: modelURL)

    print("Format version: \(bundle.manifest.formatVersion)")
    print("Profile: \(bundle.manifest.profile ?? "unknown")")
    print("Source model: \(bundle.manifest.sourceName ?? "unknown")")

    // The manifest exposes a list of shard files; each is a standard
    // safetensors v2 file you can memory-map directly.
    for shard in bundle.manifest.shards {
        print("Shard: \(shard.filename) — \(shard.tensorCount) tensors")
    }
}

// MARK: - Package.swift reference for SwiftPM adopters
//
// Add JANGRuntime as a dependency in your Package.swift:
//
// // swift-tools-version: 6.0
// import PackageDescription
//
// let package = Package(
//     name: "MyJANGApp",
//     platforms: [.macOS(.v15)],
//     dependencies: [
//         .package(url: "https://github.com/jjang-ai/jangq", branch: "main")
//     ],
//     targets: [
//         .executableTarget(
//             name: "MyJANGApp",
//             dependencies: [
//                 .product(name: "JANGCore", package: "jangq"),
//                 .product(name: "JANGCoreMetal", package: "jangq"),
//             ]
//         )
//     ]
// )

// MARK: - Entry point

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("Usage: \(args[0]) <model_dir>")
    exit(1)
}

let modelURL = URL(fileURLWithPath: args[1])

do {
    try loadJANGModel(at: modelURL)
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(2)
}
