// Minimal example: load a JANG model in Swift via JANGKit.
//
// Requirements:
//   - macOS 15+ (Apple Silicon)
//   - Add JANGRuntime to your SwiftPM dependencies (see Package.swift block below)
//   - Link the JANGKit product.
//
// Usage (after `swift build`):
//   .build/debug/MyJANGApp /path/to/model "Your prompt here"
//
// Author: Jinho Jang (eric@jangq.ai)

import Foundation
import JANGKit

@main
struct JANGExample {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 3 else {
            fputs("Usage: \(args[0]) <model_dir> <prompt>\n", stderr)
            exit(1)
        }

        let modelURL = URL(fileURLWithPath: args[1])
        let prompt = args[2]

        do {
            // JANGKit.Model.load auto-detects JANG vs JANGTQ via jang_config.json.
            // JANGTQ models will throw .jangtqNotYetSupported — use JANGTQGenerator
            // directly from the JANG module for those (see PORTING.md).
            let model = try await JANGKit.Model.load(at: modelURL)

            let result = try await model.generate(
                prompt: prompt,
                config: JANGKit.SamplingConfig(temperature: 0.0, maxTokens: 200)
            )

            print(result.text)
            fputs(
                "tokens=\(result.tokens) "
                + "elapsed=\(String(format: "%.2f", result.elapsedSeconds))s "
                + "throughput=\(String(format: "%.1f", result.tokensPerSecond))tok/s "
                + "finish=\(result.finishReason.rawValue)\n",
                stderr
            )

        } catch let error as JANGKit.ModelError {
            fputs("JANGKit error: \(error.localizedDescription ?? "\(error)")\n", stderr)
            exit(2)
        } catch {
            fputs("Unexpected error: \(error)\n", stderr)
            exit(3)
        }
    }
}

// MARK: - Chat (Qwen-family models)
//
// For Qwen im_start/im_end models you can use model.chat() instead:
//
//   let result = try await model.chat(
//       system: "You are a helpful assistant.",
//       user: prompt,
//       config: JANGKit.SamplingConfig(maxTokens: 200)
//   )
//
// NOTE: chat() uses JANGTokenizer.encodeChatPrompt(), which implements the Qwen
// <|im_start|>/<|im_end|> template only. For LLaMA / Mistral / other chat families,
// format your prompt manually (including special tokens) and call generate() directly.

// MARK: - Package.swift for adopters
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
//                 .product(name: "JANGKit", package: "jangq"),
//             ]
//         )
//     ]
// )
//
// Power users who need direct access to MXQModel / JANGInferenceEngine /
// JANGTokenizer / JANGSampler can use the `JANG` product instead of JANGKit.
