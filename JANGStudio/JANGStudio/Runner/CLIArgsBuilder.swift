// JANGStudio/JANGStudio/Runner/CLIArgsBuilder.swift
import Foundation

/// Builds the `python -m ...` argument list for a given ConversionPlan.
/// Pure function — no side effects — so it can be unit tested exhaustively.
enum CLIArgsBuilder {
    /// Returns the argument list to pass to `python3`. Returns [] if sourceURL or
    /// outputURL are missing from the plan.
    static func args(for plan: ConversionPlan) -> [String] {
        guard let src = plan.sourceURL?.path, let out = plan.outputURL?.path else { return [] }
        switch plan.family {
        case .jang:
            var args = ["-m", "jang_tools", "convert", src, "-o", out, "-p", plan.profile,
                        "-m", plan.method.rawValue, "--progress=json", "--quiet-text"]
            if plan.hadamard { args.append("--hadamard") }
            // Advanced overrides — propagated only when explicitly set by the
            // user (Architecture → Advanced overrides). Omitted otherwise so
            // Python's auto-detect stays in charge.
            if let bs = plan.overrides.forceBlockSize, bs > 0 {
                args.append(contentsOf: ["-b", String(bs)])
            }
            if let fd = plan.overrides.forceDtype, let alias = dtypeFlagValue(for: fd) {
                args.append(contentsOf: ["--force-dtype", alias])
            }
            return args
        case .jangtq:
            let mod: String = switch plan.detected?.modelType ?? "" {
                case "qwen3_5_moe": "jang_tools.convert_qwen35_jangtq"
                case "minimax_m2":  "jang_tools.convert_minimax_jangtq"
                default: "jang_tools.convert_qwen35_jangtq"
            }
            return ["-m", mod, "--progress=json", "--quiet-text", src, out, plan.profile]
        }
    }

    /// Map SourceDtype → the alias Python's `jang_tools convert --force-dtype`
    /// accepts. Returns nil for values that don't make sense to force (unknown,
    /// jangV2 — the model is already JANG format, you shouldn't be reconverting).
    private static func dtypeFlagValue(for d: SourceDtype) -> String? {
        switch d {
        case .bf16: return "bf16"
        case .fp16: return "fp16"
        case .fp8: return "fp8"
        case .unknown, .jangV2: return nil
        }
    }
}
