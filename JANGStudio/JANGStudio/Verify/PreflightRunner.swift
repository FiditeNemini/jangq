// JANGStudio/JANGStudio/Verify/PreflightRunner.swift
import Foundation

private nonisolated(unsafe) let KNOWN_512_EXPERT_TYPES: Set<String> = ["minimax_m2", "glm_moe_dsa"]

struct PreflightRunner {
    func run(plan: ConversionPlan) -> [PreflightCheck] {
        var out: [PreflightCheck] = []
        let src = plan.sourceURL
        let dst = plan.outputURL

        out.append(Self.sourceReadable(src))
        out.append(Self.configValid(src))
        out.append(Self.outputUsable(src: src, dst: dst))
        out.append(Self.diskSpace(dst: dst, estimated: Self.sizeEstimate(plan)))
        out.append(Self.ramAdequate(plan: plan))
        out.append(Self.jangtqArchSupported(plan: plan))
        out.append(Self.jangtqSourceDtype(plan: plan))
        out.append(Self.bf16For512Experts(plan: plan))
        out.append(Self.hadamardVsLowBits(plan: plan))
        out.append(Self.bundledPythonHealthy())
        return out
    }

    private static func sourceReadable(_ url: URL?) -> PreflightCheck {
        guard let url else { return .init(id: .sourceReadable, title: "Source dir exists", status: .fail, hint: "No source selected") }
        let ok = FileManager.default.isReadableFile(atPath: url.path)
        return .init(id: .sourceReadable, title: "Source dir exists",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "\(url.path) is not readable")
    }

    private static func configValid(_ url: URL?) -> PreflightCheck {
        guard let url else { return .init(id: .configJSONValid, title: "config.json parses", status: .fail, hint: nil) }
        let cfg = url.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (obj["model_type"] as? String) != nil || ((obj["text_config"] as? [String: Any])?["model_type"] as? String) != nil
        else {
            return .init(id: .configJSONValid, title: "config.json parses", status: .fail,
                         hint: "config.json missing or no model_type")
        }
        return .init(id: .configJSONValid, title: "config.json parses", status: .pass, hint: nil)
    }

    private static func outputUsable(src: URL?, dst: URL?) -> PreflightCheck {
        guard let dst else { return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Pick an output folder") }
        if dst == src { return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Output cannot equal source") }
        if dst.path.contains(".app/Contents") {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Do not write inside an .app")
        }
        let parent = dst.deletingLastPathComponent()
        if !FileManager.default.isWritableFile(atPath: parent.path) {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Parent not writable")
        }
        return .init(id: .outputUsable, title: "Output dir valid", status: .pass, hint: nil)
    }

    private static func sizeEstimate(_ plan: ConversionPlan) -> Int64 {
        guard let src = plan.detected?.totalBytes else { return 0 }
        let bitsPerWeight: Double = switch plan.profile {
            case "JANG_1L", "JANG_2S", "JANG_2M", "JANG_2L", "JANGTQ2": 2.5
            case "JANG_3K", "JANG_3S", "JANG_3M", "JANG_3L", "JANGTQ3": 3.5
            case "JANG_4K", "JANG_4S", "JANG_4M", "JANG_4L", "JANGTQ4": 4.5
            case "JANG_5K": 5.5
            case "JANG_6K", "JANG_6M": 6.5
            default: 4.5
        }
        return Int64(Double(src) * (bitsPerWeight / 16.0) * 1.10)
    }

    private static func diskSpace(dst: URL?, estimated: Int64) -> PreflightCheck {
        guard let dst else { return .init(id: .diskSpace, title: "Free disk space", status: .fail, hint: nil) }
        let parent = dst.deletingLastPathComponent()
        let rv = try? parent.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let free = Int64(rv?.volumeAvailableCapacityForImportantUsage ?? 0)
        if estimated <= 0 {
            return .init(id: .diskSpace, title: "Free disk space", status: .pass, hint: "\(free / 1_000_000_000) GB free")
        }
        let ok = free >= estimated
        return .init(id: .diskSpace, title: "Free disk space",
                     status: ok ? .pass : .fail,
                     hint: ok ? "\(free / 1_000_000_000) GB free" : "Need ~\(estimated / 1_000_000_000) GB, have \(free / 1_000_000_000) GB")
    }

    private static func ramAdequate(plan: ConversionPlan) -> PreflightCheck {
        let ram = Int64(ProcessInfo.processInfo.physicalMemory)
        guard let srcBytes = plan.detected?.totalBytes, srcBytes > 0 else {
            return .init(id: .ramAdequate, title: "RAM adequate", status: .pass, hint: nil)
        }
        let needed = Int64(Double(srcBytes) * 1.5)
        let ok = ram >= needed
        return .init(id: .ramAdequate, title: "RAM adequate",
                     status: ok ? .pass : .warn,
                     hint: ok ? nil : "~\(needed / 1_000_000_000) GB needed; you have \(ram / 1_000_000_000) GB. Conversion may swap or OOM.")
    }

    private static func jangtqArchSupported(plan: ConversionPlan) -> PreflightCheck {
        if plan.family != .jangtq { return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported", status: .pass, hint: nil) }
        let mt = plan.detected?.modelType ?? ""
        let ok = JANGTQ_V1_WHITELIST.contains(mt)
        return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ v1 supports Qwen 3.6 and MiniMax only; detected \(mt)")
    }

    private static func jangtqSourceDtype(plan: ConversionPlan) -> PreflightCheck {
        if plan.family != .jangtq { return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype", status: .pass, hint: nil) }
        let d = plan.detected?.dtype ?? .unknown
        let ok = (d == .bf16 || d == .fp8)
        return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ expects BF16 or FP8 source; detected \(d.rawValue)")
    }

    private static func bf16For512Experts(plan: ConversionPlan) -> PreflightCheck {
        let mt = plan.detected?.modelType ?? ""
        guard KNOWN_512_EXPERT_TYPES.contains(mt) else { return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .pass, hint: nil) }
        if plan.overrides.forceDtype == .fp16 {
            return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .warn,
                         hint: "\(mt) has 512+ experts — bfloat16 strongly recommended over float16 to avoid overflow")
        }
        return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .pass, hint: nil)
    }

    private static func hadamardVsLowBits(plan: ConversionPlan) -> PreflightCheck {
        let is2bit = plan.profile.contains("_2") || plan.profile == "JANG_1L" || plan.profile == "JANGTQ2"
        if plan.hadamard && is2bit {
            return .init(id: .hadamardVsLowBits, title: "Hadamard rotation sanity", status: .warn,
                         hint: "Hadamard rotation hurts quality at 2-bit. Turn off for JANG_2*/JANG_1L/JANGTQ2.")
        }
        return .init(id: .hadamardVsLowBits, title: "Hadamard rotation sanity", status: .pass, hint: nil)
    }

    private static func bundledPythonHealthy() -> PreflightCheck {
        let ok = BundleResolver.healthCheck()
        return .init(id: .bundledPythonHealthy, title: "Bundled Python runtime healthy",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "Bundled python3 missing — reinstall JANG Studio")
    }
}
