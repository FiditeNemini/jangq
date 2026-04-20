// JANGStudio/JANGStudio/Verify/PreflightRunner.swift
import Foundation

struct PreflightRunner {
    func run(plan: ConversionPlan,
             capabilities: Capabilities = .frozen,
             profiles: Profiles = .frozen) -> [PreflightCheck] {
        var out: [PreflightCheck] = []
        let src = plan.sourceURL
        let dst = plan.outputURL

        out.append(Self.sourceReadable(src))
        out.append(Self.configValid(src))
        out.append(Self.outputUsable(src: src, dst: dst))
        // M141 (iter 63): diskSpace was being called with `estimated: 0`,
        // which makes the function short-circuit to `.pass` unconditionally.
        // The gate was inert. Compute a profile-aware estimate from the
        // source bytes × (avgBits / 16) × 1.05 metadata overhead — same
        // formula as `estimate_model.predict` on the Python side, keeping
        // the two size-estimates aligned across the Swift⇄Python boundary
        // (M140 meta-lesson about cross-boundary decision-overlap).
        let estimated = Self.estimateOutputBytes(plan: plan, profiles: profiles)
        out.append(Self.diskSpace(dst: dst, estimated: estimated))
        out.append(Self.ramAdequate(plan: plan))
        out.append(Self.jangtqArchSupported(plan: plan, whitelist: capabilities.jangtqWhitelist))
        out.append(Self.jangtqSourceDtype(plan: plan))
        out.append(Self.bf16For512Experts(plan: plan, types: capabilities.knownExpert512Types))
        out.append(Self.hadamardVsLowBits(plan: plan))
        out.append(Self.bundledPythonHealthy())
        return out
    }

    /// M141 (iter 63): profile-aware output-size estimator for the
    /// preflight disk-space gate. Returns 0 when the source hasn't been
    /// inspected yet (preserves the pre-iter-63 pass-through behavior on
    /// the initial empty-plan render).
    ///
    /// Formula mirrors `jang_tools/estimate_model.predict` so the preflight
    /// warning, the wizard's predicted-size banner, and the Python-side
    /// downstream are all in agreement. Assumes source is BF16 (16 bits /
    /// weight) — for an FP8 source the real output will be slightly
    /// smaller than this estimate, but conservative-over predicts are OK
    /// (the disk-space gate is an INEQUALITY: "have at least N free").
    static func estimateOutputBytes(plan: ConversionPlan, profiles: Profiles) -> Int64 {
        guard let srcBytes = plan.detected?.totalBytes, srcBytes > 0 else { return 0 }
        let avgBits = avgBitsForProfile(plan.profile, profiles: profiles)
        guard avgBits > 0 else { return 0 }
        return Int64(Double(srcBytes) * (avgBits / 16.0) * 1.05)
    }

    /// Look up the avg bits/weight for a profile from either JANG or JANGTQ
    /// tables. Returns 0 on unknown profile (caller falls back to pass).
    static func avgBitsForProfile(_ profile: String, profiles: Profiles) -> Double {
        if let p = profiles.jang.first(where: { $0.name == profile }) { return p.avgBits }
        if let p = profiles.jangtq.first(where: { $0.name == profile }) { return Double(p.bits) }
        return 0
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
        // M139 (iter 61): reject nested src/dst. If output lives INSIDE the
        // source tree (or source inside output), the two directories share
        // safetensors shards in the same subtree. Recursive greps / future
        // cleanup passes could touch the wrong set. Also confuses users who
        // later `rm -rf source/` and discover their output went with it.
        // The plain-equal check above doesn't cover this case because the
        // paths differ — one is a strict prefix of the other.
        if let s = src {
            let srcPath = s.standardizedFileURL.path
            let dstPath = dst.standardizedFileURL.path
            // Use path + "/" to prevent sibling-prefix matches
            // (e.g. `/a/b` is NOT inside `/a/bc`).
            if dstPath.hasPrefix(srcPath + "/") {
                return .init(id: .outputUsable, title: "Output dir valid", status: .fail,
                             hint: "Output cannot be inside the source folder")
            }
            if srcPath.hasPrefix(dstPath + "/") {
                return .init(id: .outputUsable, title: "Output dir valid", status: .fail,
                             hint: "Source cannot be inside the output folder")
            }
        }
        if dst.path.contains(".app/Contents") {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Do not write inside an .app")
        }
        let parent = dst.deletingLastPathComponent()
        if !FileManager.default.isWritableFile(atPath: parent.path) {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Parent not writable")
        }
        return .init(id: .outputUsable, title: "Output dir valid", status: .pass, hint: nil)
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

    private static func jangtqArchSupported(plan: ConversionPlan, whitelist: [String]) -> PreflightCheck {
        if plan.family != .jangtq {
            return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported", status: .pass, hint: nil)
        }
        let mt = plan.detected?.modelType ?? ""
        let ok = whitelist.contains(mt)
        return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ supports \(whitelist.joined(separator: ", ")); detected \(mt)")
    }

    private static func jangtqSourceDtype(plan: ConversionPlan) -> PreflightCheck {
        if plan.family != .jangtq { return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype", status: .pass, hint: nil) }
        let d = plan.detected?.dtype ?? .unknown
        let ok = (d == .bf16 || d == .fp8)
        return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ expects BF16 or FP8 source; detected \(d.rawValue)")
    }

    private static func bf16For512Experts(plan: ConversionPlan, types: [String]) -> PreflightCheck {
        let mt = plan.detected?.modelType ?? ""
        // M140 (iter 62): preflight-side of the M131 fix. Iter 53 made
        // `_recommend_dtype` (Python) dynamically promote any MoE with
        // `expert_count >= 512` to bfloat16 instead of relying on a
        // hardcoded `{minimax_m2, glm_moe_dsa}` name list. The preflight
        // check here had the same decision-overlap bug — it only flagged
        // types in `knownExpert512Types` (a hardcoded list in
        // capabilities_cli.py). A future 512+ expert family (e.g., a
        // future Qwen / DeepSeek variant) would skip this warning despite
        // needing bfloat16 for the same float16-overflow reason.
        //
        // Fix: check BOTH the named-family list AND the dynamic expert
        // count. Mirrors the Python-side recommend.py fix exactly.
        let dynamic512 = (plan.detected?.numExperts ?? 0) >= 512
        guard types.contains(mt) || dynamic512 else {
            return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .pass, hint: nil)
        }
        if plan.overrides.forceDtype == .fp16 {
            let expertStr = dynamic512 ? "\(plan.detected?.numExperts ?? 0) experts" : mt
            return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .warn,
                         hint: "\(expertStr) — bfloat16 strongly recommended over float16 to avoid overflow")
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
