import Foundation

/// A loaded JANG v2 hot core — all tensors that stay resident in RAM:
/// embeddings, lm_head, attention q/k/v/o, routers, shared experts, norms.
public struct HotCoreTensors: Sendable {
    public let quantized: [String: QuantizedTensorView]   // base name -> handle
    public let raw: [String: RawTensorView]               // tensor name -> handle
}

/// Load `target/hot_core.safetensors` from a bundle.
///
/// The loader groups every `{base}.weight` + `{base}.scales` + `{base}.biases`
/// triple into a `QuantizedTensorView` and classifies everything else as a
/// `RawTensorView` (norms, biases without a scales partner, etc). The `File`
/// instance is held by the returned views via the `Data` backing of the
/// mmap; the file stays alive for as long as any view retains a slice of it.
public enum HotCoreLoader {
    public static func load(bundle: JangSpecBundle, groupSize: Int = 64) throws -> HotCoreTensors {
        let fileURL = bundle.hotCoreURL
        let file = try SafetensorsV2File(url: fileURL)

        var quantized: [String: QuantizedTensorView] = [:]
        var raw: [String: RawTensorView] = [:]

        // Group tensors by base name.
        // A base name is everything before a trailing .weight/.scales/.biases.
        var byBase: [String: [String]] = [:]   // base -> full names
        var unrelated: [String] = []

        for name in file.tensorNames {
            if let (base, _) = Self.splitSuffix(name) {
                byBase[base, default: []].append(name)
            } else {
                unrelated.append(name)
            }
        }

        // Group with all three (weight, scales, biases) -> quantized.
        // Otherwise -> raw tensors, one entry per tensor.
        for (base, names) in byBase {
            let set = Set(names.compactMap { Self.splitSuffix($0)?.1 })
            if set.contains("weight") && set.contains("scales") && set.contains("biases") {
                let qInfo = try file.info(for: "\(base).weight")
                let sInfo = try file.info(for: "\(base).scales")
                let bInfo = try file.info(for: "\(base).biases")
                _ = bInfo

                guard let bitInfo = BitInference.infer(
                    qweightShape: qInfo.shape,
                    scalesShape: sInfo.shape,
                    groupSize: groupSize
                ) else {
                    throw JangSpecError.invalidManifest(
                        "cannot infer bits for '\(base)' (qweight=\(qInfo.shape), scales=\(sInfo.shape))"
                    )
                }

                quantized[base] = QuantizedTensorView(
                    baseName: base,
                    bits: bitInfo.bits,
                    groupSize: groupSize,
                    inFeatures: bitInfo.inFeatures,
                    outFeatures: bitInfo.outFeatures,
                    qweightShape: qInfo.shape,
                    scalesShape: sInfo.shape,
                    qweight: try file.bytes(for: "\(base).weight"),
                    scales: try file.bytes(for: "\(base).scales"),
                    biases: try file.bytes(for: "\(base).biases")
                )
            } else {
                // Incomplete triple — emit each tensor as a raw view.
                for n in names {
                    let info = try file.info(for: n)
                    raw[n] = RawTensorView(
                        name: n,
                        dtype: info.dtype,
                        shape: info.shape,
                        bytes: try file.bytes(for: n)
                    )
                }
            }
        }

        // Unrelated (no suffix) tensors are raw.
        for n in unrelated {
            let info = try file.info(for: n)
            raw[n] = RawTensorView(
                name: n,
                dtype: info.dtype,
                shape: info.shape,
                bytes: try file.bytes(for: n)
            )
        }

        return HotCoreTensors(quantized: quantized, raw: raw)
    }

    /// Split a name like "layers.0.self_attn.q_proj.scales" into
    /// ("layers.0.self_attn.q_proj", "scales"). Returns nil if there's no
    /// recognized trailing suffix.
    private static func splitSuffix(_ name: String) -> (String, String)? {
        for s in ["weight", "scales", "biases"] {
            let dot = ".\(s)"
            if name.hasSuffix(dot) {
                return (String(name.dropLast(dot.count)), s)
            }
        }
        return nil
    }
}
