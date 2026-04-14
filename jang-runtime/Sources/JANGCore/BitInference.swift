import Foundation

/// Infers per-tensor JANG v2 bit widths from the qweight+scales shape pair.
public enum BitInference {

    public struct Result: Sendable, Equatable {
        public let bits: Int
        public let inFeatures: Int
        public let outFeatures: Int   // product of all leading dims except the last
    }

    /// Return the inferred bits/inFeatures/outFeatures, or nil if the shapes
    /// don't line up. Handles both 2D (dense linears) and 3D (MoE expert
    /// stacks) — the rule looks only at the last two axes.
    public static func infer(
        qweightShape: [Int],
        scalesShape: [Int],
        groupSize: Int
    ) -> Result? {
        guard qweightShape.count >= 2, scalesShape.count >= 2 else { return nil }
        guard qweightShape.count == scalesShape.count else { return nil }

        // All leading dims must match (batch / expert axes).
        if qweightShape.count > 2 {
            let qLeading = qweightShape.dropLast(2)
            let sLeading = scalesShape.dropLast(2)
            if Array(qLeading) != Array(sLeading) { return nil }
        }

        // The "out" dim (second to last) must match between qweight and scales.
        let qOut = qweightShape[qweightShape.count - 2]
        let sOut = scalesShape[scalesShape.count - 2]
        guard qOut == sOut else { return nil }

        let packedIn = qweightShape.last!
        let nGroups = scalesShape.last!
        guard packedIn > 0, nGroups > 0, groupSize > 0 else { return nil }

        let inFeatures = nGroups * groupSize
        // packedIn * 32 = inFeatures * bits, so bits = packedIn * 32 / inFeatures.
        let numerator = packedIn * 32
        guard numerator % inFeatures == 0 else { return nil }
        let bits = numerator / inFeatures

        let outFeatures: Int
        if qweightShape.count == 2 {
            outFeatures = qOut
        } else {
            outFeatures = qweightShape.dropLast(2).reduce(1, *) * qOut
        }

        return Result(bits: bits, inFeatures: inFeatures, outFeatures: outFeatures)
    }
}
