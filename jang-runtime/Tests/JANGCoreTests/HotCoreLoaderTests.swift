import XCTest
@testable import JANGCore

final class HotCoreLoaderTests: XCTestCase {
    /// Prefer the MiniMax bundle for this test; fall back to the Gemma
    /// fixture if MiniMax isn't present. Skip entirely if neither is there.
    private func resolveBundleURL() throws -> URL {
        let minimax = URL(fileURLWithPath: "/Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec")
        if FileManager.default.fileExists(
            atPath: minimax.appendingPathComponent(JangSpecFormat.manifestFilename).path
        ) {
            return minimax
        }
        let gemma = try Fixtures.gemmaBundle()
        return gemma
    }

    func testLoadHotCoreFromRealBundle() throws {
        let bundleURL: URL
        do {
            bundleURL = try resolveBundleURL()
        } catch {
            throw XCTSkip("no real bundle available: \(error)")
        }
        let bundle = try JangSpecBundle.open(at: bundleURL)

        // Read block_size from the bundle's jang_config.json; JANG v2 bundles
        // use either 64 or 128 depending on quantization profile. The loader
        // needs the correct group size for bit inference.
        let groupSize = Self.readGroupSize(at: bundleURL) ?? 64
        let hot = try HotCoreLoader.load(bundle: bundle, groupSize: groupSize)

        // Every base name listed in the manifest must be accounted for.
        // Quantized keys are already stored by base; raw keys retain their
        // full `.weight` suffix, so strip suffixes on both sides for a fair
        // comparison.
        let accounted = Set(hot.quantized.keys)
            .union(hot.raw.keys.map { Self.stripSuffix($0) })
        let expectedBases = Set(
            bundle.manifest.hotCoreTensors.map { Self.stripSuffix($0) }
        )
        let missing = expectedBases.subtracting(accounted)
        XCTAssertTrue(
            missing.isEmpty,
            "base names from manifest not loaded: \(Array(missing).prefix(10))"
        )

        // There must be at least one quantized and one raw tensor.
        XCTAssertFalse(hot.quantized.isEmpty, "expected quantized tensors in hot core")
        XCTAssertFalse(hot.raw.isEmpty, "expected raw tensors in hot core (norms, biases)")

        // Every quantized tensor has a sensible bit width (2..8).
        for (_, q) in hot.quantized {
            XCTAssertGreaterThanOrEqual(q.bits, 2)
            XCTAssertLessThanOrEqual(q.bits, 8)
            XCTAssertEqual(q.qweight.count, Self.expectedQweightBytes(q))
            XCTAssertEqual(q.scales.count, Self.expectedScalesBytes(q))
            XCTAssertEqual(q.biases.count, Self.expectedScalesBytes(q))
        }
    }

    private static func stripSuffix(_ full: String) -> String {
        for s in [".weight", ".scales", ".biases"] {
            if full.hasSuffix(s) {
                return String(full.dropLast(s.count))
            }
        }
        return full
    }

    private static func expectedQweightBytes(_ q: QuantizedTensorView) -> Int {
        return q.qweightShape.reduce(1, *) * 4  // uint32
    }

    private static func expectedScalesBytes(_ q: QuantizedTensorView) -> Int {
        return q.scalesShape.reduce(1, *) * 2   // float16
    }

    private static func readGroupSize(at bundleURL: URL) -> Int? {
        let cfgURL = bundleURL.appendingPathComponent("target/jang_config.json")
        guard let data = try? Data(contentsOf: cfgURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let q = obj["quantization"] as? [String: Any],
              let bs = q["block_size"] as? NSNumber else {
            return nil
        }
        return bs.intValue
    }
}
