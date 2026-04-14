import XCTest
import JANGCore
@testable import JANGCoreMetal

final class QuantizedMatmul4Tests: XCTestCase {
    private func loadFixture() throws -> (
        W_q: Data, W_s: Data, W_b: Data, x: Data, yRef: [Float],
        inFeatures: Int, outFeatures: Int, groupSize: Int
    ) {
        guard let url = Bundle.module.url(
            forResource: "matmul_4bit_64x128",
            withExtension: "safetensors",
            subdirectory: "fixtures"
        ) else {
            throw XCTSkip("fixture missing; regenerate with jang-tools/scripts/gen_matmul_fixture.py")
        }
        let file = try SafetensorsV2File(url: url)
        let yRefBytes = try file.bytes(for: "y_ref")
        var yRef = [Float](repeating: 0, count: yRefBytes.count / 4)
        yRefBytes.withUnsafeBytes { raw in
            for i in 0..<yRef.count {
                yRef[i] = raw.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
            }
        }

        let qInfo = try file.info(for: "W.weight")
        let sInfo = try file.info(for: "W.scales")

        guard let jsonURL = Bundle.module.url(
            forResource: "fixture_info",
            withExtension: "json",
            subdirectory: "fixtures"
        ) else {
            throw XCTSkip("fixture_info.json missing")
        }
        let jsonData = try Data(contentsOf: jsonURL)
        let info = try JSONSerialization.jsonObject(with: jsonData) as! [String: Any]
        let inFeatures = info["in_features"] as! Int
        let outFeatures = info["out_features"] as! Int
        let groupSize = info["group_size"] as! Int

        XCTAssertEqual(qInfo.shape, [outFeatures, inFeatures / 8])
        XCTAssertEqual(sInfo.shape, [outFeatures, inFeatures / groupSize])

        return (
            W_q: try file.bytes(for: "W.weight"),
            W_s: try file.bytes(for: "W.scales"),
            W_b: try file.bytes(for: "W.biases"),
            x: try file.bytes(for: "x"),
            yRef: yRef,
            inFeatures: inFeatures,
            outFeatures: outFeatures,
            groupSize: groupSize
        )
    }

    func testMatchesMLXReference() throws {
        let fx = try loadFixture()
        let ctx = try MetalContext()
        let op = try QuantizedMatmul4(context: ctx)

        let y = try op.run(
            qweight: fx.W_q,
            scales: fx.W_s,
            biases: fx.W_b,
            x: fx.x,
            inFeatures: fx.inFeatures,
            outFeatures: fx.outFeatures,
            groupSize: fx.groupSize
        )

        XCTAssertEqual(y.count, fx.yRef.count)

        var maxAbs: Float = 0
        var maxRel: Float = 0
        for i in 0..<y.count {
            let d = abs(y[i] - fx.yRef[i])
            maxAbs = max(maxAbs, d)
            let rel = d / max(abs(fx.yRef[i]), 1e-6)
            maxRel = max(maxRel, rel)
        }

        XCTAssertLessThan(maxAbs, 1e-2, "max abs error = \(maxAbs)")
        XCTAssertLessThan(maxRel, 1e-2, "max rel error = \(maxRel)")

        print("  QuantizedMatmul4: max abs = \(maxAbs), max rel = \(maxRel)")
    }
}
