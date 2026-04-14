import XCTest
@testable import JANGCore

final class BitInferenceTests: XCTestCase {
    func testInfer2DWeightAt4Bits() {
        // in_features = 4096, out_features = 3072, bits = 4, group_size = 64
        //   packed_in = 4096 * 4 / 32 = 512
        //   n_groups  = 4096 / 64    = 64
        let result = BitInference.infer(
            qweightShape: [3072, 512],
            scalesShape: [3072, 64],
            groupSize: 64
        )
        XCTAssertEqual(result?.bits, 4)
        XCTAssertEqual(result?.inFeatures, 4096)
        XCTAssertEqual(result?.outFeatures, 3072)
    }

    func testInfer2DWeightAt2Bits() {
        // in_features = 8192, bits = 2, group_size = 64
        //   packed_in = 8192 * 2 / 32 = 512
        //   n_groups  = 8192 / 64    = 128
        let result = BitInference.infer(
            qweightShape: [5120, 512],
            scalesShape: [5120, 128],
            groupSize: 64
        )
        XCTAssertEqual(result?.bits, 2)
        XCTAssertEqual(result?.inFeatures, 8192)
    }

    func testInfer3DExpertTensor() {
        // 128 experts, intermediate = 1024, hidden = 4096, bits = 4
        //   packed_in = 4096 * 4 / 32 = 512
        //   n_groups  = 4096 / 64    = 64
        let result = BitInference.infer(
            qweightShape: [128, 1024, 512],
            scalesShape: [128, 1024, 64],
            groupSize: 64
        )
        XCTAssertEqual(result?.bits, 4)
        XCTAssertEqual(result?.inFeatures, 4096)
    }

    func testReturnsNilOnMismatchedRank() {
        let result = BitInference.infer(
            qweightShape: [3072, 512],
            scalesShape: [64],
            groupSize: 64
        )
        XCTAssertNil(result)
    }
}
