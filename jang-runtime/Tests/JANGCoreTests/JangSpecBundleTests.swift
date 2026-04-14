import XCTest
@testable import JANGCore

final class JangSpecBundleTests: XCTestCase {
    func testOpenGemmaBundle() throws {
        let bundleURL: URL
        do {
            bundleURL = try Fixtures.gemmaBundle()
        } catch {
            throw XCTSkip("fixture unavailable: \(error)")
        }

        let bundle = try JangSpecBundle.open(at: bundleURL)

        // Manifest sanity — these values come from the Gemma-4-26B JANG_4M model.
        XCTAssertEqual(bundle.manifest.bundleVersion, 1)
        XCTAssertEqual(bundle.manifest.targetArch, "gemma4")
        XCTAssertEqual(bundle.manifest.nLayers, 30)
        XCTAssertEqual(bundle.manifest.nExpertsPerLayer, 128)
        XCTAssertEqual(bundle.manifest.nExpertsTotal, 30 * 128)
        XCTAssertFalse(bundle.manifest.hasDraft)
        XCTAssertFalse(bundle.manifest.hasRouterPrior)

        // Index sanity.
        XCTAssertEqual(bundle.index.nLayers, 30)
        XCTAssertEqual(bundle.index.nExpertsPerLayer, 128)
        XCTAssertEqual(bundle.index.entries.count, 30 * 128)

        // Hot core file exists.
        XCTAssertTrue(FileManager.default.fileExists(atPath: bundle.hotCoreURL.path))
    }

    func testLoadFirstAndLastExpert() throws {
        let bundleURL: URL
        do {
            bundleURL = try Fixtures.gemmaBundle()
        } catch {
            throw XCTSkip("fixture unavailable: \(error)")
        }

        let bundle = try JangSpecBundle.open(at: bundleURL)

        let first = try bundle.store.load(layer: 0, expert: 0)
        XCTAssertEqual(first.layerIdx, 0)
        XCTAssertEqual(first.expertID, 0)
        XCTAssertEqual(first.tensors.count, 9)
        XCTAssertGreaterThan(first.bits, 0)
        XCTAssertNotNil(first.tensor(kind: .gate, dtype: .qweight))
        XCTAssertNotNil(first.tensor(kind: .down, dtype: .biases))

        let lastLayer = bundle.manifest.nLayers - 1
        let lastExpert = bundle.manifest.nExpertsPerLayer - 1
        let last = try bundle.store.load(layer: lastLayer, expert: lastExpert)
        XCTAssertEqual(last.layerIdx, lastLayer)
        XCTAssertEqual(last.expertID, lastExpert)
        XCTAssertEqual(last.tensors.count, 9)
    }
}
