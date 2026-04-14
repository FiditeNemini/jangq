import Foundation

/// A `.jangspec` bundle opened for reading.
///
/// `JangSpecBundle.open(at:)` parses the manifest and expert index and
/// constructs an `ExpertStore` for lazy per-expert loading. The bundle
/// does not touch the hot-core safetensors file in Plan 2; that's Plan 3's
/// responsibility. The URL is exposed so later code can load it.
public struct JangSpecBundle: Sendable {
    public let url: URL
    public let manifest: JangSpecManifest
    public let index: ExpertIndex
    public let store: ExpertStore

    public var hotCoreURL: URL {
        url.appendingPathComponent(JangSpecFormat.hotCoreFilename)
    }

    public var manifestURL: URL {
        url.appendingPathComponent(JangSpecFormat.manifestFilename)
    }

    public var indexURL: URL {
        url.appendingPathComponent(JangSpecFormat.indexFilename)
    }

    public static func open(at url: URL) throws -> JangSpecBundle {
        let manifestURL = url.appendingPathComponent(JangSpecFormat.manifestFilename)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw JangSpecError.fileMissing(manifestURL)
        }
        let manifest = try JangSpecManifest.load(from: manifestURL)

        let indexURL = url.appendingPathComponent(JangSpecFormat.indexFilename)
        guard FileManager.default.fileExists(atPath: indexURL.path) else {
            throw JangSpecError.fileMissing(indexURL)
        }
        let index = try ExpertIndex(contentsOf: indexURL)

        // Sanity: manifest and index agree on dimensions.
        guard index.nLayers == manifest.nLayers else {
            throw JangSpecError.invalidManifest(
                "index.n_layers=\(index.nLayers) disagrees with manifest.n_layers=\(manifest.nLayers)"
            )
        }
        guard index.nExpertsPerLayer == manifest.nExpertsPerLayer else {
            throw JangSpecError.invalidManifest(
                "index.n_experts_per_layer=\(index.nExpertsPerLayer) disagrees with manifest.n_experts_per_layer=\(manifest.nExpertsPerLayer)"
            )
        }
        guard index.entries.count == manifest.nExpertsTotal else {
            throw JangSpecError.invalidManifest(
                "index entries=\(index.entries.count) disagrees with manifest.n_experts_total=\(manifest.nExpertsTotal)"
            )
        }

        let store = ExpertStore(bundleURL: url, index: index)
        return JangSpecBundle(url: url, manifest: manifest, index: index, store: store)
    }
}
