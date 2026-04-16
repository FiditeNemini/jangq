import Foundation
import Metal

/// Owns a `MTLDevice`, a default `MTLCommandQueue`, and the compiled
/// `MTLLibrary` built from the Metal resources shipped with this target.
///
/// The `library` field is the primary library compiled from the JANGCoreMetal
/// .metal source bundle (JangV2QuantMatmul + JANGTQMatmul + JANGTQAffine8Matmul).
///
/// `extraLibraries` holds any externally-loaded libraries that were attached
/// after init via `loadLibrary(at:)` — typically the pre-compiled
/// `jang.metallib` shipped with `jang-runtime/Metal/` which contains the
/// dense-model kernels (`jang_rms_norm`, `jang_rope`, `jang_attention_decode`,
/// `jang_softmax`, etc.). `pipeline(functionNamed:)` searches all libraries
/// in order: primary first, then `extraLibraries`.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary
    public private(set) var extraLibraries: [MTLLibrary] = []

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw JANGCoreMetalError.noDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw JANGCoreMetalError.libraryLoadFailed("makeCommandQueue returned nil")
        }
        // SwiftPM 6.2 does not auto-compile .metal files into a default
        // metallib for library targets (unlike Xcode projects). We ship each
        // .metal file as a plain resource (.copy) and compile them at runtime
        // by concatenating their sources into one source string and calling
        // `device.makeLibrary(source:)`. The combined library exposes every
        // kernel by name regardless of which file it came from.
        let library: MTLLibrary
        do {
            // Try default library first (in case future SPM versions build it).
            if let defaultLib = try? device.makeDefaultLibrary(bundle: Bundle.module),
               defaultLib.functionNames.contains("jang_v2_quant_matmul_4bit_gemv") {
                library = defaultLib
            } else {
                let resources = ["JangV2QuantMatmul", "JANGTQMatmul", "JANGTQAffine8Matmul", "JANGTQDecodeOps"]
                var combined = ""
                for name in resources {
                    guard let url = Bundle.module.url(forResource: name, withExtension: "metal") else {
                        throw JANGCoreMetalError.libraryLoadFailed(
                            "\(name).metal resource not found in bundle"
                        )
                    }
                    combined += try String(contentsOf: url, encoding: .utf8) + "\n"
                }
                library = try device.makeLibrary(source: combined, options: nil)
            }
        } catch let error as JANGCoreMetalError {
            throw error
        } catch {
            throw JANGCoreMetalError.libraryLoadFailed(String(describing: error))
        }
        self.device = device
        self.queue = queue
        self.library = library
    }

    public func pipeline(functionNamed name: String) throws -> MTLComputePipelineState {
        if let fn = library.makeFunction(name: name) {
            return try device.makeComputePipelineState(function: fn)
        }
        for lib in extraLibraries {
            if let fn = lib.makeFunction(name: name) {
                return try device.makeComputePipelineState(function: fn)
            }
        }
        throw JANGCoreMetalError.functionNotFound(name)
    }

    /// Attach an additional pre-compiled `.metallib` to this context.
    /// Subsequent `pipeline(functionNamed:)` calls will search this library
    /// after the primary one. Used by JANGTQ inference to pull in the
    /// pre-compiled `jang.metallib` containing RMSNorm/RoPE/SDPA/etc.
    public func loadLibrary(at url: URL) throws {
        let lib = try device.makeLibrary(URL: url)
        extraLibraries.append(lib)
    }
}
