import Foundation
import Metal

/// Owns a `MTLDevice`, a default `MTLCommandQueue`, and the compiled
/// `MTLLibrary` built from the Metal resources shipped with this target.
public final class MetalContext: @unchecked Sendable {
    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let library: MTLLibrary

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw JANGCoreMetalError.noDevice
        }
        guard let queue = device.makeCommandQueue() else {
            throw JANGCoreMetalError.libraryLoadFailed("makeCommandQueue returned nil")
        }
        // SwiftPM 6.2 does not auto-compile .metal files into a default
        // metallib for library targets (unlike Xcode projects). We ship the
        // kernel as a plain resource (.copy) and compile it at runtime from
        // source via `device.makeLibrary(source:)`. For Plan 4's tiny kernel
        // this is ~1 ms and keeps the toolchain simple.
        let library: MTLLibrary
        do {
            // Try default library first (in case future SPM versions build it).
            if let defaultLib = try? device.makeDefaultLibrary(bundle: Bundle.module),
               defaultLib.functionNames.contains("jang_v2_quant_matmul_4bit_gemv") {
                library = defaultLib
            } else {
                guard let url = Bundle.module.url(
                    forResource: "JangV2QuantMatmul",
                    withExtension: "metal"
                ) else {
                    throw JANGCoreMetalError.libraryLoadFailed(
                        "JangV2QuantMatmul.metal resource not found in bundle"
                    )
                }
                let source = try String(contentsOf: url, encoding: .utf8)
                library = try device.makeLibrary(source: source, options: nil)
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
        guard let fn = library.makeFunction(name: name) else {
            throw JANGCoreMetalError.functionNotFound(name)
        }
        return try device.makeComputePipelineState(function: fn)
    }
}
