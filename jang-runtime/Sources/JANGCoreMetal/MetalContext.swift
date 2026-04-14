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
        let library: MTLLibrary
        do {
            library = try device.makeDefaultLibrary(bundle: Bundle.module)
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
