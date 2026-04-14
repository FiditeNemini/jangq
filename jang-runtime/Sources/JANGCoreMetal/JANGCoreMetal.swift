//
// JANGCoreMetal — Metal compute primitives for JANG v2 tensors.
// Created by Eric Jang (eric@jangq.ai).
//
// Plan 4 scope: 4-bit GEMV correctness only. Later plans extend this to
// 2/6/8-bit, GEMM, and the gather variant used for MoE expert dispatch.
//

import Foundation
import Metal

public enum JANGCoreMetal {
    public static let version = "0.1.0"
}

public enum JANGCoreMetalError: Error, CustomStringConvertible {
    case noDevice
    case libraryLoadFailed(String)
    case functionNotFound(String)
    case bufferAllocFailed(String)
    case dispatchFailed(String)

    public var description: String {
        switch self {
        case .noDevice: return "jangcore-metal: no Metal device available"
        case .libraryLoadFailed(let s): return "jangcore-metal: library load failed: \(s)"
        case .functionNotFound(let s): return "jangcore-metal: kernel '\(s)' not found"
        case .bufferAllocFailed(let s): return "jangcore-metal: buffer alloc failed: \(s)"
        case .dispatchFailed(let s): return "jangcore-metal: dispatch failed: \(s)"
        }
    }
}
