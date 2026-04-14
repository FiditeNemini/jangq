# Plan 4 — Metal Quantized Matmul Kernel (4-bit GEMV, correctness only)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** First Metal compute kernel in JANGCore: a correctness-only 4-bit GEMV (`y = W @ x`) for JANG v2 MLX-native packed weights. Validates the bit-unpacking convention and the Data-to-MTLBuffer path, and ships a Swift binding (`QuantizedMatmul4`) that the later MoE layers will build on. Scope is deliberately tiny — one bit width, one shape class, one fixture — so that if something is wrong with our understanding of the packing convention, we find out in a ~200-line kernel, not in a 2k-line MoE implementation.

**Architecture:** New `JANGCoreMetal` library target depending on `JANGCore`. Metal shader at `jang-runtime/Metal/JangV2QuantMatmul.metal` compiled via SwiftPM resources. A `MetalContext` struct owns `MTLDevice`, `MTLCommandQueue`, and the compiled `MTLLibrary`. The `QuantizedMatmul4` struct takes a packed weight + input vector, allocates Metal buffers (via `.storageModeShared`), dispatches the kernel, and returns a fp32 output vector. A Python fixture generator under `jang-tools/scripts/` creates a deterministic `(W, x, y_ref)` triple using `mlx.quantize`; the committed safetensors fixture is ~2 KB. The Swift test loads the same fixture, runs our kernel, and compares output against `y_ref` within 1e-2 max abs error.

**Tech Stack:** Swift 6.0, SwiftPM resources, Metal 3, MSL 3.0. XCTest. Python `mlx` for fixture generation.

**Verified conventions (from `mx.quantize`, 2026-04-13):**
- 4-bit: 8 values packed per `uint32`, LSB-first. Position `k` (0..7) lives at bit `k*4`.
- Dequant: `val = q_int * scale + bias`. Scales and biases are per-group (`group_size` default 64; MiniMax JANG_2L uses 128).
- Scales/biases from `mx.quantize` are `float32`; JANG v2 writes them as `float16`. Our kernel reads them as `half`.

**Depends on:** Plans 1–3 complete. This plan branches from `jang-spec-plan3-hotcore`.

**Out of scope:**
- 2/3/6/8-bit — later increments.
- Batched GEMM (token > 1) — Plan 5.
- Gather variant for MoE expert dispatch — Plan 6.
- Threadgroup/SIMD optimizations — correctness first.
- Reading real JANG weights — Plan 5 wires this into a forward pass. Plan 4 uses a purpose-built tiny fixture.

---

## File structure

New files:

```
jang-runtime/Sources/JANGCoreMetal/
  JANGCoreMetal.swift         umbrella, errors
  MetalContext.swift          MTLDevice + library loader
  MetalBuffer.swift           Data -> MTLBuffer helpers
  QuantizedMatmul4.swift      4-bit GEMV Swift binding

jang-runtime/Metal/
  JangV2QuantMatmul.metal     the kernel (included as SPM resource)

jang-runtime/Tests/JANGCoreMetalTests/
  QuantizedMatmul4Tests.swift

jang-runtime/Tests/JANGCoreMetalTests/fixtures/
  matmul_4bit_64x128.safetensors    committed fixture (~2 KB)
  fixture_info.json                  metadata (shapes, group_size, seed)

jang-tools/scripts/
  gen_matmul_fixture.py       regenerates the fixture deterministically

jang-runtime/Package.swift     MODIFY: add JANGCoreMetal library + test target, declare Metal resource
```

---

## Task 0: Branch setup

**Files:** none

- [ ] **Step 1: Confirm state and branch**

```bash
cd /Users/eric/jang && git status && git log -1 --oneline && git branch --show-current
```
Expected: clean tree, branch `jang-spec-plan3-hotcore`, latest commit is Plan 3's STATUS update (`a4690f7` or newer).

```bash
git checkout -b jang-spec-plan4-metal-matmul
mkdir -p jang-runtime/Sources/JANGCoreMetal \
         jang-runtime/Tests/JANGCoreMetalTests/fixtures \
         jang-tools/scripts
```

---

## Task 1: Python fixture generator

**Files:**
- Create: `jang-tools/scripts/gen_matmul_fixture.py`
- Create: `jang-runtime/Tests/JANGCoreMetalTests/fixtures/matmul_4bit_64x128.safetensors`
- Create: `jang-runtime/Tests/JANGCoreMetalTests/fixtures/fixture_info.json`

**Background.** The fixture is a deterministic tiny quantized matmul: a 4-bit quantized weight `W` of shape `(out=128, in=64)`, a fp16 input vector `x` of shape `(in=64,)`, and the reference output `y_ref = W_dq @ x` of shape `(out=128,)` computed by dequantizing via MLX and multiplying in fp32. The Swift kernel's job is to produce the same `y_ref` within 1e-2 max abs error.

- [ ] **Step 1: Write the fixture generator**

Write `jang-tools/scripts/gen_matmul_fixture.py`:
```python
#!/usr/bin/env python3
"""
Generate a deterministic 4-bit quantized matmul fixture for Plan 4's Swift test.

Produces a safetensors file containing:
    W.weight   uint32   (out, in/8)     - packed 4-bit quantized weights
    W.scales   float16  (out, n_groups) - per-group scale
    W.biases   float16  (out, n_groups) - per-group bias
    x          float16  (in,)           - input vector
    y_ref      float32  (out,)          - reference output W_dq @ x

Plus a side-car fixture_info.json with shapes and metadata.
"""

import json
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors.numpy import save_file

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "jang-runtime" / "Tests" / "JANGCoreMetalTests" / "fixtures"
SEED = 0xBADC0FFEE
BITS = 4
GROUP_SIZE = 64
IN_FEATURES = 64
OUT_FEATURES = 128


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Deterministic weight matrix, fp32 for mx.quantize.
    rng = np.random.default_rng(SEED)
    W_full = rng.standard_normal((OUT_FEATURES, IN_FEATURES)).astype(np.float32)

    # Input vector, fp16 because that's what the kernel reads.
    x_f16 = rng.standard_normal((IN_FEATURES,)).astype(np.float16)

    # Quantize via MLX (authoritative packing).
    W_mx = mx.array(W_full)
    q, s, b = mx.quantize(W_mx, bits=BITS, group_size=GROUP_SIZE)

    # Force materialization by converting to numpy.
    q_np = np.array(q, dtype=np.uint32, copy=True)
    s_np = np.array(s, dtype=np.float16, copy=True)
    b_np = np.array(b, dtype=np.float16, copy=True)

    # Reference output: dequantize via MLX and matmul in fp32.
    W_dq = mx.dequantize(q, s, b, group_size=GROUP_SIZE, bits=BITS)
    W_dq_np = np.array(W_dq, dtype=np.float32, copy=True)
    y_ref = (W_dq_np.astype(np.float32) @ x_f16.astype(np.float32)).astype(np.float32)

    # Shape sanity.
    assert q_np.shape == (OUT_FEATURES, IN_FEATURES * BITS // 32), q_np.shape
    assert s_np.shape == (OUT_FEATURES, IN_FEATURES // GROUP_SIZE), s_np.shape
    assert b_np.shape == (OUT_FEATURES, IN_FEATURES // GROUP_SIZE), b_np.shape

    save_file(
        {
            "W.weight": q_np,
            "W.scales": s_np,
            "W.biases": b_np,
            "x": x_f16,
            "y_ref": y_ref,
        },
        str(OUT_DIR / "matmul_4bit_64x128.safetensors"),
    )

    info = {
        "seed": SEED,
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "qweight_shape": list(q_np.shape),
        "scales_shape": list(s_np.shape),
        "biases_shape": list(b_np.shape),
        "x_shape": list(x_f16.shape),
        "y_ref_shape": list(y_ref.shape),
        "mlx_pack_convention": "LSB-first 4-bit: position k in [0..7] at bit k*4 of each uint32",
        "dequant_formula": "val = q_int * scale + bias, per-group",
    }
    (OUT_DIR / "fixture_info.json").write_text(json.dumps(info, indent=2) + "\n")

    print(f"  wrote {OUT_DIR / 'matmul_4bit_64x128.safetensors'}")
    print(f"  qweight {q_np.shape} scales {s_np.shape} biases {b_np.shape}")
    print(f"  x {x_f16.shape}  y_ref {y_ref.shape}  y_ref[:4]={y_ref[:4].tolist()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixture**

```bash
cd /Users/eric/jang && python3 jang-tools/scripts/gen_matmul_fixture.py
```
Expected: one-line confirmation printing `y_ref[:4]` as four finite floats. The safetensors file is ~2 KB and `fixture_info.json` is a handful of lines.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-tools/scripts/gen_matmul_fixture.py \
    jang-runtime/Tests/JANGCoreMetalTests/fixtures/matmul_4bit_64x128.safetensors \
    jang-runtime/Tests/JANGCoreMetalTests/fixtures/fixture_info.json
git commit -m "jang-core-metal: 4-bit matmul fixture generator and committed fixture"
```

---

## Task 2: Package.swift — add JANGCoreMetal target with Metal resource

**Files:**
- Modify: `jang-runtime/Package.swift`

- [ ] **Step 1: Edit Package.swift**

Open `jang-runtime/Package.swift` and add to `products`:
```swift
        .library(name: "JANGCoreMetal", targets: ["JANGCoreMetal"]),
```

Add to `targets`:
```swift
        .target(
            name: "JANGCoreMetal",
            dependencies: ["JANGCore"],
            path: "Sources/JANGCoreMetal",
            resources: [
                .copy("../../Metal/JangV2QuantMatmul.metal")
            ]
        ),
        .testTarget(
            name: "JANGCoreMetalTests",
            dependencies: ["JANGCoreMetal"],
            path: "Tests/JANGCoreMetalTests",
            resources: [
                .copy("fixtures")
            ]
        ),
```

Note: the `.copy` path is relative to the target's `path`, so `Sources/JANGCoreMetal/../../Metal/JangV2QuantMatmul.metal` resolves to `jang-runtime/Metal/JangV2QuantMatmul.metal`. SwiftPM will compile the `.metal` file into a `.metallib` automatically and ship it inside the bundle; at runtime the library is accessible via `Bundle.module`.

**If SPM refuses the `..` resource path** (SwiftPM 6 sometimes does for safety), fall back to copying the kernel file into `Sources/JANGCoreMetal/JangV2QuantMatmul.metal` at Task 3 time and declare the resource as `.copy("JangV2QuantMatmul.metal")`. Document the choice in the Task 2 commit.

- [ ] **Step 2: Verify the package resolves**

```bash
cd /Users/eric/jang/jang-runtime && swift package describe 2>&1 | grep -A2 "JANGCoreMetal"
```
Expected: both `JANGCoreMetal` and `JANGCoreMetalTests` appear. The Metal file is currently absent — that's fine, Task 3 creates it.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Package.swift
git commit -m "jang-core-metal: register JANGCoreMetal target with Metal resource"
```

---

## Task 3: The Metal kernel

**Files:**
- Create: `jang-runtime/Metal/JangV2QuantMatmul.metal` (or `jang-runtime/Sources/JANGCoreMetal/JangV2QuantMatmul.metal` if Task 2 fell back to the co-located layout)

- [ ] **Step 1: Write the kernel**

Write the file:
```metal
//
// JANG v2 quantized matmul — 4-bit GEMV (y = W @ x, single token).
// Created by Eric Jang (eric@jangq.ai).
//
// Convention (verified against mx.quantize on 2026-04-13):
//
//   W  packed as uint32, shape (out, in/8). Each uint32 holds 8 consecutive
//      4-bit values, LSB-first: position k (0..7) at bit k*4.
//   scales / biases: half, shape (out, in/group_size). One value per group.
//   x: half, shape (in,).
//   y: float, shape (out,) — kept in fp32 to avoid accumulation drift.
//
//   Dequant formula: val = q_int * scale + bias, per-group.
//
// This first implementation is intentionally naive: one thread per output
// row, no SIMD reduction, no threadgroup memory. Correctness first, perf
// in a later pass.
//

#include <metal_stdlib>
using namespace metal;

struct QuantMatmul4Params {
    uint in_features;
    uint out_features;
    uint group_size;
    uint n_groups;         // in_features / group_size
    uint packed_in;        // in_features / 8   (4-bit: 8 per uint32)
};

kernel void jang_v2_quant_matmul_4bit_gemv(
    device const uint32_t*      qweight [[buffer(0)]],  // [out * packed_in]
    device const half*          scales  [[buffer(1)]],  // [out * n_groups]
    device const half*          biases  [[buffer(2)]],  // [out * n_groups]
    device const half*          x       [[buffer(3)]],  // [in]
    device       float*         y       [[buffer(4)]],  // [out]
    constant QuantMatmul4Params& p      [[buffer(5)]],
    uint tid                            [[thread_position_in_grid]]
) {
    if (tid >= p.out_features) {
        return;
    }

    const uint out = tid;
    const uint row_q_offset = out * p.packed_in;
    const uint row_s_offset = out * p.n_groups;

    float acc = 0.0f;

    for (uint g = 0; g < p.n_groups; g++) {
        const float scale = float(scales[row_s_offset + g]);
        const float bias  = float(biases[row_s_offset + g]);
        const uint  g_start = g * p.group_size;

        // Each group has group_size inputs. At 4 bits that's group_size/8
        // uint32 words per group.
        const uint words_per_group = p.group_size / 8u;

        for (uint w = 0; w < words_per_group; w++) {
            const uint i_base = g_start + w * 8u;
            const uint32_t word = qweight[row_q_offset + (g_start / 8u) + w];

            // Unroll the 8 unpacks.
            for (uint k = 0; k < 8; k++) {
                const uint q = (word >> (k * 4u)) & 0xFu;
                const float dq = float(q) * scale + bias;
                const float xv = float(x[i_base + k]);
                acc = fma(dq, xv, acc);
            }
        }
    }

    y[out] = acc;
}
```

- [ ] **Step 2: Sanity-compile the kernel standalone**

```bash
cd /Users/eric/jang/jang-runtime && \
  xcrun -sdk macosx metal -c Metal/JangV2QuantMatmul.metal -o /tmp/JangV2QuantMatmul.air 2>&1
```
(Adjust the path if Task 2 used the co-located layout.) Expected: clean compile, no errors.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Metal/ jang-runtime/Sources/JANGCoreMetal/*.metal 2>/dev/null
# (the glob covers both layouts; one will have no matches and git will silently skip it)
git commit -m "jang-core-metal: 4-bit GEMV quantized matmul Metal kernel"
```

---

## Task 4: MetalContext and MetalBuffer helpers

**Files:**
- Create: `jang-runtime/Sources/JANGCoreMetal/JANGCoreMetal.swift`
- Create: `jang-runtime/Sources/JANGCoreMetal/MetalContext.swift`
- Create: `jang-runtime/Sources/JANGCoreMetal/MetalBuffer.swift`

- [ ] **Step 1: Umbrella**

Write `jang-runtime/Sources/JANGCoreMetal/JANGCoreMetal.swift`:
```swift
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
```

- [ ] **Step 2: MetalContext**

Write `jang-runtime/Sources/JANGCoreMetal/MetalContext.swift`:
```swift
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
        // SPM compiles `.metal` resources into a default metallib and
        // places it inside Bundle.module. `makeDefaultLibrary(bundle:)` picks
        // it up automatically.
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
```

- [ ] **Step 3: MetalBuffer helper**

Write `jang-runtime/Sources/JANGCoreMetal/MetalBuffer.swift`:
```swift
import Foundation
import Metal

/// Helpers for turning `Data` slices into `MTLBuffer`s.
///
/// Plan 4 uses plain `makeBuffer(bytes:length:options:)` which copies into
/// a shared-storage buffer. A later plan may switch to `bytesNoCopy` for
/// zero-copy against mmap-backed `Data` when alignment allows; for tiny
/// fixtures the copy cost is negligible.
public enum MetalBuffer {
    public static func fromData(_ data: Data, device: MTLDevice) throws -> MTLBuffer {
        return try data.withUnsafeBytes { raw -> MTLBuffer in
            guard let base = raw.baseAddress, raw.count > 0 else {
                throw JANGCoreMetalError.bufferAllocFailed("empty Data")
            }
            guard let buf = device.makeBuffer(
                bytes: base,
                length: raw.count,
                options: [.storageModeShared]
            ) else {
                throw JANGCoreMetalError.bufferAllocFailed("\(raw.count) bytes")
            }
            return buf
        }
    }

    public static func empty(bytes: Int, device: MTLDevice) throws -> MTLBuffer {
        guard let buf = device.makeBuffer(length: bytes, options: [.storageModeShared]) else {
            throw JANGCoreMetalError.bufferAllocFailed("empty \(bytes) bytes")
        }
        // Zero-fill so reads before writes are deterministic.
        memset(buf.contents(), 0, bytes)
        return buf
    }
}
```

- [ ] **Step 4: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build 2>&1 | tail -10
```
Expected: build success.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCoreMetal/JANGCoreMetal.swift \
    jang-runtime/Sources/JANGCoreMetal/MetalContext.swift \
    jang-runtime/Sources/JANGCoreMetal/MetalBuffer.swift
git commit -m "jang-core-metal: MetalContext + MetalBuffer helpers"
```

---

## Task 5: QuantizedMatmul4 Swift binding

**Files:**
- Create: `jang-runtime/Sources/JANGCoreMetal/QuantizedMatmul4.swift`

- [ ] **Step 1: Implement the binding**

Write `jang-runtime/Sources/JANGCoreMetal/QuantizedMatmul4.swift`:
```swift
import Foundation
import Metal
import JANGCore

/// 4-bit GEMV: y (fp32) = W (4-bit packed) @ x (fp16).
///
/// `W` is supplied as three `Data` slices following the JANG v2 convention:
///   qweight: uint32, shape (out, in/8)    — LSB-first 4-bit packing
///   scales : float16, shape (out, n_groups)
///   biases : float16, shape (out, n_groups)
/// with `n_groups = in / group_size` and `group_size` a power of 2.
///
/// The kernel uses one thread per output row. This is not a perf-tuned
/// implementation; Plan 4 exists to validate correctness against an MLX
/// reference, not to compete on tokens/sec.
public struct QuantizedMatmul4 {
    public let context: MetalContext
    public let pipeline: MTLComputePipelineState

    public init(context: MetalContext) throws {
        self.context = context
        self.pipeline = try context.pipeline(functionNamed: "jang_v2_quant_matmul_4bit_gemv")
    }

    public func run(
        qweight: Data,
        scales: Data,
        biases: Data,
        x: Data,
        inFeatures: Int,
        outFeatures: Int,
        groupSize: Int
    ) throws -> [Float] {
        let packedIn = inFeatures / 8
        let nGroups = inFeatures / groupSize
        let expectedQ = outFeatures * packedIn * 4
        let expectedS = outFeatures * nGroups * 2
        let expectedX = inFeatures * 2
        precondition(qweight.count == expectedQ, "qweight bytes \(qweight.count) != \(expectedQ)")
        precondition(scales.count == expectedS, "scales bytes \(scales.count) != \(expectedS)")
        precondition(biases.count == expectedS, "biases bytes \(biases.count) != \(expectedS)")
        precondition(x.count == expectedX, "x bytes \(x.count) != \(expectedX)")

        let dev = context.device

        let qBuf = try MetalBuffer.fromData(qweight, device: dev)
        let sBuf = try MetalBuffer.fromData(scales, device: dev)
        let bBuf = try MetalBuffer.fromData(biases, device: dev)
        let xBuf = try MetalBuffer.fromData(x, device: dev)
        let yBuf = try MetalBuffer.empty(bytes: outFeatures * MemoryLayout<Float>.stride, device: dev)

        var params = QuantMatmul4Params(
            in_features: UInt32(inFeatures),
            out_features: UInt32(outFeatures),
            group_size: UInt32(groupSize),
            n_groups: UInt32(nGroups),
            packed_in: UInt32(packedIn)
        )
        let paramBuf: MTLBuffer = try withUnsafeBytes(of: &params) { raw -> MTLBuffer in
            guard let base = raw.baseAddress else {
                throw JANGCoreMetalError.bufferAllocFailed("params")
            }
            guard let buf = dev.makeBuffer(
                bytes: base, length: raw.count, options: [.storageModeShared]
            ) else {
                throw JANGCoreMetalError.bufferAllocFailed("params \(raw.count)")
            }
            return buf
        }

        guard let commandBuffer = context.queue.makeCommandBuffer() else {
            throw JANGCoreMetalError.dispatchFailed("no command buffer")
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw JANGCoreMetalError.dispatchFailed("no compute encoder")
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(qBuf, offset: 0, index: 0)
        encoder.setBuffer(sBuf, offset: 0, index: 1)
        encoder.setBuffer(bBuf, offset: 0, index: 2)
        encoder.setBuffer(xBuf, offset: 0, index: 3)
        encoder.setBuffer(yBuf, offset: 0, index: 4)
        encoder.setBuffer(paramBuf, offset: 0, index: 5)

        let threadsPerThreadgroup = MTLSize(
            width: min(pipeline.maxTotalThreadsPerThreadgroup, 64),
            height: 1,
            depth: 1
        )
        let threadgroups = MTLSize(
            width: (outFeatures + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: 1,
            depth: 1
        )
        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if let err = commandBuffer.error {
            throw JANGCoreMetalError.dispatchFailed(String(describing: err))
        }

        var result = [Float](repeating: 0, count: outFeatures)
        let ptr = yBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<outFeatures {
            result[i] = ptr[i]
        }
        return result
    }
}

/// Must match Metal-side `struct QuantMatmul4Params`.
struct QuantMatmul4Params {
    var in_features: UInt32
    var out_features: UInt32
    var group_size: UInt32
    var n_groups: UInt32
    var packed_in: UInt32
}
```

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build 2>&1 | tail -10
```
Expected: build success.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCoreMetal/QuantizedMatmul4.swift
git commit -m "jang-core-metal: QuantizedMatmul4 Swift binding"
```

---

## Task 6: Correctness test against fixture

**Files:**
- Create: `jang-runtime/Tests/JANGCoreMetalTests/QuantizedMatmul4Tests.swift`

- [ ] **Step 1: Write the test**

Write `jang-runtime/Tests/JANGCoreMetalTests/QuantizedMatmul4Tests.swift`:
```swift
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
```

- [ ] **Step 2: Run the test**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter QuantizedMatmul4Tests 2>&1 | tail -25
```
Expected: 1 test passes. The print line shows max abs and max rel errors (typically < 1e-3).

If the test **fails** on a numerical mismatch:
1. Double-check the kernel unpacking matches MLX's LSB-first convention. `fixture_info.json` documents the convention; the kernel must match.
2. Suspect `half` precision in the scale/bias load. If max abs error is ~1e-2 but max rel is small, it's expected fp16 drift. Report DONE_WITH_CONCERNS with numbers, don't silently relax the bound.
3. If results are garbage, the kernel has a structural bug. Check the per-group base index arithmetic — `(g_start / 8u) + w` should read the correct word within the row.
4. STOP and report BLOCKED if you can't figure out the mismatch. Capture the first 8 values of `y` vs `y_ref`.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add -f jang-runtime/Tests/JANGCoreMetalTests/QuantizedMatmul4Tests.swift
git commit -m "jang-core-metal: 4-bit GEMV correctness test against MLX reference"
```

---

## Task 7: Final sweep + STATUS update

**Files:**
- Modify: `docs/superpowers/notes/jang-spec-STATUS.md`

- [ ] **Step 1: Full Swift test suite**

```bash
cd /Users/eric/jang/jang-runtime && swift test 2>&1 | tail -20
```
Expected: all `JANGCoreTests`, `JANGCoreMetalTests`, and the existing `JANGTests` pass.

- [ ] **Step 2: Full release build**

```bash
cd /Users/eric/jang/jang-runtime && swift build -c release 2>&1 | tail -10
```
Expected: `jang`, `jang-spec-iobench`, `jang-core`, `JANGCore`, `JANGCoreMetal` all build.

- [ ] **Step 3: Update STATUS.md**

Edit `docs/superpowers/notes/jang-spec-STATUS.md`:
- Plans table: mark Plan 4 **DONE** with branch `jang-spec-plan4-metal-matmul`, artifacts `JANGCoreMetal library, JangV2QuantMatmul.metal, QuantizedMatmul4`
- Tests line: bump Swift count to include the new correctness test
- TL;DR: add a sentence confirming first Metal compute kernel validated against MLX reference
- Immediate next: describe Plan 5 (dense v2 forward pass on a small dense JANG model, using QuantizedMatmul4 for all linears)
- Add a Plan 4 notes subsection with measured max abs/rel error from the test and any deviations

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/jang && git add docs/superpowers/notes/jang-spec-STATUS.md
git commit -m "jang-spec: update STATUS after Plan 4 completion"
```

- [ ] **Step 5: Print commit log**

```bash
cd /Users/eric/jang && git log --oneline jang-spec-plan3-hotcore..HEAD
```
Expected: 7–8 commits from this plan.
