# Plan 3 — Hot-Core Loader (JANGCore, pure Swift)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `JANGCore` with a pure-Swift reader for JANG v2 MLX-native safetensors files and a `HotCoreLoader` that opens a bundle's `target/hot_core.safetensors`, classifies each base name into (quantized triple) vs (raw tensor), infers the per-tensor bit width from the qweight shape, and returns typed Swift handles. Ship a `jang-core hot-core` CLI subcommand that prints a summary table. Zero Metal code in this plan — Plan 4 takes the tensor handles and dispatches Metal kernels.

**Architecture:** Small, single-responsibility files under `Sources/JANGCore/`. The v2 safetensors reader is standalone (no dependency on the existing v1 `SafetensorsReader` in the `JANG` target) and is pure `Foundation` + `Data`. The `HotCoreLoader` consumes it and produces a `HotCoreTensors` struct containing a dict of base-name → handle. Tests validate synthetic files and then the real MiniMax-M2.7-JANG_2L bundle end to end.

**Tech Stack:** Swift 6.0, SwiftPM, `Foundation` only. `@testable import JANGCore`. No Metal dependency in `JANGCore` yet; Plan 4 introduces it.

**Spec:** `docs/superpowers/specs/2026-04-13-jang-spec-design.md` §5.1 (hot core definition), §8.1 (JANGCore target). JANG v2 format: `FORMAT.md` in the repo root (`.weight` uint32, `.scales` + `.biases` float16, mixed bit widths inferred from shape ratios).

**Depends on:** Plans 1 + 2 merged/available. This plan branches from `jang-spec-plan2-swift-loader`.

**Out of scope:**
- Metal buffers, Metal device handling, Metal kernels — Plan 4.
- Dense v2 forward pass — Plan 5.
- Router, shared-expert loading — Plan 6.
- Expert streaming — Plan 7.

**Test fixtures:**
- Synthetic safetensors files written at runtime in `tmp_path` for unit tests.
- **MiniMax-M2.7-JANG_2L bundle at `/Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec`** for the integration test. This is the real 228 B model, 62 layers, 4.03 GB hot core — perfect coverage. Fall back to the Gemma bundle at `/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec` if MiniMax is missing (shouldn't happen on this machine).

---

## File structure

New files:

```
jang-runtime/Sources/JANGCore/
  SafetensorsV2.swift         Pure-Foundation v2 safetensors reader
  BitInference.swift          Infer per-tensor bit width from qweight+scales shapes
  QuantizedTensorView.swift   Grouped triple (qweight, scales, biases) + bits + shape
  HotCoreLoader.swift         Open hot_core.safetensors, return HotCoreTensors

jang-runtime/Sources/jang-core/
  main.swift                  MODIFY: add `hot-core` subcommand

jang-runtime/Tests/JANGCoreTests/
  SafetensorsV2Tests.swift    Synthetic file parse + dtype/shape/offset checks
  BitInferenceTests.swift     Unit test the bit math
  HotCoreLoaderTests.swift    Integration against MiniMax bundle
```

No other files modified.

---

## Task 0: Branch setup

**Files:** none

- [ ] **Step 1: Confirm starting state**

```bash
cd /Users/eric/jang && git status && git log -1 --oneline && git branch --show-current
```
Expected: clean tree, current branch `jang-spec-plan2-swift-loader`, latest commit is the final Plan 2 commit (`e0f4efa jang-core: jang-core inspect subcommand with Python parity` or newer).

- [ ] **Step 2: Create Plan 3 branch**

```bash
git checkout -b jang-spec-plan3-hotcore
```
Expected: `Switched to a new branch 'jang-spec-plan3-hotcore'`.

---

## Task 1: Pure-Swift v2 safetensors reader (TDD)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/SafetensorsV2.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/SafetensorsV2Tests.swift`

**Background.** A safetensors file is:
```
[8 bytes: header_size, uint64 LE]
[header_size bytes: JSON object { tensor_name: { dtype, shape, data_offsets: [start, end] }, ... }]
[data section: concatenated tensor bytes]
```
JANG v2 uses dtypes `U32` for packed qweight, `F16` for scales/biases/norms. Everything is little-endian and byte-tight.

- [ ] **Step 1: Write the failing test**

Write `jang-runtime/Tests/JANGCoreTests/SafetensorsV2Tests.swift`:
```swift
import XCTest
@testable import JANGCore

final class SafetensorsV2Tests: XCTestCase {
    /// Build a tiny synthetic safetensors file with 2 tensors.
    private func writeSynthetic(to url: URL) throws {
        // Tensor A: "alpha" — U32 shape [2, 3], 6 * 4 = 24 bytes
        // Tensor B: "beta"  — F16 shape [4],    4 * 2 = 8 bytes
        let alphaBytes: [UInt8] = [
            0x01, 0x00, 0x00, 0x00,
            0x02, 0x00, 0x00, 0x00,
            0x03, 0x00, 0x00, 0x00,
            0x04, 0x00, 0x00, 0x00,
            0x05, 0x00, 0x00, 0x00,
            0x06, 0x00, 0x00, 0x00,
        ]
        let betaBytes: [UInt8] = [0x00, 0x3C, 0x00, 0x40, 0x00, 0x42, 0x00, 0x44]
        // offsets in the data section:
        //   alpha: [0, 24]
        //   beta:  [24, 32]
        let headerObj: [String: Any] = [
            "alpha": [
                "dtype": "U32",
                "shape": [2, 3],
                "data_offsets": [0, 24],
            ],
            "beta": [
                "dtype": "F16",
                "shape": [4],
                "data_offsets": [24, 32],
            ],
        ]
        let headerJSON = try JSONSerialization.data(
            withJSONObject: headerObj, options: [.sortedKeys]
        )
        var out = Data()
        var headerSize = UInt64(headerJSON.count)
        withUnsafeBytes(of: &headerSize) { out.append(contentsOf: $0) }
        out.append(headerJSON)
        out.append(contentsOf: alphaBytes)
        out.append(contentsOf: betaBytes)
        try out.write(to: url)
    }

    func testParsesSyntheticFile() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("stv2-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try writeSynthetic(to: tmp)

        let file = try SafetensorsV2File(url: tmp)
        XCTAssertEqual(file.tensorNames.sorted(), ["alpha", "beta"])

        let alpha = try file.info(for: "alpha")
        XCTAssertEqual(alpha.dtype, .u32)
        XCTAssertEqual(alpha.shape, [2, 3])
        XCTAssertEqual(alpha.dataLength, 24)

        let beta = try file.info(for: "beta")
        XCTAssertEqual(beta.dtype, .f16)
        XCTAssertEqual(beta.shape, [4])
        XCTAssertEqual(beta.dataLength, 8)

        // Actual bytes match what we wrote.
        let alphaBytes = try file.bytes(for: "alpha")
        XCTAssertEqual(alphaBytes.count, 24)
        alphaBytes.withUnsafeBytes { raw in
            XCTAssertEqual(raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self), 1)
            XCTAssertEqual(raw.loadUnaligned(fromByteOffset: 4, as: UInt32.self), 2)
            XCTAssertEqual(raw.loadUnaligned(fromByteOffset: 20, as: UInt32.self), 6)
        }
    }

    func testMissingTensorThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("stv2-missing-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try writeSynthetic(to: tmp)

        let file = try SafetensorsV2File(url: tmp)
        XCTAssertThrowsError(try file.info(for: "nope")) { err in
            guard case SafetensorsV2Error.missingTensor = err else {
                XCTFail("expected missingTensor, got \(err)")
                return
            }
        }
    }

    func testTruncatedFileThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("stv2-short-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try Data([0x01, 0x02]).write(to: tmp)

        XCTAssertThrowsError(try SafetensorsV2File(url: tmp)) { err in
            guard case SafetensorsV2Error.truncated = err else {
                XCTFail("expected truncated, got \(err)")
                return
            }
        }
    }
}
```

- [ ] **Step 2: Run and verify it fails**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter SafetensorsV2Tests 2>&1 | tail -10
```
Expected: compile error — `SafetensorsV2File` not in scope.

- [ ] **Step 3: Implement SafetensorsV2.swift**

Write `jang-runtime/Sources/JANGCore/SafetensorsV2.swift`:
```swift
import Foundation

/// Errors raised by `SafetensorsV2File`.
public enum SafetensorsV2Error: Error, CustomStringConvertible {
    case truncated(URL, expected: Int, actual: Int)
    case malformedHeader(URL, reason: String)
    case unknownDType(String)
    case missingTensor(String)

    public var description: String {
        switch self {
        case .truncated(let url, let e, let a):
            return "safetensors: truncated \(url.lastPathComponent): expected \(e) bytes, got \(a)"
        case .malformedHeader(let url, let r):
            return "safetensors: malformed header in \(url.lastPathComponent): \(r)"
        case .unknownDType(let s):
            return "safetensors: unknown dtype \(s)"
        case .missingTensor(let name):
            return "safetensors: missing tensor '\(name)'"
        }
    }
}

/// Subset of safetensors dtypes that JANG v2 emits.
public enum SafetensorsDType: String, Sendable {
    case u32 = "U32"
    case f16 = "F16"
    case bf16 = "BF16"
    case f32 = "F32"

    public var byteWidth: Int {
        switch self {
        case .u32: return 4
        case .f16, .bf16: return 2
        case .f32: return 4
        }
    }
}

/// Metadata for one tensor inside a safetensors file.
public struct SafetensorsV2Info: Sendable, Equatable {
    public let name: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    public let dataOffset: Int    // absolute byte offset in the file
    public let dataLength: Int    // byte length of the tensor payload
}

/// A mmap'd JANG v2 safetensors file.
///
/// Construction parses the JSON header and records per-tensor metadata.
/// Actual bytes are returned as `Data` slices that share storage with the
/// mmap'd backing — zero-copy on Apple Silicon unified memory.
public final class SafetensorsV2File: @unchecked Sendable {
    public let url: URL
    public let tensorNames: [String]

    private let mapped: Data
    private let dataSectionStart: Int
    private let infoByName: [String: SafetensorsV2Info]

    public init(url: URL) throws {
        self.url = url
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        self.mapped = data

        guard data.count >= 8 else {
            throw SafetensorsV2Error.truncated(url, expected: 8, actual: data.count)
        }
        let headerSize: UInt64 = data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: 0, as: UInt64.self)
        }
        let headerStart = 8
        let headerEnd = headerStart + Int(headerSize)
        guard data.count >= headerEnd else {
            throw SafetensorsV2Error.truncated(url, expected: headerEnd, actual: data.count)
        }
        self.dataSectionStart = headerEnd

        let headerJSON = data.subdata(in: headerStart..<headerEnd)
        let anyObj = try JSONSerialization.jsonObject(
            with: headerJSON, options: [.fragmentsAllowed]
        )
        guard let obj = anyObj as? [String: Any] else {
            throw SafetensorsV2Error.malformedHeader(url, reason: "top level is not an object")
        }

        var out: [String: SafetensorsV2Info] = [:]
        out.reserveCapacity(obj.count)
        var names: [String] = []

        for (name, raw) in obj {
            // safetensors may include a "__metadata__" key; skip it.
            if name == "__metadata__" { continue }

            guard let entry = raw as? [String: Any] else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) is not an object"
                )
            }
            guard let dtypeString = entry["dtype"] as? String else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) missing dtype"
                )
            }
            guard let dtype = SafetensorsDType(rawValue: dtypeString) else {
                throw SafetensorsV2Error.unknownDType(dtypeString)
            }
            guard let shapeAny = entry["shape"] as? [Any] else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) missing shape"
                )
            }
            let shape = shapeAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard shape.count == shapeAny.count else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) shape contains non-integers"
                )
            }
            guard let offsetsAny = entry["data_offsets"] as? [Any], offsetsAny.count == 2 else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) missing data_offsets"
                )
            }
            let offsets = offsetsAny.compactMap { ($0 as? NSNumber)?.intValue }
            guard offsets.count == 2 else {
                throw SafetensorsV2Error.malformedHeader(
                    url, reason: "entry \(name) data_offsets not integers"
                )
            }
            let absStart = headerEnd + offsets[0]
            let absEnd = headerEnd + offsets[1]
            guard absEnd <= data.count else {
                throw SafetensorsV2Error.truncated(url, expected: absEnd, actual: data.count)
            }
            out[name] = SafetensorsV2Info(
                name: name,
                dtype: dtype,
                shape: shape,
                dataOffset: absStart,
                dataLength: absEnd - absStart
            )
            names.append(name)
        }

        self.infoByName = out
        self.tensorNames = names.sorted()
    }

    public func info(for name: String) throws -> SafetensorsV2Info {
        guard let i = infoByName[name] else {
            throw SafetensorsV2Error.missingTensor(name)
        }
        return i
    }

    /// Return a `Data` slice pointing at the tensor bytes. Zero-copy over the
    /// mmap'd backing; the slice lifetime is bounded by the file object.
    public func bytes(for name: String) throws -> Data {
        let info = try info(for: name)
        return mapped.subdata(in: info.dataOffset..<(info.dataOffset + info.dataLength))
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter SafetensorsV2Tests 2>&1 | tail -15
```
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/SafetensorsV2.swift
git add -f jang-runtime/Tests/JANGCoreTests/SafetensorsV2Tests.swift
git commit -m "jang-core: pure-Swift v2 safetensors reader with synthetic round-trip"
```

Note: `git add -f` is required because the global `.gitignore` has `test_*`.

---

## Task 2: Bit inference (TDD)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/BitInference.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/BitInferenceTests.swift`

**Background.** JANG v2 does not store a per-tensor bit field. The bit width is inferred from the shape ratio of `qweight` and `scales`:
```
packed_in_features = qweight.shape[-1]         // uint32 count per row
n_groups           = scales.shape[-1]           // number of quant groups per row
in_features        = n_groups * group_size
bits               = (packed_in_features * 32) / in_features
```
Group size defaults to 64 but can be any power of 2. For MoE experts (3D tensors), the logic is identical, just applied per-expert slab.

- [ ] **Step 1: Write the failing test**

Write `jang-runtime/Tests/JANGCoreTests/BitInferenceTests.swift`:
```swift
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
```

- [ ] **Step 2: Run and verify failure**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter BitInferenceTests 2>&1 | tail -10
```
Expected: compile error — `BitInference` not in scope.

- [ ] **Step 3: Implement BitInference.swift**

Write `jang-runtime/Sources/JANGCore/BitInference.swift`:
```swift
import Foundation

/// Infers per-tensor JANG v2 bit widths from the qweight+scales shape pair.
public enum BitInference {

    public struct Result: Sendable, Equatable {
        public let bits: Int
        public let inFeatures: Int
        public let outFeatures: Int   // product of all leading dims except the last
    }

    /// Return the inferred bits/inFeatures/outFeatures, or nil if the shapes
    /// don't line up. Handles both 2D (dense linears) and 3D (MoE expert
    /// stacks) — the rule looks only at the last two axes.
    public static func infer(
        qweightShape: [Int],
        scalesShape: [Int],
        groupSize: Int
    ) -> Result? {
        guard qweightShape.count >= 2, scalesShape.count >= 2 else { return nil }
        guard qweightShape.count == scalesShape.count else { return nil }

        // All leading dims must match (batch / expert axes).
        if qweightShape.count > 2 {
            let qLeading = qweightShape.dropLast(2)
            let sLeading = scalesShape.dropLast(2)
            if Array(qLeading) != Array(sLeading) { return nil }
        }

        // The "out" dim (second to last) must match between qweight and scales.
        let qOut = qweightShape[qweightShape.count - 2]
        let sOut = scalesShape[scalesShape.count - 2]
        guard qOut == sOut else { return nil }

        let packedIn = qweightShape.last!
        let nGroups = scalesShape.last!
        guard packedIn > 0, nGroups > 0, groupSize > 0 else { return nil }

        let inFeatures = nGroups * groupSize
        // packedIn * 32 = inFeatures * bits, so bits = packedIn * 32 / inFeatures.
        let numerator = packedIn * 32
        guard numerator % inFeatures == 0 else { return nil }
        let bits = numerator / inFeatures

        let outFeatures: Int
        if qweightShape.count == 2 {
            outFeatures = qOut
        } else {
            outFeatures = qweightShape.dropLast(2).reduce(1, *) * qOut
        }

        return Result(bits: bits, inFeatures: inFeatures, outFeatures: outFeatures)
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter BitInferenceTests 2>&1 | tail -15
```
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/BitInference.swift
git add -f jang-runtime/Tests/JANGCoreTests/BitInferenceTests.swift
git commit -m "jang-core: BitInference — infer v2 bit width from qweight+scales shapes"
```

---

## Task 3: QuantizedTensorView

**Files:**
- Create: `jang-runtime/Sources/JANGCore/QuantizedTensorView.swift`

- [ ] **Step 1: Write QuantizedTensorView.swift**

Write `jang-runtime/Sources/JANGCore/QuantizedTensorView.swift`:
```swift
import Foundation

/// A typed handle over one JANG v2 quantized tensor.
///
/// Holds three `Data` slices (qweight, scales, biases) that reference the
/// mmap'd backing of a safetensors file, plus the inferred bit width and
/// logical shape. No Metal buffers yet — that's Plan 4. The slices are
/// zero-copy views on the mmap.
public struct QuantizedTensorView: Sendable {
    public let baseName: String
    public let bits: Int
    public let groupSize: Int
    public let inFeatures: Int
    public let outFeatures: Int
    public let qweightShape: [Int]    // e.g. [out, packedIn] or [E, out, packedIn]
    public let scalesShape: [Int]     // e.g. [out, nGroups] or [E, out, nGroups]
    public let qweight: Data          // U32 payload
    public let scales: Data           // F16 payload
    public let biases: Data           // F16 payload

    public var isExpertStacked: Bool {
        return qweightShape.count == 3
    }
}

/// A typed handle over one non-quantized tensor (norms, biases, embeddings).
public struct RawTensorView: Sendable {
    public let name: String
    public let dtype: SafetensorsDType
    public let shape: [Int]
    public let bytes: Data
}
```

- [ ] **Step 2: Build to confirm it compiles**

```bash
cd /Users/eric/jang/jang-runtime && swift build 2>&1 | tail -5
```
Expected: build success.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/QuantizedTensorView.swift
git commit -m "jang-core: QuantizedTensorView and RawTensorView typed handles"
```

---

## Task 4: HotCoreLoader (integration test against MiniMax)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/HotCoreLoader.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/HotCoreLoaderTests.swift`

- [ ] **Step 1: Write the failing integration test**

Write `jang-runtime/Tests/JANGCoreTests/HotCoreLoaderTests.swift`:
```swift
import XCTest
@testable import JANGCore

final class HotCoreLoaderTests: XCTestCase {
    /// Prefer the MiniMax bundle for this test; fall back to the Gemma
    /// fixture if MiniMax isn't present. Skip entirely if neither is there.
    private func bundleURL() throws -> URL {
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
            bundleURL = try bundleURL()
        } catch {
            throw XCTSkip("no real bundle available: \(error)")
        }
        let bundle = try JangSpecBundle.open(at: bundleURL)

        let hot = try HotCoreLoader.load(bundle: bundle, groupSize: 64)

        // Every base name listed in the manifest must be accounted for.
        let accounted = Set(hot.quantized.keys) .union(hot.raw.keys)
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
}
```

- [ ] **Step 2: Run and verify failure**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter HotCoreLoaderTests 2>&1 | tail -10
```
Expected: compile error — `HotCoreLoader` not in scope.

- [ ] **Step 3: Implement HotCoreLoader.swift**

Write `jang-runtime/Sources/JANGCore/HotCoreLoader.swift`:
```swift
import Foundation

/// A loaded JANG v2 hot core — all tensors that stay resident in RAM:
/// embeddings, lm_head, attention q/k/v/o, routers, shared experts, norms.
public struct HotCoreTensors: Sendable {
    public let quantized: [String: QuantizedTensorView]   // base name -> handle
    public let raw: [String: RawTensorView]               // tensor name -> handle
}

/// Load `target/hot_core.safetensors` from a bundle.
///
/// The loader groups every `{base}.weight` + `{base}.scales` + `{base}.biases`
/// triple into a `QuantizedTensorView` and classifies everything else as a
/// `RawTensorView` (norms, biases without a scales partner, etc). The `File`
/// instance is held by the returned views via the `Data` backing of the
/// mmap; the file stays alive for as long as any view retains a slice of it.
public enum HotCoreLoader {
    public static func load(bundle: JangSpecBundle, groupSize: Int = 64) throws -> HotCoreTensors {
        let fileURL = bundle.hotCoreURL
        let file = try SafetensorsV2File(url: fileURL)

        var quantized: [String: QuantizedTensorView] = [:]
        var raw: [String: RawTensorView] = [:]

        // Group tensors by base name.
        // A base name is everything before a trailing .weight/.scales/.biases.
        var byBase: [String: [String]] = [:]   // base -> full names
        var unrelated: [String] = []

        for name in file.tensorNames {
            if let (base, _) = Self.splitSuffix(name) {
                byBase[base, default: []].append(name)
            } else {
                unrelated.append(name)
            }
        }

        // Group with all three (weight, scales, biases) -> quantized.
        // Otherwise -> raw tensors, one entry per tensor.
        for (base, names) in byBase {
            let set = Set(names.compactMap { Self.splitSuffix($0)?.1 })
            if set.contains("weight") && set.contains("scales") && set.contains("biases") {
                let qInfo = try file.info(for: "\(base).weight")
                let sInfo = try file.info(for: "\(base).scales")
                let bInfo = try file.info(for: "\(base).biases")

                guard let bitInfo = BitInference.infer(
                    qweightShape: qInfo.shape,
                    scalesShape: sInfo.shape,
                    groupSize: groupSize
                ) else {
                    throw JangSpecError.invalidManifest(
                        "cannot infer bits for '\(base)' (qweight=\(qInfo.shape), scales=\(sInfo.shape))"
                    )
                }

                quantized[base] = QuantizedTensorView(
                    baseName: base,
                    bits: bitInfo.bits,
                    groupSize: groupSize,
                    inFeatures: bitInfo.inFeatures,
                    outFeatures: bitInfo.outFeatures,
                    qweightShape: qInfo.shape,
                    scalesShape: sInfo.shape,
                    qweight: try file.bytes(for: "\(base).weight"),
                    scales: try file.bytes(for: "\(base).scales"),
                    biases: try file.bytes(for: "\(base).biases")
                )
            } else {
                // Incomplete triple — emit each tensor as a raw view.
                for n in names {
                    let info = try file.info(for: n)
                    raw[n] = RawTensorView(
                        name: n,
                        dtype: info.dtype,
                        shape: info.shape,
                        bytes: try file.bytes(for: n)
                    )
                }
            }
        }

        // Unrelated (no suffix) tensors are raw.
        for n in unrelated {
            let info = try file.info(for: n)
            raw[n] = RawTensorView(
                name: n,
                dtype: info.dtype,
                shape: info.shape,
                bytes: try file.bytes(for: n)
            )
        }

        return HotCoreTensors(quantized: quantized, raw: raw)
    }

    /// Split a name like "layers.0.self_attn.q_proj.scales" into
    /// ("layers.0.self_attn.q_proj", "scales"). Returns nil if there's no
    /// recognized trailing suffix.
    private static func splitSuffix(_ name: String) -> (String, String)? {
        for s in ["weight", "scales", "biases"] {
            let dot = ".\(s)"
            if name.hasSuffix(dot) {
                return (String(name.dropLast(dot.count)), s)
            }
        }
        return nil
    }
}
```

- [ ] **Step 4: Run the integration test**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter HotCoreLoaderTests 2>&1 | tail -20
```
Expected: 1 test passes. The MiniMax hot core is 4.03 GB but mmap is zero-copy so the test should complete in a few seconds. If it fails with a missing base name or bit-inference failure on a specific tensor, capture the tensor name and report BLOCKED — that means the manifest includes a name the loader doesn't classify cleanly.

- [ ] **Step 5: Full JANGCore sweep**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter JANGCoreTests 2>&1 | tail -30
```
Expected: all previous JANGCore tests + the new SafetensorsV2/BitInference/HotCoreLoader tests pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/HotCoreLoader.swift
git add -f jang-runtime/Tests/JANGCoreTests/HotCoreLoaderTests.swift
git commit -m "jang-core: HotCoreLoader — classify quantized triples and raw tensors"
```

---

## Task 5: `jang-core hot-core` subcommand

**Files:**
- Modify: `jang-runtime/Sources/jang-core/main.swift`

- [ ] **Step 1: Add the subcommand**

Edit `jang-runtime/Sources/jang-core/main.swift`. Keep the existing `Inspect` subcommand. Add a new `HotCore` subcommand to the `JangCore` `CommandConfiguration(subcommands:)` list, and define it at the bottom of the file:

```swift
struct HotCore: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "hot-core",
        abstract: "Print a per-tensor summary of a bundle's hot core."
    )

    @Argument(help: "Path to a .jangspec directory.")
    var bundle: String

    @Option(name: .long, help: "Quant group size (default 64).")
    var groupSize: Int = 64

    func run() async throws {
        let url = URL(fileURLWithPath: (bundle as NSString).expandingTildeInPath).resolvingSymlinksInPath()
        let spec = try JangSpecBundle.open(at: url)
        let hot = try HotCoreLoader.load(bundle: spec, groupSize: groupSize)

        print("  bundle:     \(url.path)")
        print("  quantized:  \(hot.quantized.count) base tensors")
        print("  raw:        \(hot.raw.count) tensors")

        // Histogram of bit widths.
        var bitCounts: [Int: Int] = [:]
        var bitBytes: [Int: Int] = [:]
        for q in hot.quantized.values {
            bitCounts[q.bits, default: 0] += 1
            bitBytes[q.bits, default: 0] += q.qweight.count + q.scales.count + q.biases.count
        }
        print("")
        print("  bit distribution:")
        for b in bitCounts.keys.sorted() {
            let cnt = bitCounts[b]!
            let gb = Double(bitBytes[b]!) / 1e9
            print(String(format: "    %d-bit: %5d tensors, %.2f GB", b, cnt, gb))
        }

        // Totals.
        let totalQBytes = hot.quantized.values.reduce(0) { $0 + $1.qweight.count + $1.scales.count + $1.biases.count }
        let totalRBytes = hot.raw.values.reduce(0) { $0 + $1.bytes.count }
        print("")
        print(String(format: "  total quantized: %.2f GB", Double(totalQBytes) / 1e9))
        print(String(format: "  total raw:       %.2f GB", Double(totalRBytes) / 1e9))
        print(String(format: "  total hot core:  %.2f GB", Double(totalQBytes + totalRBytes) / 1e9))
    }
}
```

And update `JangCore.configuration.subcommands` to include `HotCore.self`:
```swift
subcommands: [Inspect.self, HotCore.self]
```

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build -c release --product jang-core 2>&1 | tail -5
```
Expected: build success.

- [ ] **Step 3: Run against the MiniMax bundle**

```bash
cd /Users/eric/jang/jang-runtime && ./.build/release/jang-core hot-core /Users/eric/models/MiniMax-M2.7-JANG_2L.jangspec 2>&1
```
Expected: per-bit-width table and totals. The sum of quantized + raw should be close to the 4.03 GB reported by `jang-core inspect`.

- [ ] **Step 4: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/jang-core/main.swift
git commit -m "jang-core: jang-core hot-core subcommand for bit histogram"
```

---

## Task 6: Final sweep + STATUS update

**Files:**
- Modify: `docs/superpowers/notes/jang-spec-STATUS.md`

- [ ] **Step 1: Full Swift test suite**

```bash
cd /Users/eric/jang/jang-runtime && swift test 2>&1 | tail -15
```
Expected: all `JANGCoreTests` + existing `JANGTests` pass.

- [ ] **Step 2: Full release build**

```bash
cd /Users/eric/jang/jang-runtime && swift build -c release 2>&1 | tail -10
```
Expected: all products build.

- [ ] **Step 3: Update STATUS.md**

Edit `docs/superpowers/notes/jang-spec-STATUS.md`:
- Change Plan 3's row in the plan table to `**DONE**` with branch `jang-spec-plan3-hotcore`
- Add Plan 3 artifacts: "`HotCoreLoader`, `jang-core hot-core` CLI"
- Under "Tests", bump the Swift count to reflect the new XCTest cases
- Under "Immediate next", change Plan 3 prose to describe Plan 4 (Metal v2 quantized matmul kernel) as the next step

- [ ] **Step 4: Commit the STATUS update**

```bash
cd /Users/eric/jang && git add docs/superpowers/notes/jang-spec-STATUS.md
git commit -m "jang-spec: update STATUS after Plan 3 completion"
```

- [ ] **Step 5: Print the commit log for the plan**

```bash
cd /Users/eric/jang && git log --oneline jang-spec-plan2-swift-loader..HEAD
```
Expected: the ~6 Plan 3 commits in order.
