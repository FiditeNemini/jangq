# Plan 2 — Swift Bundle Loader (JANGCore)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new `JANGCore` SwiftPM library + `jang-core` CLI that can load a `.jangspec` bundle (built by Plan 1's Python tooling), parse the manifest and expert index, and read individual expert blobs by `(layer_idx, expert_id)` — all in pure Swift with zero-copy mmap. No Metal kernels, no forward pass; those belong to Plan 3.

**Architecture:** Small, single-responsibility Swift files inside `Sources/JANGCore/`. Each file parses exactly one on-disk structure defined by Plan 1. A `JangSpecBundle` façade assembles them and exposes a small, test-driven API. A `jang-core` executable provides an `inspect` subcommand that reproduces Python's `jang spec inspect` output in Swift, proving round-trip parity.

**Tech Stack:** Swift 6.0, SwiftPM, macOS 15+, Foundation only (no Metal in this plan). Tests via XCTest. Fixture bundle is built on-the-fly from the existing Gemma-4-26B JANG model using the Plan 1 Python builder.

**Spec:** `docs/superpowers/specs/2026-04-13-jang-spec-design.md` §5 (bundle format), §8.1 (JANGCore target).

**Depends on:** Plan 1 complete (merged to `main` or available on the current working branch).

**Out of scope:**
- Metal kernels, dequant, matmul, forward pass — Plan 3.
- Expert streaming via `MTLIOCommandQueue`. Plan 2 reads experts via `mmap` + `Data` slicing, which is exactly equivalent on Apple Silicon unified memory and simpler to unit-test. Plan 3 swaps in `MTLIOCommandQueue` for the hot path.
- Tokenizer, sampler, KV cache — later plans.
- Dense v1 JANG compatibility — Plan 2 is v2-bundle-only.

**Test fixture:** The Gemma-4-26B `.jangspec` bundle built by Plan 1. Tests use a session-scoped fixture that invokes the Python builder once per test run and caches the bundle in a tmp directory.

---

## File structure

New files:

```
jang-runtime/
  Package.swift                                  MODIFY: add JANGCore library, JangCoreCLI executable, JANGCoreTests

  Sources/JANGCore/
    JANGCore.swift                               umbrella, re-exports public API
    JangSpecError.swift                          Error enum
    JangSpecFormat.swift                         Mirror of Python jangspec/format.py constants
    JangSpecManifest.swift                       Codable struct for jangspec.json
    ExpertIndex.swift                            Parses experts.jsidx into a flat array + lookup
    ExpertBlob.swift                             Parses one ExpertBlob: header, per-tensor headers, payload slices
    ExpertStore.swift                            Lazy mmap over experts-*.bin shards, load(layer, expert) -> ExpertBlob
    JangSpecBundle.swift                         Façade: open(bundleURL) -> { manifest, index, store, hotCoreURL }

  Sources/jang-core/
    main.swift                                   `jang-core inspect <bundle>` subcommand

  Tests/JANGCoreTests/
    JangSpecFormatTests.swift                    Sanity checks on struct sizes and constants
    ExpertIndexTests.swift                       Synthetic index, round-trip lookups
    ExpertBlobTests.swift                        Parse a synthetic blob
    JangSpecBundleTests.swift                    Integration: open the Gemma fixture bundle, verify fields, read expert (0,0) bytes against the Python reader's bytes
    Fixtures.swift                               Session-scoped Gemma bundle builder (shells out to `jang spec build`)
```

Modified files:
- `jang-runtime/Package.swift` — add `JANGCore` library target, `jang-core` executable target, `JANGCoreTests` test target.

---

## Task 0: Branch setup

**Files:** none (git state only)

- [ ] **Step 1: Confirm starting state**

```bash
cd /Users/eric/jang && git status && git log -1 --oneline
```

Expected: clean tree, latest commit is Plan 1's last commit (either `89dd2a0 README for the jangspec Python subpackage` or later if the branch has been merged to main). If uncertain, run `git branch` to see the current branch.

- [ ] **Step 2: Create a branch for this plan**

```bash
git checkout -b jang-spec-plan2-swift-loader
```

Expected: `Switched to a new branch 'jang-spec-plan2-swift-loader'`.

- [ ] **Step 3: Scaffold directories**

```bash
mkdir -p jang-runtime/Sources/JANGCore \
         jang-runtime/Sources/jang-core \
         jang-runtime/Tests/JANGCoreTests
```

Expected: no output.

---

## Task 1: Package.swift — register targets

**Files:**
- Modify: `jang-runtime/Package.swift`

- [ ] **Step 1: Edit Package.swift**

Replace the entire file with:

```swift
// swift-tools-version: 6.0
// JANG Runtime — Jang Adaptive N-bit Grading
// Created by Eric Jang (eric@vmlx.net)

import PackageDescription

let package = Package(
    name: "JANGRuntime",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .executable(name: "jang", targets: ["JANGCLI"]),
        .executable(name: "jang-spec-iobench", targets: ["JangSpecIOBench"]),
        .executable(name: "jang-core", targets: ["JangCoreCLI"]),
        .library(name: "JANG", targets: ["JANG"]),
        .library(name: "JANGCore", targets: ["JANGCore"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(name: "JANGMetal", dependencies: [], path: "Sources/JANGMetal"),
        .target(name: "JANG", dependencies: ["JANGMetal"], path: "Sources/JANG"),
        .target(name: "JANGCore", dependencies: [], path: "Sources/JANGCore"),
        .executableTarget(
            name: "JANGCLI",
            dependencies: ["JANG", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/JANGCLI"
        ),
        .executableTarget(
            name: "JangSpecIOBench",
            dependencies: [],
            path: "Sources/jang-spec-iobench"
        ),
        .executableTarget(
            name: "JangCoreCLI",
            dependencies: ["JANGCore", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/jang-core"
        ),
        .testTarget(name: "JANGTests", dependencies: ["JANG"], path: "Tests/JANGTests"),
        .testTarget(name: "JANGCoreTests", dependencies: ["JANGCore"], path: "Tests/JANGCoreTests"),
    ]
)
```

- [ ] **Step 2: Verify the package resolves**

```bash
cd /Users/eric/jang/jang-runtime && swift package describe 2>&1 | head -20
```
Expected: describes the package including `JANGCore` and `jang-core` targets without errors. The JANGCore target will show "no source files" — that's fine at this point.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Package.swift
git commit -m "jang-core: register JANGCore library and jang-core CLI targets"
```

---

## Task 2: Format constants mirror (JangSpecFormat.swift)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/JangSpecFormat.swift`
- Create: `jang-runtime/Sources/JANGCore/JangSpecError.swift`
- Create: `jang-runtime/Sources/JANGCore/JANGCore.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/JangSpecFormatTests.swift`

- [ ] **Step 1: Write the umbrella file**

Write `jang-runtime/Sources/JANGCore/JANGCore.swift`:
```swift
//
// JANGCore — v2 JANG bundle loader in pure Swift.
// Created by Eric Jang (eric@jangq.ai).
//
// This module is format-focused: it parses a .jangspec bundle's
// manifest, expert index, and individual expert blobs. It does NOT
// dequantize, allocate Metal buffers, or run any inference. Those
// responsibilities belong to later plans.
//
// See docs/superpowers/specs/2026-04-13-jang-spec-design.md §5, §8.1.
//

import Foundation

public enum JANGCore {
    public static let version = "0.1.0"
}
```

- [ ] **Step 2: Write the error type**

Write `jang-runtime/Sources/JANGCore/JangSpecError.swift`:
```swift
import Foundation

/// Errors raised by JANGCore when parsing a .jangspec bundle.
public enum JangSpecError: Error, CustomStringConvertible {
    case fileMissing(URL)
    case badMagic(expected: UInt32, actual: UInt32, at: URL)
    case unsupportedVersion(field: String, value: Int, supported: Int)
    case truncated(URL, expected: Int, actual: Int)
    case missingEntry(layer: Int, expert: Int)
    case invalidManifest(String)
    case invalidBlob(String)

    public var description: String {
        switch self {
        case .fileMissing(let url):
            return "jangspec: file missing: \(url.path)"
        case .badMagic(let expected, let actual, let url):
            return String(
                format: "jangspec: bad magic 0x%08x (expected 0x%08x) in %@",
                actual, expected, url.lastPathComponent
            )
        case .unsupportedVersion(let field, let value, let supported):
            return "jangspec: unsupported \(field) version \(value), this build supports \(supported)"
        case .truncated(let url, let expected, let actual):
            return "jangspec: truncated file \(url.lastPathComponent): expected \(expected) bytes, got \(actual)"
        case .missingEntry(let layer, let expert):
            return "jangspec: no index entry for (layer=\(layer), expert=\(expert))"
        case .invalidManifest(let msg):
            return "jangspec: invalid manifest: \(msg)"
        case .invalidBlob(let msg):
            return "jangspec: invalid expert blob: \(msg)"
        }
    }
}
```

- [ ] **Step 3: Write the format constants**

Write `jang-runtime/Sources/JANGCore/JangSpecFormat.swift`:
```swift
import Foundation

/// Mirror of `jang_tools.jangspec.format` — on-disk layout constants.
///
/// If you change a value here, change it in the Python module and in the
/// bundle version number. The Swift and Python sides MUST agree.
public enum JangSpecFormat {
    public static let bundleVersion: Int = 1

    // Filenames inside a <name>.jangspec directory.
    public static let manifestFilename = "jangspec.json"
    public static let indexFilename = "target/experts.jsidx"
    public static let hotCoreFilename = "target/hot_core.safetensors"
    public static func expertFilename(idx: Int) -> String {
        return String(format: "target/experts-%05d.bin", idx)
    }

    // Alignment used for expert blob offsets.
    public static let blobAlignment: Int = 4096

    // Magic numbers — "JSPE" and "SJIX" little-endian uint32.
    public static let blobMagic: UInt32 = 0x4550_534A
    public static let indexMagic: UInt32 = 0x58_494A_53

    // Struct sizes (verified at compile time by static asserts below and
    // at runtime by tests).
    public static let blobHeaderSize: Int = 32
    public static let tensorHeaderSize: Int = 36
    public static let indexEntrySize: Int = 28
    public static let indexHeaderSize: Int = 24

    // Tensor-kind enum (matches Python KIND_* constants).
    public enum TensorKind: UInt8 {
        case gate = 0
        case up = 1
        case down = 2
    }

    // Dtype enum (matches Python DTYPE_* constants).
    public enum TensorDType: UInt32 {
        case qweight = 0  // uint32 packed
        case scales = 1   // float16
        case biases = 2   // float16
    }

    @inlinable
    public static func alignUp(_ n: Int, to align: Int = blobAlignment) -> Int {
        return (n + align - 1) & ~(align - 1)
    }
}
```

- [ ] **Step 4: Write the failing test**

Write `jang-runtime/Tests/JANGCoreTests/JangSpecFormatTests.swift`:
```swift
import XCTest
@testable import JANGCore

final class JangSpecFormatTests: XCTestCase {
    func testStructSizesMatchPython() {
        // These values must match jang_tools.jangspec.format exactly.
        XCTAssertEqual(JangSpecFormat.blobHeaderSize, 32)
        XCTAssertEqual(JangSpecFormat.tensorHeaderSize, 36)
        XCTAssertEqual(JangSpecFormat.indexEntrySize, 28)
        XCTAssertEqual(JangSpecFormat.indexHeaderSize, 24)
    }

    func testMagicNumbers() {
        // "JSPE" little-endian = 0x4550534A
        XCTAssertEqual(JangSpecFormat.blobMagic, 0x4550_534A)
        // "SJIX" little-endian = 0x58494A53
        XCTAssertEqual(JangSpecFormat.indexMagic, 0x58_494A_53)
    }

    func testAlignUp() {
        XCTAssertEqual(JangSpecFormat.alignUp(0), 0)
        XCTAssertEqual(JangSpecFormat.alignUp(1), 4096)
        XCTAssertEqual(JangSpecFormat.alignUp(4095), 4096)
        XCTAssertEqual(JangSpecFormat.alignUp(4096), 4096)
        XCTAssertEqual(JangSpecFormat.alignUp(4097), 8192)
    }

    func testExpertFilename() {
        XCTAssertEqual(JangSpecFormat.expertFilename(idx: 0), "target/experts-00000.bin")
        XCTAssertEqual(JangSpecFormat.expertFilename(idx: 42), "target/experts-00042.bin")
    }
}
```

- [ ] **Step 5: Build and test**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter JangSpecFormatTests 2>&1 | tail -20
```
Expected: `Test Suite 'JangSpecFormatTests' passed` with 4 tests.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/ jang-runtime/Tests/JANGCoreTests/JangSpecFormatTests.swift
git commit -m "jang-core: format constants mirror with parity tests"
```

---

## Task 3: Expert index parser (ExpertIndex.swift)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/ExpertIndex.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/ExpertIndexTests.swift`

- [ ] **Step 1: Write the failing test**

Write `jang-runtime/Tests/JANGCoreTests/ExpertIndexTests.swift`:
```swift
import XCTest
@testable import JANGCore

final class ExpertIndexTests: XCTestCase {
    /// Build a synthetic index file matching the format Python writes.
    private func writeSynthetic(
        nLayers: Int,
        nExpertsPerLayer: Int,
        entries: [(layer: Int, expert: Int, file: Int, offset: Int, nbytes: Int)],
        to url: URL
    ) throws {
        var data = Data()

        // IndexHeader: <I magic, H version, H _pad, I n_layers, I n_experts_per_layer, Q n_entries>
        var magic: UInt32 = JangSpecFormat.indexMagic
        var version: UInt16 = 1
        var pad: UInt16 = 0
        var nL: UInt32 = UInt32(nLayers)
        var nE: UInt32 = UInt32(nExpertsPerLayer)
        var n: UInt64 = UInt64(entries.count)
        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &pad) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nL) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nE) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }

        // Each entry: <I layer, I expert, H file_id, H _pad, Q offset, Q nbytes>
        for e in entries {
            var l: UInt32 = UInt32(e.layer)
            var ex: UInt32 = UInt32(e.expert)
            var f: UInt16 = UInt16(e.file)
            var p: UInt16 = 0
            var off: UInt64 = UInt64(e.offset)
            var nb: UInt64 = UInt64(e.nbytes)
            withUnsafeBytes(of: &l) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &ex) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &f) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &p) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &off) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &nb) { data.append(contentsOf: $0) }
        }
        try data.write(to: url)
    }

    func testRoundTripSynthetic() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangcore-idx-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let path = tmp.appendingPathComponent("experts.jsidx")
        try writeSynthetic(
            nLayers: 2,
            nExpertsPerLayer: 2,
            entries: [
                (0, 0, 0, 0, 4096),
                (0, 1, 0, 4096, 8192),
                (1, 0, 1, 0, 4096),
                (1, 1, 1, 4096, 4096),
            ],
            to: path
        )

        let idx = try ExpertIndex(contentsOf: path)
        XCTAssertEqual(idx.nLayers, 2)
        XCTAssertEqual(idx.nExpertsPerLayer, 2)
        XCTAssertEqual(idx.entries.count, 4)

        let hit = try idx.entry(layer: 1, expert: 0)
        XCTAssertEqual(hit.fileID, 1)
        XCTAssertEqual(hit.offset, 0)
        XCTAssertEqual(hit.nbytes, 4096)

        let hit2 = try idx.entry(layer: 0, expert: 1)
        XCTAssertEqual(hit2.offset, 4096)
        XCTAssertEqual(hit2.nbytes, 8192)
    }

    func testLookupMissingThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangcore-idx-miss-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let path = tmp.appendingPathComponent("experts.jsidx")
        try writeSynthetic(nLayers: 1, nExpertsPerLayer: 1, entries: [(0, 0, 0, 0, 4096)], to: path)

        let idx = try ExpertIndex(contentsOf: path)
        XCTAssertThrowsError(try idx.entry(layer: 99, expert: 99)) { err in
            guard case JangSpecError.missingEntry = err else {
                XCTFail("expected missingEntry error, got \(err)")
                return
            }
        }
    }

    func testBadMagicThrows() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("jangcore-idx-badmagic-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let path = tmp.appendingPathComponent("bad.jsidx")
        try Data(count: JangSpecFormat.indexHeaderSize).write(to: path)

        XCTAssertThrowsError(try ExpertIndex(contentsOf: path)) { err in
            guard case JangSpecError.badMagic = err else {
                XCTFail("expected badMagic error, got \(err)")
                return
            }
        }
    }
}
```

- [ ] **Step 2: Run the failing test**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter ExpertIndexTests 2>&1 | tail -10
```
Expected: compile error `cannot find 'ExpertIndex' in scope`. That's the failing-test signal.

- [ ] **Step 3: Implement ExpertIndex.swift**

Write `jang-runtime/Sources/JANGCore/ExpertIndex.swift`:
```swift
import Foundation

/// One row of `experts.jsidx`. Mirrors Python `ExpertIndexEntry`.
public struct ExpertIndexEntry: Sendable, Equatable {
    public let layerIdx: Int
    public let expertID: Int
    public let fileID: Int       // index into experts-NNNNN.bin
    public let offset: Int       // absolute byte offset in that file
    public let nbytes: Int       // total blob length (aligned)
}

/// Loaded `experts.jsidx` — flat array of entries plus layer/expert counts.
///
/// Lookup is O(1) via a layer-major dictionary keyed by `(layer, expert)`.
/// The backing file is read once at init; this struct holds the decoded
/// entries only, not the file handle.
public struct ExpertIndex: Sendable {
    public let nLayers: Int
    public let nExpertsPerLayer: Int
    public let entries: [ExpertIndexEntry]
    private let byKey: [Int: ExpertIndexEntry]

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= JangSpecFormat.indexHeaderSize else {
            throw JangSpecError.truncated(
                url, expected: JangSpecFormat.indexHeaderSize, actual: data.count
            )
        }

        // Read header.
        let header = data.withUnsafeBytes { raw -> (UInt32, UInt16, UInt32, UInt32, UInt64) in
            let magic = raw.load(fromByteOffset: 0, as: UInt32.self)
            let version = raw.load(fromByteOffset: 4, as: UInt16.self)
            // 2 bytes pad at offset 6
            let nL = raw.load(fromByteOffset: 8, as: UInt32.self)
            let nE = raw.load(fromByteOffset: 12, as: UInt32.self)
            let n = raw.load(fromByteOffset: 16, as: UInt64.self)
            return (magic, version, nL, nE, n)
        }

        let (magic, version, nLRaw, nERaw, nRaw) = header
        guard magic == JangSpecFormat.indexMagic else {
            throw JangSpecError.badMagic(
                expected: JangSpecFormat.indexMagic, actual: magic, at: url
            )
        }
        guard version == 1 else {
            throw JangSpecError.unsupportedVersion(
                field: "index", value: Int(version), supported: 1
            )
        }

        let nEntries = Int(nRaw)
        let expectedSize = JangSpecFormat.indexHeaderSize + nEntries * JangSpecFormat.indexEntrySize
        guard data.count >= expectedSize else {
            throw JangSpecError.truncated(url, expected: expectedSize, actual: data.count)
        }

        var parsed: [ExpertIndexEntry] = []
        parsed.reserveCapacity(nEntries)

        data.withUnsafeBytes { raw in
            var cursor = JangSpecFormat.indexHeaderSize
            for _ in 0..<nEntries {
                let layer = raw.load(fromByteOffset: cursor + 0, as: UInt32.self)
                let expert = raw.load(fromByteOffset: cursor + 4, as: UInt32.self)
                let fileID = raw.load(fromByteOffset: cursor + 8, as: UInt16.self)
                // 2 bytes pad at offset cursor + 10
                let offset = raw.load(fromByteOffset: cursor + 12, as: UInt64.self)
                let nbytes = raw.load(fromByteOffset: cursor + 20, as: UInt64.self)
                parsed.append(
                    ExpertIndexEntry(
                        layerIdx: Int(layer),
                        expertID: Int(expert),
                        fileID: Int(fileID),
                        offset: Int(offset),
                        nbytes: Int(nbytes)
                    )
                )
                cursor += JangSpecFormat.indexEntrySize
            }
        }

        self.nLayers = Int(nLRaw)
        self.nExpertsPerLayer = Int(nERaw)
        self.entries = parsed

        var map: [Int: ExpertIndexEntry] = [:]
        map.reserveCapacity(parsed.count)
        for e in parsed {
            map[Self.key(layer: e.layerIdx, expert: e.expertID)] = e
        }
        self.byKey = map
    }

    /// Constant-time lookup by (layer, expert).
    public func entry(layer: Int, expert: Int) throws -> ExpertIndexEntry {
        guard let e = byKey[Self.key(layer: layer, expert: expert)] else {
            throw JangSpecError.missingEntry(layer: layer, expert: expert)
        }
        return e
    }

    @inline(__always)
    private static func key(layer: Int, expert: Int) -> Int {
        // Safe up to ~2^31 layers and ~2^32 experts, which will never happen.
        return (layer << 32) | expert
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter ExpertIndexTests 2>&1 | tail -15
```
Expected: 3 tests passing.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/ExpertIndex.swift jang-runtime/Tests/JANGCoreTests/ExpertIndexTests.swift
git commit -m "jang-core: ExpertIndex parser with synthetic round-trip tests"
```

---

## Task 4: ExpertBlob parser (ExpertBlob.swift)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/ExpertBlob.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/ExpertBlobTests.swift`

- [ ] **Step 1: Write the failing test**

Write `jang-runtime/Tests/JANGCoreTests/ExpertBlobTests.swift`:
```swift
import XCTest
@testable import JANGCore

final class ExpertBlobTests: XCTestCase {
    /// Build a minimal synthetic blob matching the format Python writes.
    /// Uses 9 tensor entries (gate/up/down × qweight/scales/biases) with
    /// tiny payloads so we can hand-verify offsets.
    private func makeSyntheticBlob(bits: UInt8 = 4) -> Data {
        // One uint32 per qweight (4 bytes), one f16 per scale (2 bytes),
        // one f16 per bias (2 bytes), times 3 kinds = 24 bytes payload.
        let payload: [UInt8] = Array(repeating: 0xAB, count: 24)
        // But we want distinct bytes per tensor to detect offset bugs.
        var distinct = [UInt8]()
        for i in 0..<24 { distinct.append(UInt8(i) | 0x80) }

        let headerArea = JangSpecFormat.blobHeaderSize + 9 * JangSpecFormat.tensorHeaderSize
        let payloadOffset = headerArea
        let payloadBytes = distinct.count

        var data = Data()

        // BlobHeader: <I magic, H version, H n_tensors, I layer_idx, I expert_id, Q payload_offset, Q payload_bytes>
        var magic = JangSpecFormat.blobMagic
        var version: UInt16 = 1
        var nTensors: UInt16 = 9
        var layerIdx: UInt32 = 7
        var expertID: UInt32 = 3
        var payOff: UInt64 = UInt64(payloadOffset)
        var payBytes: UInt64 = UInt64(payloadBytes)
        withUnsafeBytes(of: &magic) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &version) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &nTensors) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &layerIdx) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &expertID) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &payOff) { data.append(contentsOf: $0) }
        withUnsafeBytes(of: &payBytes) { data.append(contentsOf: $0) }

        // 9 tensor headers: <B kind, B bits, H _pad, I dtype, I d0, I d1, I d2, Q offset, Q nbytes>
        // Serialization order (matches Python _KINDS × _DTYPES):
        //   (gate, qweight), (gate, scales), (gate, biases),
        //   (up,   qweight), (up,   scales), (up,   biases),
        //   (down, qweight), (down, scales), (down, biases)
        let kindsDtypes: [(UInt8, UInt32, [UInt32])] = [
            (JangSpecFormat.TensorKind.gate.rawValue, JangSpecFormat.TensorDType.qweight.rawValue, [1, 1, 0]),
            (JangSpecFormat.TensorKind.gate.rawValue, JangSpecFormat.TensorDType.scales.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.gate.rawValue, JangSpecFormat.TensorDType.biases.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.up.rawValue,   JangSpecFormat.TensorDType.qweight.rawValue, [1, 1, 0]),
            (JangSpecFormat.TensorKind.up.rawValue,   JangSpecFormat.TensorDType.scales.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.up.rawValue,   JangSpecFormat.TensorDType.biases.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.down.rawValue, JangSpecFormat.TensorDType.qweight.rawValue, [1, 1, 0]),
            (JangSpecFormat.TensorKind.down.rawValue, JangSpecFormat.TensorDType.scales.rawValue,  [1, 1, 0]),
            (JangSpecFormat.TensorKind.down.rawValue, JangSpecFormat.TensorDType.biases.rawValue,  [1, 1, 0]),
        ]
        var runningOffset: UInt64 = 0
        let sliceSizes: [Int] = [
            4, 2, 2,  // gate: qweight=4, scales=2, biases=2
            4, 2, 2,  // up
            4, 2, 2,  // down
        ]
        for (i, triple) in kindsDtypes.enumerated() {
            var kind = triple.0
            var bitsLocal = bits
            var pad: UInt16 = 0
            var dtype = triple.1
            var d0 = triple.2[0]
            var d1 = triple.2[1]
            var d2 = triple.2[2]
            var off = runningOffset
            var nb = UInt64(sliceSizes[i])
            withUnsafeBytes(of: &kind) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &bitsLocal) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &pad) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &dtype) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &d0) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &d1) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &d2) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &off) { data.append(contentsOf: $0) }
            withUnsafeBytes(of: &nb) { data.append(contentsOf: $0) }
            runningOffset += UInt64(sliceSizes[i])
        }

        // Payload
        data.append(contentsOf: distinct)

        // Pad to alignment
        let padded = JangSpecFormat.alignUp(data.count)
        data.append(contentsOf: [UInt8](repeating: 0, count: padded - data.count))

        XCTAssertEqual(data.count % JangSpecFormat.blobAlignment, 0)
        return data
    }

    func testParseSyntheticBlob() throws {
        let bytes = makeSyntheticBlob()
        let blob = try ExpertBlob(rawBytes: bytes)

        XCTAssertEqual(blob.layerIdx, 7)
        XCTAssertEqual(blob.expertID, 3)
        XCTAssertEqual(blob.bits, 4)
        XCTAssertEqual(blob.tensors.count, 9)

        // The gate/qweight should be bytes 0..3 of the payload region.
        let gateQ = blob.tensor(kind: .gate, dtype: .qweight)
        XCTAssertNotNil(gateQ)
        XCTAssertEqual(gateQ!.count, 4)
        XCTAssertEqual(gateQ![0], UInt8(0) | 0x80)

        // The down/biases is the last 2 bytes of the payload.
        let downB = blob.tensor(kind: .down, dtype: .biases)
        XCTAssertNotNil(downB)
        XCTAssertEqual(downB!.count, 2)
        XCTAssertEqual(downB![0], UInt8(22) | 0x80)
        XCTAssertEqual(downB![1], UInt8(23) | 0x80)
    }

    func testBadMagicThrows() throws {
        var bytes = makeSyntheticBlob()
        bytes.replaceSubrange(0..<4, with: Data([0, 0, 0, 0]))
        XCTAssertThrowsError(try ExpertBlob(rawBytes: bytes)) { err in
            guard case JangSpecError.badMagic = err else {
                XCTFail("expected badMagic, got \(err)")
                return
            }
        }
    }
}
```

- [ ] **Step 2: Run the failing test**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter ExpertBlobTests 2>&1 | tail -10
```
Expected: compile error — `ExpertBlob` not found.

- [ ] **Step 3: Implement ExpertBlob.swift**

Write `jang-runtime/Sources/JANGCore/ExpertBlob.swift`:
```swift
import Foundation

/// One tensor slice inside an expert blob.
///
/// The `slice` is a view on the blob's backing bytes. Consumers must not
/// retain it beyond the blob's lifetime unless they make their own copy.
public struct ExpertBlobTensor: Sendable {
    public let kind: JangSpecFormat.TensorKind
    public let dtype: JangSpecFormat.TensorDType
    public let bits: Int
    public let dims: [Int]       // zero-padded dims stripped
    public let slice: Data       // the raw bytes for this tensor
}

/// A parsed expert blob — header plus 9 tensor slices.
///
/// `ExpertBlob(rawBytes:)` validates the magic and header and decodes the
/// per-tensor offsets. The tensor byte ranges are held as `Data` slices
/// that reference the caller-owned backing buffer. For mmap'd reads this
/// is zero-copy; for in-memory test data the slices share storage with
/// the originating `Data`.
public struct ExpertBlob: Sendable {
    public let layerIdx: Int
    public let expertID: Int
    public let bits: Int
    public let tensors: [ExpertBlobTensor]

    public init(rawBytes data: Data) throws {
        guard data.count >= JangSpecFormat.blobHeaderSize else {
            throw JangSpecError.invalidBlob("blob too short for header (\(data.count) bytes)")
        }

        let (magic, version, nTensors, layer, expert, payloadOffset, payloadBytes):
            (UInt32, UInt16, UInt16, UInt32, UInt32, UInt64, UInt64) =
                data.withUnsafeBytes { raw in
                    let m = raw.load(fromByteOffset: 0, as: UInt32.self)
                    let v = raw.load(fromByteOffset: 4, as: UInt16.self)
                    let n = raw.load(fromByteOffset: 6, as: UInt16.self)
                    let l = raw.load(fromByteOffset: 8, as: UInt32.self)
                    let e = raw.load(fromByteOffset: 12, as: UInt32.self)
                    let po = raw.load(fromByteOffset: 16, as: UInt64.self)
                    let pb = raw.load(fromByteOffset: 24, as: UInt64.self)
                    return (m, v, n, l, e, po, pb)
                }

        guard magic == JangSpecFormat.blobMagic else {
            throw JangSpecError.badMagic(
                expected: JangSpecFormat.blobMagic,
                actual: magic,
                at: URL(fileURLWithPath: "(blob)")
            )
        }
        guard version == 1 else {
            throw JangSpecError.unsupportedVersion(
                field: "blob", value: Int(version), supported: 1
            )
        }
        guard nTensors == 9 else {
            throw JangSpecError.invalidBlob("expected 9 tensor entries, got \(nTensors)")
        }

        let tensorCount = Int(nTensors)
        let headerArea = JangSpecFormat.blobHeaderSize + tensorCount * JangSpecFormat.tensorHeaderSize
        let payOff = Int(payloadOffset)
        let payBytes = Int(payloadBytes)

        guard payOff == headerArea else {
            throw JangSpecError.invalidBlob(
                "payload_offset \(payOff) does not match header area \(headerArea)"
            )
        }
        guard data.count >= payOff + payBytes else {
            throw JangSpecError.invalidBlob(
                "blob shorter than declared payload: \(data.count) < \(payOff + payBytes)"
            )
        }

        var bitsSeen: Int? = nil
        var parsed: [ExpertBlobTensor] = []
        parsed.reserveCapacity(tensorCount)

        for i in 0..<tensorCount {
            let cursor = JangSpecFormat.blobHeaderSize + i * JangSpecFormat.tensorHeaderSize
            let (kindRaw, bitsVal, dtypeRaw, d0, d1, d2, offRaw, nbRaw):
                (UInt8, UInt8, UInt32, UInt32, UInt32, UInt32, UInt64, UInt64) =
                    data.withUnsafeBytes { raw in
                        let k = raw.load(fromByteOffset: cursor + 0, as: UInt8.self)
                        let b = raw.load(fromByteOffset: cursor + 1, as: UInt8.self)
                        // 2 bytes pad at cursor + 2
                        let dt = raw.load(fromByteOffset: cursor + 4, as: UInt32.self)
                        let x = raw.load(fromByteOffset: cursor + 8, as: UInt32.self)
                        let y = raw.load(fromByteOffset: cursor + 12, as: UInt32.self)
                        let z = raw.load(fromByteOffset: cursor + 16, as: UInt32.self)
                        let o = raw.load(fromByteOffset: cursor + 20, as: UInt64.self)
                        let n = raw.load(fromByteOffset: cursor + 28, as: UInt64.self)
                        return (k, b, dt, x, y, z, o, n)
                    }

            guard let kind = JangSpecFormat.TensorKind(rawValue: kindRaw) else {
                throw JangSpecError.invalidBlob("unknown tensor kind \(kindRaw)")
            }
            guard let dtype = JangSpecFormat.TensorDType(rawValue: dtypeRaw) else {
                throw JangSpecError.invalidBlob("unknown tensor dtype \(dtypeRaw)")
            }

            let bi = Int(bitsVal)
            if let prev = bitsSeen {
                if prev != bi {
                    throw JangSpecError.invalidBlob("mixed bits in one blob: \(prev) vs \(bi)")
                }
            } else {
                bitsSeen = bi
            }

            let rawDims = [Int(d0), Int(d1), Int(d2)]
            let dims = rawDims.filter { $0 != 0 }

            let start = payOff + Int(offRaw)
            let end = start + Int(nbRaw)
            guard end <= data.count else {
                throw JangSpecError.invalidBlob(
                    "tensor slice out of range: \(start)..<\(end) (blob size \(data.count))"
                )
            }
            let slice = data.subdata(in: start..<end)

            parsed.append(
                ExpertBlobTensor(
                    kind: kind,
                    dtype: dtype,
                    bits: bi,
                    dims: dims,
                    slice: slice
                )
            )
        }

        self.layerIdx = Int(layer)
        self.expertID = Int(expert)
        self.bits = bitsSeen ?? 0
        self.tensors = parsed
    }

    /// Convenience: find the first tensor matching a given kind and dtype.
    public func tensor(kind: JangSpecFormat.TensorKind, dtype: JangSpecFormat.TensorDType) -> Data? {
        for t in tensors where t.kind == kind && t.dtype == dtype {
            return t.slice
        }
        return nil
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter ExpertBlobTests 2>&1 | tail -15
```
Expected: 2 tests passing.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/ExpertBlob.swift jang-runtime/Tests/JANGCoreTests/ExpertBlobTests.swift
git commit -m "jang-core: ExpertBlob parser with 9-tensor synthetic round-trip"
```

---

## Task 5: Bundle manifest (JangSpecManifest.swift)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/JangSpecManifest.swift`

- [ ] **Step 1: Implement JangSpecManifest.swift**

Write `jang-runtime/Sources/JANGCore/JangSpecManifest.swift`:
```swift
import Foundation

/// Mirror of Python `jang_tools.jangspec.manifest.Manifest`.
///
/// Decoded from `jangspec.json` via `JSONDecoder`. Field names must match
/// Python's `dataclasses.asdict` output exactly.
public struct JangSpecManifest: Codable, Sendable, Equatable {
    public var bundleVersion: Int
    public var sourceJang: String
    public var sourceJangDir: String
    public var targetArch: String
    public var nLayers: Int
    public var nExpertsPerLayer: Int
    public var targetTopK: Int
    public var tokenizerHash: String
    public var hotCoreTensors: [String]
    public var expertTensorNames: [String]
    public var nExpertsTotal: Int
    public var hotCoreBytes: Int
    public var expertBytes: Int
    public var hasDraft: Bool
    public var hasRouterPrior: Bool
    public var draftJang: String
    public var toolVersion: String
    public var schema: String

    enum CodingKeys: String, CodingKey {
        case bundleVersion = "bundle_version"
        case sourceJang = "source_jang"
        case sourceJangDir = "source_jang_dir"
        case targetArch = "target_arch"
        case nLayers = "n_layers"
        case nExpertsPerLayer = "n_experts_per_layer"
        case targetTopK = "target_top_k"
        case tokenizerHash = "tokenizer_hash"
        case hotCoreTensors = "hot_core_tensors"
        case expertTensorNames = "expert_tensor_names"
        case nExpertsTotal = "n_experts_total"
        case hotCoreBytes = "hot_core_bytes"
        case expertBytes = "expert_bytes"
        case hasDraft = "has_draft"
        case hasRouterPrior = "has_router_prior"
        case draftJang = "draft_jang"
        case toolVersion = "tool_version"
        case schema
    }

    public static func load(from url: URL) throws -> JangSpecManifest {
        let data = try Data(contentsOf: url)
        let decoder = JSONDecoder()
        let m = try decoder.decode(JangSpecManifest.self, from: data)
        guard m.bundleVersion == JangSpecFormat.bundleVersion else {
            throw JangSpecError.unsupportedVersion(
                field: "bundle",
                value: m.bundleVersion,
                supported: JangSpecFormat.bundleVersion
            )
        }
        return m
    }
}
```

- [ ] **Step 2: Build to confirm it compiles**

```bash
cd /Users/eric/jang/jang-runtime && swift build 2>&1 | tail -10
```
Expected: build success. No tests for this file yet — it's covered by the integration test in Task 7.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/JangSpecManifest.swift
git commit -m "jang-core: JangSpecManifest Codable struct"
```

---

## Task 6: Expert store (ExpertStore.swift)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/ExpertStore.swift`

- [ ] **Step 1: Implement ExpertStore.swift**

Write `jang-runtime/Sources/JANGCore/ExpertStore.swift`:
```swift
import Foundation

/// Lazy mmap over a bundle's `experts-NNNNN.bin` shard files.
///
/// Plan 2 uses `Data(contentsOf:options:.mappedIfSafe)` for zero-copy
/// access on Apple Silicon. Plan 3 will add a parallel `MTLIOCommandQueue`
/// path for direct-to-GPU reads; the `load` signature here is the contract
/// that streaming path will satisfy.
public final class ExpertStore: @unchecked Sendable {
    public let bundleURL: URL
    public let index: ExpertIndex

    private var mapped: [Int: Data] = [:]
    private let lock = NSLock()

    public init(bundleURL: URL, index: ExpertIndex) {
        self.bundleURL = bundleURL
        self.index = index
    }

    /// Load one expert by (layer, expert) — returns a parsed `ExpertBlob`.
    /// Throws if the shard file is missing or the blob fails validation.
    public func load(layer: Int, expert: Int) throws -> ExpertBlob {
        let entry = try index.entry(layer: layer, expert: expert)
        let mm = try mapFile(id: entry.fileID)
        guard entry.offset + entry.nbytes <= mm.count else {
            throw JangSpecError.truncated(
                bundleURL.appendingPathComponent(JangSpecFormat.expertFilename(idx: entry.fileID)),
                expected: entry.offset + entry.nbytes,
                actual: mm.count
            )
        }
        let slice = mm.subdata(in: entry.offset..<(entry.offset + entry.nbytes))
        return try ExpertBlob(rawBytes: slice)
    }

    private func mapFile(id: Int) throws -> Data {
        lock.lock()
        defer { lock.unlock() }
        if let hit = mapped[id] {
            return hit
        }
        let url = bundleURL.appendingPathComponent(JangSpecFormat.expertFilename(idx: id))
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw JangSpecError.fileMissing(url)
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        mapped[id] = data
        return data
    }
}
```

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build 2>&1 | tail -10
```
Expected: build success.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/ExpertStore.swift
git commit -m "jang-core: ExpertStore — mmap experts-*.bin, load-by-key"
```

---

## Task 7: Bundle façade + integration test (JangSpecBundle.swift)

**Files:**
- Create: `jang-runtime/Sources/JANGCore/JangSpecBundle.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/Fixtures.swift`
- Create: `jang-runtime/Tests/JANGCoreTests/JangSpecBundleTests.swift`

- [ ] **Step 1: Write the bundle façade**

Write `jang-runtime/Sources/JANGCore/JangSpecBundle.swift`:
```swift
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
```

- [ ] **Step 2: Write the test fixture helper**

Write `jang-runtime/Tests/JANGCoreTests/Fixtures.swift`:
```swift
import Foundation

/// Builds a Gemma-4-26B `.jangspec` fixture bundle via the Python
/// `jang spec build` CLI. The bundle is cached under
/// `/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec` and reused
/// across test runs as long as the manifest is present.
enum Fixtures {
    static let sourceModelPath = "/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M"
    static let cacheDir = URL(fileURLWithPath: "/tmp/jangcore-fixtures")
    static let bundleURL = cacheDir.appendingPathComponent("Gemma-4-26B-A4B-it-JANG_4M.jangspec")

    static func gemmaBundle() throws -> URL {
        let manifest = bundleURL.appendingPathComponent("jangspec.json")
        if FileManager.default.fileExists(atPath: manifest.path) {
            return bundleURL
        }
        guard FileManager.default.fileExists(atPath: sourceModelPath) else {
            throw NSError(
                domain: "Fixtures",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "Gemma-4-26B-A4B-it-JANG_4M fixture model not found at \(sourceModelPath). Set JANGSPEC_TEST_MODEL env var if using a different path (fixture helper does not read it yet)."]
            )
        }
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)

        let proc = Process()
        proc.launchPath = "/bin/bash"
        proc.arguments = [
            "-c",
            "jang spec build '\(sourceModelPath)' --out '\(bundleURL.path)' --force"
        ]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        guard proc.terminationStatus == 0 else {
            throw NSError(
                domain: "Fixtures",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey:
                    "jang spec build failed with exit code \(proc.terminationStatus)"]
            )
        }
        return bundleURL
    }
}
```

- [ ] **Step 3: Write the integration test**

Write `jang-runtime/Tests/JANGCoreTests/JangSpecBundleTests.swift`:
```swift
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
```

- [ ] **Step 4: Build and test**

```bash
cd /Users/eric/jang/jang-runtime && swift test --filter JangSpecBundleTests 2>&1 | tail -25
```
Expected: 2 tests passing (or `SKIPPED` with a clear reason if the Gemma fixture is unavailable). If skipped on this machine, verify the source model exists at `/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M`.

- [ ] **Step 5: Full test sweep**

```bash
cd /Users/eric/jang/jang-runtime && swift test 2>&1 | tail -30
```
Expected: `JangSpecFormatTests`, `ExpertIndexTests`, `ExpertBlobTests`, `JangSpecBundleTests` all pass. Existing `JANGTests` should be unaffected.

- [ ] **Step 6: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/JANGCore/JangSpecBundle.swift jang-runtime/Tests/JANGCoreTests/Fixtures.swift jang-runtime/Tests/JANGCoreTests/JangSpecBundleTests.swift
git commit -m "jang-core: JangSpecBundle façade + Gemma fixture integration tests"
```

---

## Task 8: jang-core CLI — inspect

**Files:**
- Create: `jang-runtime/Sources/jang-core/main.swift`

- [ ] **Step 1: Implement main.swift**

Write `jang-runtime/Sources/jang-core/main.swift`:
```swift
import ArgumentParser
import Foundation
import JANGCore

@main
struct JangCore: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "jang-core",
        abstract: "Load and inspect .jangspec bundles in pure Swift.",
        version: JANGCore.version,
        subcommands: [Inspect.self]
    )
}

struct Inspect: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        abstract: "Print a bundle's manifest summary."
    )

    @Argument(help: "Path to a .jangspec directory.")
    var bundle: String

    func run() async throws {
        let url = URL(fileURLWithPath: (bundle as NSString).expandingTildeInPath).resolvingSymlinksInPath()
        let spec = try JangSpecBundle.open(at: url)
        let m = spec.manifest

        let hotGB = Double(m.hotCoreBytes) / 1e9
        let expGB = Double(m.expertBytes) / 1e9

        print("  bundle:        \(url.path)")
        print("  source jang:   \(m.sourceJang)")
        print("  arch:          \(m.targetArch)")
        print("  n_layers:      \(m.nLayers)")
        print("  experts/layer: \(m.nExpertsPerLayer)")
        print("  top_k:         \(m.targetTopK)")
        print(String(format: "  hot_core:      %.2f GB", hotGB))
        print(String(format: "  expert_bytes:  %.2f GB", expGB))
        print("  draft:         \(m.hasDraft)")
        print("  router_prior:  \(m.hasRouterPrior)")
        print("  bundle_version: \(m.bundleVersion)")
        print("  tool_version:  \(m.toolVersion)")
    }
}
```

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/jang-runtime && swift build -c release --product jang-core 2>&1 | tail -10
```
Expected: build success.

- [ ] **Step 3: Smoke-test against the fixture bundle**

```bash
cd /Users/eric/jang/jang-runtime && ./.build/release/jang-core inspect /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec 2>&1
```
Expected: the same fields Python's `jang spec inspect` prints, in Swift.
If the fixture bundle doesn't exist yet at that path (because the integration test didn't run), build it first:
```bash
jang spec build /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M --out /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec --force
```

- [ ] **Step 4: Parity check against Python**

```bash
jang spec inspect /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec
./.build/release/jang-core inspect /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec
```
Compare the two outputs visually. All fields shared between the two implementations (source jang, arch, n_layers, experts/layer, top_k, hot_core GB, expert_bytes GB, draft, router_prior) must match exactly.

- [ ] **Step 5: Commit**

```bash
cd /Users/eric/jang && git add jang-runtime/Sources/jang-core/main.swift
git commit -m "jang-core: jang-core inspect subcommand with Python parity"
```

---

## Task 9: Final sweep

**Files:** none

- [ ] **Step 1: Full Swift test suite**

```bash
cd /Users/eric/jang/jang-runtime && swift test 2>&1 | tail -20
```
Expected: all `JANGCoreTests` pass, existing `JANGTests` unchanged.

- [ ] **Step 2: Rebuild all products**

```bash
cd /Users/eric/jang/jang-runtime && swift build -c release 2>&1 | tail -15
```
Expected: `jang`, `jang-spec-iobench`, `jang-core` all build.

- [ ] **Step 3: Summarize commits**

```bash
cd /Users/eric/jang && git log --oneline main..HEAD
```
Expected: the tasks from this plan, each as one commit.

- [ ] **Step 4: Write a one-paragraph plan outcome summary**

- JANGCore Swift library ships, reads `.jangspec` bundles produced by Plan 1.
- `jang-core inspect` parity-checked against Python's `jang spec inspect`.
- Bundle format round-trip now has two independent implementations (Python + Swift).
- Next: Plan 3 — hot-core safetensors loader + Metal dequant kernels + MoE forward pass (in RAM).
