# Plan 6 — Native `.jangspec` bundle loading in vmlx-swift-lm

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land `.jangspec` bundle support inside vmlx-swift-lm — the production fast Swift runtime that already does ~101 tok/s on Gemma-4-26B-A4B — so jang-spec models run in the existing engine instead of waiting for a from-scratch JANGCore engine. Pure Swift, no Python wrappers, no cross-package dependencies.

**Architecture:** A new `JangSpecBundleLoader.swift` in `Libraries/MLXLMCommon/` self-contained: parses the `.jangspec` manifest + flat expert index + per-expert blobs, mmaps the hot-core safetensors via the existing reader, restacks experts into the 3D `[E, ...]` tensors mlx-swift's models expect under their original `switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` keys, and returns a `[String: MLXArray]` dict shaped identically to what the existing `loadWeights(modelDirectory:)` enumeration produces from a regular JANG directory. `Load.swift loadWeights()` gains a one-branch fast path: if the model directory is a `.jangspec` bundle, the new loader replaces the `.safetensors` enumeration step and the rest of the pipeline (sanitize, MoE gate dequant, per-layer quant inference, model.update) runs unchanged.

**Tech Stack:** Swift 6, mlx-swift 0.31.3 fork, Foundation, MLX. Zero new SwiftPM dependencies. Format constants are duplicated from `jang_tools.jangspec.format` and `jang-runtime/Sources/JANGCore/JangSpecFormat.swift` (they're 32 bytes of magic numbers and struct sizes — duplication is cheaper than adding a cross-package SwiftPM dep at this stage; we can deduplicate later if needed).

**Spec:** `docs/superpowers/specs/2026-04-13-jang-spec-design.md` §5 (bundle format), §8.2 (vmlx integration vehicle).

**Depends on:**
- Plan 1 (Python builder) — needs a small additive update to copy configs to the bundle root.
- Plans 2–4 (JANGCore Swift loader) — provides the proven format parser shape we mirror in this plan, but is NOT a dependency.
- Plan 5 (Python validation) — confirms the bundle is byte-correct, so when the Swift port disagrees with Python the bug is in Swift.

**Out of scope (deferred):**
- Running the existing `RunBench` end-to-end on the bundle. Eric runs that manually when RAM is free.
- Streaming experts from SSD via `MTLIOCommandQueue`. This plan is "load whole bundle into RAM via mmap and restack." Streaming is Plan 7.
- Speculative decoding draft model loading. Plan 8+.
- TQ-compressed bundle reads. The current builder writes plain-quantized blobs; TQ-compressed bundle support is a sibling task to L2 cache TQ-compressed restore (already shipped in vmlx-swift-lm cache work, but not wired through the bundle loader).
- `JANGCoreMetal` Plan 4 kernel. mlx-swift's existing quantized matmul kernels are the fast path for vmlx-swift-lm; we do not need a parallel kernel.

**Test fixtures:**
- Source: `/Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M/` (16 GB)
- Bundle: `/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec/` (existing, but Task 1 rebuilds it after the builder update so configs land at the bundle root)

---

## Why this approach

We have two parallel Swift codepaths that do overlapping work:

- **`jang-runtime/JANGCore`** (Plans 2–4 of this session): pure-Foundation bundle parser + one Metal 4-bit GEMV kernel. Has no forward pass. Building a from-scratch model architecture for Gemma-4-26B (dual-head attention, GQA, sliding window, K=V tying, 128-expert top-8 MoE, logit softcap, GELU) is realistically 4–6 weeks of work for one model and gives zero speed gain over what already exists.
- **`vmlx-swift-lm`** (the canonical fast runtime): already runs Gemma-4-26B-A4B at 101.5 tok/s avg / 107.1 peak via mlx-swift. Already has every model architecture, KV cache (with the L2 disk fixes from this session), sampling, tokenizer, MoE dispatch, sliding window, the lot. The remaining ~8 tok/s gap to 110 is mlx-swift compile fusion micro-optimization, not architecture work.

Building a second Swift engine to do the same job doesn't accelerate anything. **Teaching vmlx-swift-lm to read `.jangspec` bundles** gets jang-spec into a real production runtime in one session of work, ~500 lines of new Swift, no architecture porting.

The JANGCore work from Plans 2–4 is **not deleted** — it stays as a standalone debug/inspection toolkit (`jang-core inspect`, `jang-core hot-core`, `jang-spec-iobench`). It's just no longer the path to running models fast. The Metal kernel work in Plan 4 is shelved (research detour); mlx-swift's existing quantized matmul kernels are the fast path for vmlx-swift-lm.

---

## File structure

New files:

```
vmlx-swift-lm/Libraries/MLXLMCommon/
  JangSpecBundleLoader.swift     bundle loader: format, blob, index, manifest, restack
```

Modified files:

```
jang-tools/jang_tools/jangspec/builder.py
  Copy `config.json` and `jang_config.json` into the bundle ROOT (in addition to
  the existing target/ copies) so vmlx-swift-lm's existing JangLoader.findConfigPath
  picks them up at the canonical location without any factory changes.

vmlx-swift-lm/Libraries/MLXLMCommon/Load.swift
  Detect `jangspec.json` in the model directory and route weight loading through
  JangSpecBundleLoader instead of the standard safetensors enumeration. ~10 line
  branch at the start of loadWeights().

vmlx-swift-lm/Libraries/MLXLMCommon/JangLoader.swift
  Defensive: extend findConfigPath() to fall back to <root>/target/<name> if the
  config isn't at the root. Keeps old bundles built before this plan loadable.
```

No dependency changes in any Package.swift. No new SwiftPM products.

---

## Bundle vs RAM-load trade-off (worth knowing)

The `.jangspec` bundle splits experts into one blob per `(layer, expert_id)` (15 872 blobs for MiniMax-M2.7-JANG_2L; 3 840 for Gemma-4-26B-A4B-it-JANG_4M). This layout is **optimal for SSD streaming** and forced the design of Plans 1–3.

For "load entire bundle into RAM via mmap" (the Plan 6 path), the per-blob layout adds a small cost on top of vanilla safetensors loading:

- **Vanilla safetensors:** 1 mmap per shard file (~4 shards for Gemma-4-26B) → models read tensors directly from mmap views, zero stack ops
- **`.jangspec` bundle:** 1 mmap per `experts-NNNNN.bin` file (~3 files for Gemma-4) + 1 mmap for `hot_core.safetensors`. Each MoE base name then needs **256 expert blob slices stacked into a 3D tensor** via `MLX.stacked(_:axis: 0)`. 30 layers × 3 base names = 90 stack operations of 256 arrays each.

Empirically (Apple Silicon unified memory, M4 Max), one such stack op of 256 small arrays runs in ~10–30 ms. So total bundle load adds **~1–3 seconds** on top of vanilla safetensors load on Gemma-4-26B. Negligible compared to inference cost.

This is acceptable for v1. If it ever becomes the bottleneck a future plan can add a **"fat bundle"** mode that keeps experts in their original 3D stacked layout (one safetensors file per layer, similar to source) — read at vanilla safetensors speed, no stacking. That mode loses the per-expert SSD streaming property, so it would be a build-time choice (`jang spec build --layout fat`).

---

## Task 0: Branches and confirm starting state

**Files:** none

- [ ] **Step 1: Confirm jang repo state**

```bash
cd /Users/eric/jang && git status && git log -1 --oneline && git branch --show-current
```

Expected: clean tree, current branch `jang-spec-plan5-bundle-python-validation` (or whichever branch holds the latest committed work), latest commit is the Plan 5 STATUS update.

- [ ] **Step 2: Create branches in both repos**

The jang-tools changes (builder update) and vmlx-swift-lm changes are in different repos. We branch each:

```bash
cd /Users/eric/jang && git checkout -b jang-spec-plan6-vmlx-integration
cd /Users/eric/jang/vmlx-swift-lm && git checkout -b jang-spec-bundle-loader
```

The vmlx-swift-lm copy at `/Users/eric/jang/vmlx-swift-lm/` is a separate git repo from the parent jang repo.

- [ ] **Step 3: Verify vmlx-swift-lm builds clean before any edits**

```bash
cd /Users/eric/jang/vmlx-swift-lm && swift build -c release 2>&1 | tail -3
```

Expected: `Build complete!` Anything else, stop and report.

---

## Task 1: Builder update — copy configs to bundle root

**Files:**
- Modify: `jang-tools/jang_tools/jangspec/builder.py` (add 5 lines to `_copy_tokenizer`)

**Background.** Currently the builder copies `config.json` and `jang_config.json` only to `<bundle>/target/`. vmlx-swift-lm's `JangLoader.findConfigPath` looks at the model directory root. To avoid touching the factories, the builder also copies them to the bundle root. Older bundles built before this plan will still load via Task 5's defensive fallback.

- [ ] **Step 1: Edit `_copy_tokenizer`**

Open `jang-tools/jang_tools/jangspec/builder.py`. Find the existing `_copy_tokenizer` method and update it so configs are copied to BOTH the bundle root AND `target/`:

```python
    def _copy_tokenizer(self) -> None:
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            src = self.source_dir / name
            if src.exists():
                shutil.copy2(src, self.out_dir / name)
        # Copy source jang_config.json and config.json into target/ for the
        # Swift streaming runtime, AND into the bundle root so vmlx-swift-lm's
        # existing JangLoader.findConfigPath picks them up at the canonical
        # location without any factory changes.
        (self.out_dir / "target").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "jang_config.json"):
            src = self.source_dir / name
            if src.exists():
                shutil.copy2(src, self.out_dir / "target" / name)
                shutil.copy2(src, self.out_dir / name)
```

(The change is the single added `shutil.copy2(src, self.out_dir / name)` line.)

- [ ] **Step 2: Re-run the builder unit tests to confirm nothing broke**

```bash
cd /Users/eric/jang/jang-tools && python3 -m pytest tests/jangspec/test_builder.py tests/jangspec/test_reader.py -v 2>&1 | tail -15
```

Expected: all builder + reader tests pass (or skip if fixture is unavailable; should not skip on this machine).

- [ ] **Step 3: Rebuild the Gemma-4-26B fixture bundle with the updated builder**

```bash
cd /Users/eric/jang && jang spec build /Users/eric/jang/models/Gemma-4-26B-A4B-it-JANG_4M --out /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec --force 2>&1 | tail -10
```

Expected: build completes in ~6–10s, prints layer/expert summary.

- [ ] **Step 4: Verify the configs landed at both locations**

```bash
ls -la /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec/{config.json,jang_config.json,target/config.json,target/jang_config.json}
```

Expected: all four files present.

- [ ] **Step 5: Commit the builder change**

```bash
cd /Users/eric/jang && git add jang-tools/jang_tools/jangspec/builder.py
git commit -m "jangspec(builder): copy config.json + jang_config.json to bundle root"
```

---

## Task 2: `JangSpecBundleLoader.swift` — format, parse, materialize

**Files:**
- Create: `vmlx-swift-lm/Libraries/MLXLMCommon/JangSpecBundleLoader.swift`

**Background.** Self-contained Swift implementation of the bundle reader, mirroring the proven Python and JANGCore parsers but with mlx-swift `MLXArray` outputs. No dependency on `jang-runtime/Sources/JANGCore/` — the format constants and struct layouts are 32 bytes of magic numbers; duplicating them is much cheaper than introducing a cross-package SwiftPM dep at this stage.

The flow inside `loadWeights(from:)`:

1. Open `<bundle>/jangspec.json` → decode `Manifest`
2. Verify `manifest.bundle_version == 1`
3. Open `<bundle>/target/experts.jsidx` → parse flat binary index → `(layer, expert) → (file_id, offset, nbytes)` lookup
4. mmap `<bundle>/target/hot_core.safetensors` via the existing safetensors loader (`loadArraysAndMetadata`) → copy every tensor key-value into the output dict
5. mmap each `<bundle>/target/experts-NNNNN.bin` shard
6. Group `manifest.expert_tensor_names` by layer index (regex `\.layers\.(\d+)\.`)
7. For each `(layer, base_name)` triple, iterate experts `0..<n_experts_per_layer`, parse the blob header at the indexed offset, slice out the `(qweight, scales, biases)` byte ranges, materialize each as an `MLXArray` of the right dtype/shape, then `MLX.stacked(_:axis: 0)` the per-expert arrays into a 3D `[E, ...]` tensor
8. Emit each stacked tensor under `{base_name}.{weight,scales,biases}` in the output dict

The output dict has the same keys, dtypes, and shapes that the existing `loadWeights` safetensors enumeration would produce on the source `JANG_4M/` directory. Downstream `model.sanitize`, `JangLoader.dequantizeMoEGates`, `JangLoader.inferPerLayerQuantization`, `model.update` all run unchanged.

- [ ] **Step 1: Write the file**

Write `vmlx-swift-lm/Libraries/MLXLMCommon/JangSpecBundleLoader.swift`:
```swift
// Copyright © 2025 JANG. All rights reserved.
//
// JangSpecBundleLoader — read a .jangspec bundle into the {key: MLXArray}
// dict that the existing loadWeights() pipeline expects.
//
// A .jangspec bundle is a self-contained directory holding:
//
//   <name>.jangspec/
//     jangspec.json                 manifest (bundle_version, tensor lists, sizes)
//     config.json                   model config (copied from source for factory detection)
//     jang_config.json              JANG quant metadata (copied from source)
//     tokenizer.json                tokenizer (copied from source)
//     tokenizer_config.json
//     target/
//       config.json                 same as root, kept for the streaming runtime
//       jang_config.json
//       hot_core.safetensors        attention/router/norm/embed/lm_head, mmap-ready
//       experts.jsidx               flat binary index, one entry per (layer, expert)
//       experts-00000.bin           per-expert blobs, 4 KB-aligned
//       experts-00001.bin
//       ...
//
// This loader is the inverse of jang_tools.jangspec.builder and produces the
// same {tensor_name: MLXArray} dict that mlx-swift's existing
// loadWeights(modelDirectory:) safetensors enumeration would produce on the
// source JANG model directory. Downstream sanitize/quant/update is unchanged.

import Foundation
import MLX

// MARK: - Format constants
//
// These mirror jang_tools/jangspec/format.py and
// jang-runtime/Sources/JANGCore/JangSpecFormat.swift exactly. If the on-disk
// format ever changes, update all three sites in lockstep.

public enum JangSpecBundleFormat {
    public static let bundleVersion: Int = 1
    public static let manifestFilename = "jangspec.json"
    public static let indexFilename = "target/experts.jsidx"
    public static let hotCoreFilename = "target/hot_core.safetensors"
    public static func expertFilename(idx: Int) -> String {
        return String(format: "target/experts-%05d.bin", idx)
    }

    public static let blobAlignment: Int = 4096
    public static let blobMagic: UInt32 = 0x4550_534A    // "JSPE"
    public static let indexMagic: UInt32 = 0x58_494A_53  // "SJIX"

    public static let blobHeaderSize: Int = 32
    public static let tensorHeaderSize: Int = 36
    public static let indexEntrySize: Int = 28
    public static let indexHeaderSize: Int = 24

    public enum TensorKind: UInt8 {
        case gate = 0
        case up = 1
        case down = 2
    }

    public enum TensorDType: UInt32 {
        case qweight = 0   // uint32 packed
        case scales = 1    // float16
        case biases = 2    // float16
    }
}

// MARK: - Errors

public enum JangSpecBundleError: Error, CustomStringConvertible {
    case fileMissing(URL)
    case unsupportedVersion(field: String, value: Int, supported: Int)
    case truncated(URL, expected: Int, actual: Int)
    case missingEntry(layer: Int, expert: Int)
    case invalidManifest(String)
    case invalidBlob(String)
    case invalidIndex(String)
    case missingBaseName(String)

    public var description: String {
        switch self {
        case .fileMissing(let url):
            return "jangspec: file missing: \(url.path)"
        case .unsupportedVersion(let field, let value, let supported):
            return "jangspec: unsupported \(field) version \(value), supported \(supported)"
        case .truncated(let url, let e, let a):
            return "jangspec: truncated \(url.lastPathComponent): expected \(e), got \(a)"
        case .missingEntry(let layer, let expert):
            return "jangspec: no entry for (layer=\(layer), expert=\(expert))"
        case .invalidManifest(let m):
            return "jangspec: invalid manifest: \(m)"
        case .invalidBlob(let m):
            return "jangspec: invalid blob: \(m)"
        case .invalidIndex(let m):
            return "jangspec: invalid index: \(m)"
        case .missingBaseName(let n):
            return "jangspec: cannot parse layer index from base name: \(n)"
        }
    }
}

// MARK: - Manifest

/// Mirror of jang_tools.jangspec.manifest.Manifest.
public struct JangSpecBundleManifest: Codable, Sendable {
    public var bundleVersion: Int
    public var sourceJang: String
    public var targetArch: String
    public var nLayers: Int
    public var nExpertsPerLayer: Int
    public var targetTopK: Int
    public var hotCoreTensors: [String]
    public var expertTensorNames: [String]
    public var nExpertsTotal: Int

    enum CodingKeys: String, CodingKey {
        case bundleVersion = "bundle_version"
        case sourceJang = "source_jang"
        case targetArch = "target_arch"
        case nLayers = "n_layers"
        case nExpertsPerLayer = "n_experts_per_layer"
        case targetTopK = "target_top_k"
        case hotCoreTensors = "hot_core_tensors"
        case expertTensorNames = "expert_tensor_names"
        case nExpertsTotal = "n_experts_total"
    }
}

// MARK: - Index entry

private struct ExpertIndexEntry {
    let layerIdx: Int
    let expertID: Int
    let fileID: Int
    let offset: Int
    let nbytes: Int
}

// MARK: - Bundle loader

public enum JangSpecBundleLoader {

    /// Returns true iff `directory` looks like a `.jangspec` bundle.
    public static func isBundle(at directory: URL) -> Bool {
        let manifest = directory.appendingPathComponent(JangSpecBundleFormat.manifestFilename)
        return FileManager.default.fileExists(atPath: manifest.path)
    }

    /// Load every tensor a model needs from a `.jangspec` bundle.
    ///
    /// The returned dict has the same keys and dtypes as a vanilla
    /// safetensors enumeration on the source JANG directory. Hot-core
    /// tensors are mmap'd via the existing `loadArraysAndMetadata` helper;
    /// expert tensors are restacked into 3D `[E, ...]` arrays.
    public static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        let manifestURL = directory.appendingPathComponent(
            JangSpecBundleFormat.manifestFilename)
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw JangSpecBundleError.fileMissing(manifestURL)
        }

        // 1. Manifest.
        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(
            JangSpecBundleManifest.self, from: manifestData)
        guard manifest.bundleVersion == JangSpecBundleFormat.bundleVersion else {
            throw JangSpecBundleError.unsupportedVersion(
                field: "bundle",
                value: manifest.bundleVersion,
                supported: JangSpecBundleFormat.bundleVersion
            )
        }

        var out: [String: MLXArray] = [:]

        // 2. Hot core — read every tensor via the existing safetensors loader.
        let hotCoreURL = directory.appendingPathComponent(
            JangSpecBundleFormat.hotCoreFilename)
        guard FileManager.default.fileExists(atPath: hotCoreURL.path) else {
            throw JangSpecBundleError.fileMissing(hotCoreURL)
        }
        let (hotArrays, _) = try loadArraysAndMetadata(url: hotCoreURL)
        for (key, value) in hotArrays {
            out[key] = value
        }

        // 3. Index.
        let indexURL = directory.appendingPathComponent(
            JangSpecBundleFormat.indexFilename)
        let entriesByKey = try parseIndex(at: indexURL)

        // 4. Expert shards — mmap once, reuse for every blob.
        var shardCache: [Int: Data] = [:]
        func shard(forID id: Int) throws -> Data {
            if let hit = shardCache[id] { return hit }
            let url = directory.appendingPathComponent(
                JangSpecBundleFormat.expertFilename(idx: id))
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw JangSpecBundleError.fileMissing(url)
            }
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            shardCache[id] = data
            return data
        }

        // 5. Group expert base names by layer.
        let layerGroups = try groupExpertBasesByLayer(manifest.expertTensorNames)

        // 6. For each (layer, base), walk experts in order and stack.
        for (layerIdx, baseNames) in layerGroups {
            // Each layer has gate_proj, up_proj, down_proj. Map kind → base.
            var kindToBase: [JangSpecBundleFormat.TensorKind: String] = [:]
            for base in baseNames {
                if base.hasSuffix(".switch_mlp.gate_proj") {
                    kindToBase[.gate] = base
                } else if base.hasSuffix(".switch_mlp.up_proj") {
                    kindToBase[.up] = base
                } else if base.hasSuffix(".switch_mlp.down_proj") {
                    kindToBase[.down] = base
                }
            }

            // Buffers for this layer's three bases. Each is a list of per-expert
            // arrays we'll stack at the end.
            struct PerExpertSlots {
                var qweight: [MLXArray] = []
                var scales: [MLXArray] = []
                var biases: [MLXArray] = []
            }
            var slots: [JangSpecBundleFormat.TensorKind: PerExpertSlots] = [
                .gate: PerExpertSlots(),
                .up: PerExpertSlots(),
                .down: PerExpertSlots(),
            ]

            for expertID in 0..<manifest.nExpertsPerLayer {
                let key = ExpertKey(layer: layerIdx, expert: expertID)
                guard let entry = entriesByKey[key] else {
                    throw JangSpecBundleError.missingEntry(
                        layer: layerIdx, expert: expertID)
                }
                let shardData = try shard(forID: entry.fileID)
                let blobBytes = shardData.subdata(
                    in: entry.offset..<(entry.offset + entry.nbytes))
                let parsed = try parseExpertBlob(blobBytes)

                for kind in [
                    JangSpecBundleFormat.TensorKind.gate,
                    .up,
                    .down,
                ] {
                    guard kindToBase[kind] != nil else { continue }
                    guard let triple = parsed.tensors[kind] else { continue }
                    slots[kind]!.qweight.append(triple.qweight)
                    slots[kind]!.scales.append(triple.scales)
                    slots[kind]!.biases.append(triple.biases)
                }
            }

            // Stack and emit.
            for kind in [
                JangSpecBundleFormat.TensorKind.gate,
                .up,
                .down,
            ] {
                guard let base = kindToBase[kind] else { continue }
                let s = slots[kind]!
                guard !s.qweight.isEmpty else { continue }
                out["\(base).weight"] = MLX.stacked(s.qweight, axis: 0)
                out["\(base).scales"] = MLX.stacked(s.scales, axis: 0)
                out["\(base).biases"] = MLX.stacked(s.biases, axis: 0)
            }
        }

        return out
    }

    // MARK: - Index parsing

    private struct ExpertKey: Hashable {
        let layer: Int
        let expert: Int
    }

    private static func parseIndex(at url: URL) throws -> [ExpertKey: ExpertIndexEntry] {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw JangSpecBundleError.fileMissing(url)
        }
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= JangSpecBundleFormat.indexHeaderSize else {
            throw JangSpecBundleError.truncated(
                url, expected: JangSpecBundleFormat.indexHeaderSize, actual: data.count)
        }

        let (magic, version, _, _, nEntries): (UInt32, UInt16, UInt32, UInt32, UInt64) =
            data.withUnsafeBytes { raw in
                let m = raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
                let v = raw.loadUnaligned(fromByteOffset: 4, as: UInt16.self)
                let nL = raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
                let nE = raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
                let n = raw.loadUnaligned(fromByteOffset: 16, as: UInt64.self)
                return (m, v, nL, nE, n)
            }
        guard magic == JangSpecBundleFormat.indexMagic else {
            throw JangSpecBundleError.invalidIndex(
                String(format: "bad magic 0x%08x", magic))
        }
        guard version == 1 else {
            throw JangSpecBundleError.unsupportedVersion(
                field: "index", value: Int(version), supported: 1)
        }

        let count = Int(nEntries)
        let expectedSize =
            JangSpecBundleFormat.indexHeaderSize + count * JangSpecBundleFormat.indexEntrySize
        guard data.count >= expectedSize else {
            throw JangSpecBundleError.truncated(url, expected: expectedSize, actual: data.count)
        }

        var entries: [ExpertKey: ExpertIndexEntry] = [:]
        entries.reserveCapacity(count)
        data.withUnsafeBytes { raw in
            var cursor = JangSpecBundleFormat.indexHeaderSize
            for _ in 0..<count {
                let layer = raw.loadUnaligned(fromByteOffset: cursor + 0, as: UInt32.self)
                let expert = raw.loadUnaligned(fromByteOffset: cursor + 4, as: UInt32.self)
                let fileID = raw.loadUnaligned(fromByteOffset: cursor + 8, as: UInt16.self)
                // 2 bytes pad
                let offset = raw.loadUnaligned(fromByteOffset: cursor + 12, as: UInt64.self)
                let nbytes = raw.loadUnaligned(fromByteOffset: cursor + 20, as: UInt64.self)
                entries[ExpertKey(layer: Int(layer), expert: Int(expert))] =
                    ExpertIndexEntry(
                        layerIdx: Int(layer),
                        expertID: Int(expert),
                        fileID: Int(fileID),
                        offset: Int(offset),
                        nbytes: Int(nbytes)
                    )
                cursor += JangSpecBundleFormat.indexEntrySize
            }
        }
        return entries
    }

    // MARK: - Blob parsing

    private struct ParsedBlob {
        struct Triple {
            let qweight: MLXArray
            let scales: MLXArray
            let biases: MLXArray
        }
        let layerIdx: Int
        let expertID: Int
        let bits: Int
        let tensors: [JangSpecBundleFormat.TensorKind: Triple]
    }

    private static func parseExpertBlob(_ data: Data) throws -> ParsedBlob {
        guard data.count >= JangSpecBundleFormat.blobHeaderSize else {
            throw JangSpecBundleError.invalidBlob("blob too short")
        }
        let (magic, version, nTensors, layer, expert, payloadOffset, payloadBytes):
            (UInt32, UInt16, UInt16, UInt32, UInt32, UInt64, UInt64) =
            data.withUnsafeBytes { raw in
                let m = raw.loadUnaligned(fromByteOffset: 0, as: UInt32.self)
                let v = raw.loadUnaligned(fromByteOffset: 4, as: UInt16.self)
                let n = raw.loadUnaligned(fromByteOffset: 6, as: UInt16.self)
                let l = raw.loadUnaligned(fromByteOffset: 8, as: UInt32.self)
                let e = raw.loadUnaligned(fromByteOffset: 12, as: UInt32.self)
                let po = raw.loadUnaligned(fromByteOffset: 16, as: UInt64.self)
                let pb = raw.loadUnaligned(fromByteOffset: 24, as: UInt64.self)
                return (m, v, n, l, e, po, pb)
            }
        guard magic == JangSpecBundleFormat.blobMagic else {
            throw JangSpecBundleError.invalidBlob(
                String(format: "bad magic 0x%08x", magic))
        }
        guard version == 1 else {
            throw JangSpecBundleError.unsupportedVersion(
                field: "blob", value: Int(version), supported: 1)
        }
        guard nTensors == 9 else {
            throw JangSpecBundleError.invalidBlob("expected 9 tensor entries, got \(nTensors)")
        }

        let payOff = Int(payloadOffset)
        let payBytes = Int(payloadBytes)
        guard data.count >= payOff + payBytes else {
            throw JangSpecBundleError.invalidBlob("declared payload exceeds data length")
        }

        // Walk 9 tensor headers, accumulate per-kind triples.
        var bitsSeen: Int? = nil
        var collected: [JangSpecBundleFormat.TensorKind: [JangSpecBundleFormat.TensorDType: MLXArray]] = [:]

        for i in 0..<Int(nTensors) {
            let cursor = JangSpecBundleFormat.blobHeaderSize + i * JangSpecBundleFormat.tensorHeaderSize
            let (kindRaw, bitsVal, dtypeRaw, d0, d1, d2, off, nb):
                (UInt8, UInt8, UInt32, UInt32, UInt32, UInt32, UInt64, UInt64) =
                data.withUnsafeBytes { raw in
                    let k = raw.loadUnaligned(fromByteOffset: cursor + 0, as: UInt8.self)
                    let b = raw.loadUnaligned(fromByteOffset: cursor + 1, as: UInt8.self)
                    let dt = raw.loadUnaligned(fromByteOffset: cursor + 4, as: UInt32.self)
                    let x = raw.loadUnaligned(fromByteOffset: cursor + 8, as: UInt32.self)
                    let y = raw.loadUnaligned(fromByteOffset: cursor + 12, as: UInt32.self)
                    let z = raw.loadUnaligned(fromByteOffset: cursor + 16, as: UInt32.self)
                    let o = raw.loadUnaligned(fromByteOffset: cursor + 20, as: UInt64.self)
                    let n = raw.loadUnaligned(fromByteOffset: cursor + 28, as: UInt64.self)
                    return (k, b, dt, x, y, z, o, n)
                }
            guard let kind = JangSpecBundleFormat.TensorKind(rawValue: kindRaw) else {
                throw JangSpecBundleError.invalidBlob("unknown tensor kind \(kindRaw)")
            }
            guard let dtype = JangSpecBundleFormat.TensorDType(rawValue: dtypeRaw) else {
                throw JangSpecBundleError.invalidBlob("unknown tensor dtype \(dtypeRaw)")
            }

            let bi = Int(bitsVal)
            if let prev = bitsSeen {
                if prev != bi {
                    throw JangSpecBundleError.invalidBlob(
                        "mixed bits in one blob: \(prev) vs \(bi)")
                }
            } else {
                bitsSeen = bi
            }

            let start = payOff + Int(off)
            let end = start + Int(nb)
            guard end <= data.count else {
                throw JangSpecBundleError.invalidBlob("tensor slice out of range")
            }

            // Materialize as MLXArray via the appropriate dtype.
            let dims = [Int(d0), Int(d1), Int(d2)].filter { $0 != 0 }
            let arr = try materializeTensor(
                bytes: data.subdata(in: start..<end),
                dtype: dtype,
                shape: dims
            )
            collected[kind, default: [:]][dtype] = arr
        }

        var tensors: [JangSpecBundleFormat.TensorKind: ParsedBlob.Triple] = [:]
        for kind in [
            JangSpecBundleFormat.TensorKind.gate,
            .up,
            .down,
        ] {
            guard let kindMap = collected[kind] else { continue }
            guard let q = kindMap[.qweight],
                  let s = kindMap[.scales],
                  let b = kindMap[.biases]
            else { continue }
            tensors[kind] = ParsedBlob.Triple(qweight: q, scales: s, biases: b)
        }

        return ParsedBlob(
            layerIdx: Int(layer),
            expertID: Int(expert),
            bits: bitsSeen ?? 0,
            tensors: tensors
        )
    }

    /// Materialize a `Data` slice into an `MLXArray` of the right dtype/shape.
    /// uint32 → MLX `.uint32`, float16 → MLX `.float16`.
    private static func materializeTensor(
        bytes: Data,
        dtype: JangSpecBundleFormat.TensorDType,
        shape: [Int]
    ) throws -> MLXArray {
        switch dtype {
        case .qweight:
            // uint32 packed quantized weights.
            let count = bytes.count / 4
            var u32: [UInt32] = []
            u32.reserveCapacity(count)
            bytes.withUnsafeBytes { raw in
                for i in 0..<count {
                    u32.append(raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self))
                }
            }
            return MLXArray(u32, shape)
        case .scales, .biases:
            // float16 — MLX has Float16 support via `.float16` dtype.
            // We stage through Float32 then asType, which is correct but
            // adds a conversion. A future optimization can read raw bytes
            // directly into a Float16 buffer if MLX exposes it.
            let count = bytes.count / 2
            var f32: [Float] = []
            f32.reserveCapacity(count)
            bytes.withUnsafeBytes { raw in
                for i in 0..<count {
                    let bits = raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)
                    f32.append(Float16(bitPattern: bits)._asFloat)
                }
            }
            return MLXArray(f32, shape).asType(.float16)
        }
    }

    // MARK: - Expert grouping

    /// Match the layer index in tensor names like
    /// "model.language_model.layers.7.switch_mlp.gate_proj".
    private static let layerRegex = try! NSRegularExpression(
        pattern: #"\.?layers\.(\d+)\."#)

    private static func layerIndex(of base: String) throws -> Int {
        let range = NSRange(base.startIndex..., in: base)
        guard let match = layerRegex.firstMatch(in: base, range: range),
              match.numberOfRanges >= 2,
              let r = Range(match.range(at: 1), in: base),
              let idx = Int(base[r])
        else {
            throw JangSpecBundleError.missingBaseName(base)
        }
        return idx
    }

    private static func groupExpertBasesByLayer(_ baseNames: [String]) throws
        -> [(Int, [String])]
    {
        var byLayer: [Int: [String]] = [:]
        for base in baseNames {
            let idx = try layerIndex(of: base)
            byLayer[idx, default: []].append(base)
        }
        return byLayer.keys.sorted().map { ($0, byLayer[$0]!) }
    }
}

// MARK: - Float16 helper

extension Float16 {
    /// Bridge to a plain Swift Float so we can build the staging buffer
    /// without pulling in any external numerics library.
    fileprivate var _asFloat: Float {
        return Float(self)
    }
}
```

- [ ] **Step 2: Build to confirm it compiles standalone**

```bash
cd /Users/eric/jang/vmlx-swift-lm && swift build -c release 2>&1 | tail -15
```

Expected: build success. If it fails on `MLX.stacked`, check the actual mlx-swift API name in this fork (it may be `stack`, `stacked`, `MLXNN.stack`, or similar — adapt to whatever the local fork exposes). Same for `MLXArray(u32, shape)` initializer — older mlx-swift uses positional args, newer versions take labeled `(values:shape:)`.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang/vmlx-swift-lm && git add Libraries/MLXLMCommon/JangSpecBundleLoader.swift
git commit -m "jangspec: native Swift bundle loader in MLXLMCommon"
```

---

## Task 3: Wire bundle detection into `Load.swift`

**Files:**
- Modify: `vmlx-swift-lm/Libraries/MLXLMCommon/Load.swift` (insert ~10 lines at the start of `loadWeights`)

- [ ] **Step 1: Edit Load.swift**

Replace the existing weight-loading block in `Load.swift` so it routes bundles through `JangSpecBundleLoader`. The change is bounded: the `.jangspec` branch comes BEFORE the existing JANG v1 / standard safetensors enumeration.

In `loadWeights(modelDirectory:model:...)`, change:

```swift
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()

    // Resolve symlinks (mlxstudio uses symlinked model directories)
    let modelDirectory = modelDirectory.resolvingSymlinksInPath()

    // JANG v1 models use .jang.safetensors files that need uint8->uint32 repacking
    if let jangConfig, !jangConfig.isV2, JangLoader.hasV1Weights(at: modelDirectory) {
        weights = try JangLoader.loadV1Weights(at: modelDirectory)
    } else {
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let (w, m) = try loadArraysAndMetadata(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
                if metadata.isEmpty {
                    metadata = m
                }
            }
        }
    }
```

…to:

```swift
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()

    // Resolve symlinks (mlxstudio uses symlinked model directories)
    let modelDirectory = modelDirectory.resolvingSymlinksInPath()

    // .jangspec bundle: per-expert blobs + hot-core safetensors + flat index.
    // Read everything via JangSpecBundleLoader, which produces a {key: MLXArray}
    // dict identical in shape to the standard safetensors enumeration.
    if JangSpecBundleLoader.isBundle(at: modelDirectory) {
        weights = try JangSpecBundleLoader.loadWeights(from: modelDirectory)
    } else if let jangConfig, !jangConfig.isV2, JangLoader.hasV1Weights(at: modelDirectory) {
        // JANG v1 models use .jang.safetensors files that need uint8->uint32 repacking
        weights = try JangLoader.loadV1Weights(at: modelDirectory)
    } else {
        let enumerator = FileManager.default.enumerator(
            at: modelDirectory, includingPropertiesForKeys: nil)!
        for case let url as URL in enumerator {
            if url.pathExtension == "safetensors" {
                let (w, m) = try loadArraysAndMetadata(url: url)
                for (key, value) in w {
                    weights[key] = value
                }
                if metadata.isEmpty {
                    metadata = m
                }
            }
        }
    }
```

The rest of `loadWeights` is unchanged — `model.sanitize`, `JangLoader.dequantizeMoEGates`, `JangLoader.inferPerLayerQuantization`, `model.update`, the float→bfloat16 conversion all run identically because the bundle loader's output dict has the same keys, dtypes, and shapes.

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/vmlx-swift-lm && swift build -c release 2>&1 | tail -10
```

Expected: build success.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang/vmlx-swift-lm && git add Libraries/MLXLMCommon/Load.swift
git commit -m "jangspec: route .jangspec bundles through JangSpecBundleLoader in loadWeights"
```

---

## Task 4: Defensive fallback in `JangLoader.findConfigPath`

**Files:**
- Modify: `vmlx-swift-lm/Libraries/MLXLMCommon/JangLoader.swift` (extend one method)

- [ ] **Step 1: Edit findConfigPath**

```swift
    /// Find the JANG config file in a model directory.
    public static func findConfigPath(at modelPath: URL) -> URL? {
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        // .jangspec bundles built before Plan 6's builder update only place
        // jang_config.json under target/. Fall back to the bundle layout so
        // those still load without re-building.
        for name in jangConfigFileNames {
            let configURL = modelPath.appendingPathComponent("target")
                .appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: configURL.path) {
                return configURL
            }
        }
        return nil
    }
```

- [ ] **Step 2: Build**

```bash
cd /Users/eric/jang/vmlx-swift-lm && swift build -c release 2>&1 | tail -10
```

Expected: build success.

- [ ] **Step 3: Commit**

```bash
cd /Users/eric/jang/vmlx-swift-lm && git add Libraries/MLXLMCommon/JangLoader.swift
git commit -m "jangspec(JangLoader): findConfigPath falls back to bundle target/ subdir"
```

---

## Task 5: RunBench compile + smoke (no inference)

**Files:** none

- [ ] **Step 1: Build the RunBench product**

```bash
cd /Users/eric/jang/vmlx-swift-lm && swift build -c release --product RunBench 2>&1 | tail -10
```

Expected: build success including the new bundle loader.

- [ ] **Step 2: Confirm the binary launches and prints the model header without doing inference**

We do NOT actually run the bench — Eric runs that when RAM is free. We do confirm the binary links and immediately exits when given a missing model:

```bash
cd /Users/eric/jang/vmlx-swift-lm && BENCH_MODEL=/nonexistent/path BENCH_TOKENS=/nonexistent.json ./.build/release/RunBench 2>&1 | head -20
```

Expected: prints the bench banner, then errors out cleanly on the missing model. We're checking that the binary links against the new code without crashing at startup, NOT that it generates tokens.

- [ ] **Step 3: Confirm the bundle is now detectable as a valid model directory**

```bash
ls -la /tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec/ | head -20
```

Expected: `jangspec.json`, `config.json`, `jang_config.json`, `tokenizer.json`, `tokenizer_config.json`, and a `target/` subdir all present at the bundle root.

---

## Task 6: STATUS update + commit

**Files:**
- Modify: `docs/superpowers/notes/jang-spec-STATUS.md`

- [ ] **Step 1: Update STATUS**

Edit `docs/superpowers/notes/jang-spec-STATUS.md`:

- TL;DR: add a sentence saying jang-spec bundles now load natively in vmlx-swift-lm via `JangSpecBundleLoader`; existing fast runtime + L2 disk cache work transparently with bundle-loaded models
- Plans table: mark Plan 6 **CODE READY (run pending)** with branches `jang-spec-plan6-vmlx-integration` (jang repo) and `jang-spec-bundle-loader` (vmlx-swift-lm repo). Artifacts: `JangSpecBundleLoader.swift`, builder root-config copy, `findConfigPath` fallback
- Tests line: no count change (no new XCTests added; verification is the full RunBench end-to-end run when Eric is ready)
- Immediate next: rewrite to describe Plan 7 — measure RunBench tok/s on a `.jangspec` bundle, compare against vanilla safetensors load, and decide whether the per-blob restack overhead warrants a "fat bundle" optimization
- Add a "Plan 6 notes" subsection summarizing the decision to integrate into vmlx-swift-lm rather than build a parallel JANGCore engine, the format constants duplication strategy, and the load-time trade-off (~1–3 s per 16 GB model from per-blob restack)

- [ ] **Step 2: Commit STATUS**

```bash
cd /Users/eric/jang && git add docs/superpowers/notes/jang-spec-STATUS.md
git commit -m "jang-spec: STATUS update — Plan 6 ready, vmlx-swift-lm reads bundles natively"
```

- [ ] **Step 3: Print plan commit log across both repos**

```bash
echo "=== jang ==="; cd /Users/eric/jang && git log --oneline jang-spec-plan5-bundle-python-validation..HEAD
echo "=== vmlx-swift-lm ==="; cd /Users/eric/jang/vmlx-swift-lm && git log --oneline main^..HEAD
```

Expected: 2 commits in jang (builder update + STATUS), 3 commits in vmlx-swift-lm (loader + Load.swift wiring + JangLoader fallback).

---

## When you're ready to run

The full validation needs RAM. Eric runs it manually:

```bash
cd /Users/eric/jang/vmlx-swift-lm
pkill -9 -f "ollama|omlx|Python.*mlx_lm" 2>/dev/null
BENCH_MODEL=/tmp/jangcore-fixtures/Gemma-4-26B-A4B-it-JANG_4M.jangspec \
    ./.build/release/RunBench
```

Expected: the same multi-turn benchmark output as before (load time, prefill/decode rates per turn, final summary). Token generation against the bundle should produce identical output to a direct load of the source `JANG_4M/` directory because the bundle loader emits a byte-identical weight dict — Plan 5's Python validation script proved this round-trip.

If decode tok/s on the bundle is within ~5% of the source-directory number (which was 101.5 avg / 107.1 peak), the integration is a success and Plan 7 can begin measuring real-world bundle load characteristics.

If decode is materially slower, the most likely cause is the per-blob restack adding to load time (one-time cost) rather than per-token compute (which uses the same mlx-swift kernels regardless of where weights came from). Confirm via the load-time line: "Load: X.Xs" — the bundle path may take 1–3 s longer than vanilla.

---

## What this plan does NOT do

- Run any model. All test runs are deferred to Eric.
- Add any Swift unit tests for `JangSpecBundleLoader` — XCTest cases require MLX device init and small fixture files that we'd need to bake into the test bundle. A future plan can add them; for now the integration test is "RunBench produces correct output on the Gemma fixture."
- Touch `jang-runtime/JANGCore` or `JANGCoreMetal`. Those stay as standalone debug tools (`jang-core inspect`, `jang-core hot-core`, `jang-spec-iobench`). Plan 4's Metal kernel is shelved as research; mlx-swift is the production fast path.
- Modify the `vmlx-swift` mlx-swift fork. Bundle support is purely at the LM layer.
- Touch the existing per-tensor `loadArraysAndMetadata` safetensors loader. The bundle's hot core uses it unchanged.
