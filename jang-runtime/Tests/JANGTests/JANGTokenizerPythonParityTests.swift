//
// M208 (iter 141): Swift JANGTokenizer ↔ Python AutoTokenizer parity test.
//
// The Swift jang-runtime ships its own BPE tokenizer implementation
// (JANGTokenizer.swift). If its output diverges from Python's
// transformers AutoTokenizer on the SAME tokenizer.json, inference
// against a JANG bundle will feed the model slightly-wrong input
// tokens and produce garbage/degraded output — invisible to users
// (who see plausible but subtly-off generation) and invisible to
// unit tests that only exercise the Swift side.
//
// This test tokenizes known strings with the Swift implementation
// against a real model fixture (Qwen3.6-35B-A3B-JANG_2L), and
// compares token-by-token against hardcoded reference IDs captured
// from `transformers.AutoTokenizer` Python (iter-141, 2026-04-20).
//
// The test SKIPS if the fixture isn't present — it can't be in-tree
// because the tokenizer.json is ~10 MB and the full model is hundreds
// of GB. Developers with the model pulled run the test against it;
// CI that lacks the fixture will pass the skip without a failing
// red bar. If the skip rate rises, consider adding the fixture to
// a test-only HF repo that CI pulls.
//

import Foundation
import XCTest
@testable import JANG

final class JANGTokenizerPythonParityTests: XCTestCase {

    /// The path to a real Qwen3.6-35B-A3B-JANG_2L bundle with a
    /// tokenizer.json + tokenizer_config.json. Skip if not present.
    /// Override with JANG_TEST_QWEN36_BUNDLE env var.
    private static let fixturePath: URL = {
        if let env = ProcessInfo.processInfo.environment["JANG_TEST_QWEN36_BUNDLE"] {
            return URL(fileURLWithPath: env)
        }
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent("models/Qwen3.6-35B-A3B-JANG_2L")
    }()

    /// Reference token IDs captured from Python's transformers
    /// AutoTokenizer.encode(text, add_special_tokens=False) against
    /// the SAME tokenizer.json iter-141 runs Swift against. If Qwen
    /// publishes a tokenizer update that changes these IDs, the
    /// fixture path needs to be updated alongside the reference IDs.
    ///
    /// Captured 2026-04-20 by:
    ///   python3 -c "from transformers import AutoTokenizer;
    ///               tok = AutoTokenizer.from_pretrained(FIXTURE);
    ///               for s in STRINGS: print(s, tok.encode(s, add_special_tokens=False))"
    private static let pythonReference: [(String, [Int])] = [
        ("Hello, world!", [9419, 11, 1814, 0]),
        ("The quick brown fox jumps over the lazy dog.",
         [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13]),
        ("What is 2 + 2?",
         [3710, 369, 220, 17, 478, 220, 17, 30]),
    ]

    /// Skip the whole suite if the fixture isn't present. XCTest
    /// reports skips as neutral (not red), so CI that lacks the
    /// fixture doesn't fail — but local dev runs do exercise the
    /// parity check.
    private func skipIfNoFixture() throws {
        let fm = FileManager.default
        let tokPath = Self.fixturePath.appendingPathComponent("tokenizer.json")
        let cfgPath = Self.fixturePath.appendingPathComponent("tokenizer_config.json")
        guard fm.fileExists(atPath: tokPath.path),
              fm.fileExists(atPath: cfgPath.path) else {
            throw XCTSkip(
                "Parity test fixture not present at "
                + "\(Self.fixturePath.path). Download Qwen3.6-35B-A3B-JANG_2L "
                + "locally (or adjust fixturePath) to exercise Swift↔Python "
                + "tokenizer parity."
            )
        }
    }

    /// Core parity: for each (string, python_ids) pair, Swift's
    /// JANGTokenizer.encode MUST produce the same IDs. Any divergence
    /// = wrong input tokens fed to the model = degraded output.
    func test_swift_tokenizer_matches_python_reference() throws {
        try skipIfNoFixture()
        let tokPath = Self.fixturePath.appendingPathComponent("tokenizer.json")
        let tok = try JANGTokenizer(tokenizerPath: tokPath)

        for (text, expected) in Self.pythonReference {
            let swiftIds = tok.encode(text)
            XCTAssertEqual(
                swiftIds, expected,
                "Swift tokenizer diverged from Python reference on "
                + "\"\(text)\". Python: \(expected). Swift: \(swiftIds). "
                + "This is a M208 parity regression — same input, different "
                + "token IDs, model will receive wrong input. Fix in "
                + "JANGTokenizer.swift before shipping."
            )
        }
    }

    /// Regression guard: Python's AutoTokenizer will NEVER emit an ID
    /// outside the vocab. Swift must match.
    func test_swift_tokenizer_only_emits_valid_ids() throws {
        try skipIfNoFixture()
        let tokPath = Self.fixturePath.appendingPathComponent("tokenizer.json")
        let tok = try JANGTokenizer(tokenizerPath: tokPath)
        for (text, _) in Self.pythonReference {
            let ids = tok.encode(text)
            XCTAssertTrue(!ids.isEmpty, "encode returned empty for \"\(text)\"")
            for id in ids {
                XCTAssertTrue(id >= 0 && id < tok.vocabSize,
                    "Swift tokenizer emitted out-of-vocab ID \(id) for "
                    + "\"\(text)\" (vocabSize=\(tok.vocabSize)). This "
                    + "should be impossible — a bug in encode() or bpeEncode().")
            }
        }
    }

    // MARK: - M216 (iter 145): Swift decode parity with Python
    //
    // M208 (iter 141) pinned ENCODE parity: same string → same IDs.
    // M216 pins the complementary DECODE parity: same IDs → same
    // string. Without this, Swift could:
    //   - Round-trip (decode(encode(s)) == s) but produce different
    //     output than Python on the SAME IDs (e.g. leaking special
    //     tokens as text, getting byte-fallback wrong, mis-assembling
    //     multi-byte chars).
    //   - Or fail round-trip but match Python's broken output (hiding
    //     a shared bug behind a "looks identical" test).
    //
    // The capture was taken iter-145 against the same fixture as M208
    // — the Python output is guaranteed to round-trip on these strings
    // (verified live: all 3 reference pairs round-tripped). So Swift
    // MUST produce identical decode output, which implies Swift MUST
    // round-trip too.
    //
    // Captured 2026-04-20 by:
    //   python3 -c "from transformers import AutoTokenizer;
    //               tok = AutoTokenizer.from_pretrained(FIXTURE);
    //               for s in STRINGS: print(repr(tok.decode(
    //                   tok.encode(s, add_special_tokens=False),
    //                   skip_special_tokens=True)))"

    /// The Python reference from M208 ALREADY round-trips
    /// (verified live). So for each (text, ids) pair, Python
    /// `tok.decode(ids) == text`. Swift must produce the same.
    func test_swift_decode_matches_python_on_same_ids() throws {
        try skipIfNoFixture()
        let tokPath = Self.fixturePath.appendingPathComponent("tokenizer.json")
        let tok = try JANGTokenizer(tokenizerPath: tokPath)

        for (expected, ids) in Self.pythonReference {
            let decoded = tok.decode(ids)
            XCTAssertEqual(
                decoded, expected,
                "Swift decode diverged from Python on ids=\(ids). "
                + "Python.decode(ids) == \(expected.debugDescription). "
                + "Swift.decode(ids) == \(decoded.debugDescription). "
                + "This is a M216 parity regression — same IDs, different "
                + "text output. User-visible symptom: model output rendered "
                + "differently in Swift UI vs. the Python test-inference "
                + "path. Fix in JANGTokenizer.swift decode() before shipping."
            )
        }
    }

    /// Round-trip: encode then decode must recover the original
    /// string. Complementary to the cross-language test — catches
    /// bugs where Swift encode AND decode both drift but happen to
    /// cancel each other out against Python (implausible but
    /// worth pinning).
    func test_swift_encode_decode_roundtrips() throws {
        try skipIfNoFixture()
        let tokPath = Self.fixturePath.appendingPathComponent("tokenizer.json")
        let tok = try JANGTokenizer(tokenizerPath: tokPath)
        for (original, _) in Self.pythonReference {
            let ids = tok.encode(original)
            let roundTripped = tok.decode(ids)
            XCTAssertEqual(
                roundTripped, original,
                "Swift encode→decode round-trip lost content: "
                + "\(original.debugDescription) → ids → "
                + "\(roundTripped.debugDescription). "
                + "Byte-level BPE + byte-fallback must be inverse "
                + "operations. Divergence means either the byteEncoder "
                + "map or the merges table is asymmetric vs. the "
                + "canonical HF tokenizer."
            )
        }
    }
}
