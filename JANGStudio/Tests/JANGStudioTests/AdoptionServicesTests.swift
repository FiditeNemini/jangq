import XCTest
@testable import JANGStudio

final class AdoptionServicesTests: XCTestCase {
    func test_example_language_cases_unique() {
        let names = ExampleLanguage.allCases.map { $0.rawValue }
        XCTAssertEqual(Set(names).count, names.count)
        XCTAssertTrue(names.contains("python"))
        XCTAssertTrue(names.contains("swift"))
        XCTAssertTrue(names.contains("server"))
        XCTAssertTrue(names.contains("hf"))
    }

    func test_example_language_display_names() {
        XCTAssertEqual(ExampleLanguage.python.displayName, "Python")
        XCTAssertEqual(ExampleLanguage.hf.displayName, "HuggingFace")
    }

    func test_example_snippet_decodes() throws {
        let json = #"{"lang":"python","snippet":"from jang_tools import x","model":"/m"}"#
        let s = try JSONDecoder().decode(ExampleSnippet.self, from: Data(json.utf8))
        XCTAssertEqual(s.lang, "python")
        XCTAssertEqual(s.model, "/m")
    }

    func test_model_card_result_decodes() throws {
        let json = #"""
        {"license":"apache-2.0","base_model":"Qwen/Qwen3-0.6B","quantization_config":{"family":"jang","profile":"JANG_4K","actual_bits":4.23,"block_size":64,"size_gb":0.3},"card_markdown":"---\nlicense: apache-2.0\n---"}
        """#
        let card = try JSONDecoder().decode(ModelCardResult.self, from: Data(json.utf8))
        XCTAssertEqual(card.license, "apache-2.0")
        XCTAssertEqual(card.quantizationConfig.family, "jang")
        XCTAssertEqual(card.quantizationConfig.actualBits, 4.23, accuracy: 0.01)
    }

    func test_publish_result_dry_run_decodes() throws {
        let json = #"{"dry_run":true,"repo":"my/model","files_count":8,"total_size_bytes":1000000}"#
        let r = try JSONDecoder().decode(PublishResult.self, from: Data(json.utf8))
        XCTAssertTrue(r.dryRun)
        XCTAssertEqual(r.repo, "my/model")
        XCTAssertEqual(r.filesCount, 8)
        XCTAssertNil(r.url)
    }

    func test_publish_service_rejects_empty_token() async {
        do {
            _ = try await PublishService.dryRun(modelPath: URL(fileURLWithPath: "/tmp"), repo: "x/y", isPrivate: false, token: "")
            XCTFail("expected missingToken error")
        } catch let e as PublishServiceError {
            switch e {
            case .missingToken: break // expected
            default: XCTFail("wrong error \(e)")
            }
        } catch {
            XCTFail("wrong error type \(error)")
        }
    }

    // MARK: - Iter 7: M44 commit_url decode + M46 repo-name validation

    func test_publish_result_decodes_commit_url() throws {
        let json = #"""
        {"dry_run":false,"repo":"dealignai/MyModel-JANG_4K","url":"https://huggingface.co/dealignai/MyModel-JANG_4K","commit_url":"https://huggingface.co/dealignai/MyModel-JANG_4K/commit/abc123"}
        """#
        let r = try JSONDecoder().decode(PublishResult.self, from: Data(json.utf8))
        XCTAssertFalse(r.dryRun)
        XCTAssertEqual(r.url, "https://huggingface.co/dealignai/MyModel-JANG_4K")
        XCTAssertEqual(r.commitUrl, "https://huggingface.co/dealignai/MyModel-JANG_4K/commit/abc123")
    }

    func test_publish_result_commit_url_optional() throws {
        // Dry-run responses don't carry a commit_url — make sure decoder tolerates absence.
        let json = #"{"dry_run":true,"repo":"my/m","files_count":3,"total_size_bytes":100}"#
        let r = try JSONDecoder().decode(PublishResult.self, from: Data(json.utf8))
        XCTAssertNil(r.commitUrl)
    }

    @MainActor
    func test_repo_validator_accepts_canonical() {
        XCTAssertNil(HFRepoValidator.validationError("dealignai/MyModel-JANG_4K"))
        XCTAssertNil(HFRepoValidator.validationError("a/b"))
        XCTAssertNil(HFRepoValidator.validationError("org_name/model.v2"))
        XCTAssertNil(HFRepoValidator.validationError("org-name/model-name"))
    }

    @MainActor
    func test_repo_validator_rejects_empty() {
        XCTAssertNotNil(HFRepoValidator.validationError(""))
        XCTAssertNotNil(HFRepoValidator.validationError("   "))
    }

    @MainActor
    func test_repo_validator_rejects_no_slash() {
        let err = HFRepoValidator.validationError("justname")
        XCTAssertNotNil(err)
        XCTAssertTrue(err!.contains("org/model-name"), "got: \(err ?? "nil")")
    }

    @MainActor
    func test_repo_validator_rejects_spaces() {
        let err = HFRepoValidator.validationError("my org/my model")
        XCTAssertNotNil(err)
        XCTAssertTrue(err!.contains("spaces"), "got: \(err ?? "nil")")
    }

    @MainActor
    func test_repo_validator_rejects_double_slash() {
        XCTAssertNotNil(HFRepoValidator.validationError("org//name"))
        XCTAssertNotNil(HFRepoValidator.validationError("org/name/extra"))
    }

    @MainActor
    func test_repo_validator_rejects_leading_special_char() {
        // Segments must START with a letter or digit (no leading `-`, `.`, `_`).
        XCTAssertNotNil(HFRepoValidator.validationError("-badorg/model"))
        XCTAssertNotNil(HFRepoValidator.validationError("org/-badname"))
        XCTAssertNotNil(HFRepoValidator.validationError(".dotfile/model"))
    }

    @MainActor
    func test_repo_validator_rejects_segment_overlength() {
        // Max 96 chars per segment (HF enforced). 97 `a`s should fail.
        let longOrg = String(repeating: "a", count: 97)
        XCTAssertNotNil(HFRepoValidator.validationError("\(longOrg)/model"))
    }

    @MainActor
    func test_repo_validator_trims_whitespace() {
        XCTAssertNil(HFRepoValidator.validationError("  dealignai/MyModel  "))
    }

    // MARK: - Iter 87 M164: align with huggingface_hub's stricter rules.
    //
    // Pre-M164 the Swift validator accepted names that huggingface_hub
    // REJECTS at upload time (after a 30-minute publish dispatch has
    // already started). These additions close that fail-slow gap.

    @MainActor
    func test_repo_validator_rejects_consecutive_dots() {
        // HF forbids ".." anywhere in a segment — prevents directory-traversal
        // style constructions at the filesystem layer on the server.
        XCTAssertNotNil(HFRepoValidator.validationError("org/my..model"),
            "consecutive dots must fail client-side (HF rejects)")
        XCTAssertNotNil(HFRepoValidator.validationError("my..org/name"),
            "consecutive dots in ORG segment must fail client-side")
    }

    @MainActor
    func test_repo_validator_rejects_consecutive_dashes() {
        // HF also forbids "--" anywhere in a segment.
        XCTAssertNotNil(HFRepoValidator.validationError("org/my--model"),
            "consecutive dashes must fail client-side (HF rejects)")
        XCTAssertNotNil(HFRepoValidator.validationError("my--org/name"),
            "consecutive dashes in ORG segment must fail client-side")
    }

    @MainActor
    func test_repo_validator_rejects_trailing_special_char() {
        // HF forbids segments ending with `.` or `-`. Very common user typo —
        // auto-complete dropping a trailing period, or a stray "-" from
        // accidental typing.
        XCTAssertNotNil(HFRepoValidator.validationError("org/my-model-"),
            "trailing dash must fail")
        XCTAssertNotNil(HFRepoValidator.validationError("org/my.model."),
            "trailing dot must fail")
        XCTAssertNotNil(HFRepoValidator.validationError("org-/model"),
            "trailing dash in ORG must fail")
        XCTAssertNotNil(HFRepoValidator.validationError("org./model"),
            "trailing dot in ORG must fail")
    }

    @MainActor
    func test_repo_validator_still_accepts_safe_names_with_specials() {
        // Regression guard: the new rules shouldn't reject legitimate names.
        // Single dots/dashes/underscores inside the segment stay legal.
        XCTAssertNil(HFRepoValidator.validationError("org-name/model-name"))
        XCTAssertNil(HFRepoValidator.validationError("org.name/model.v2"))
        XCTAssertNil(HFRepoValidator.validationError("my_org/model_name"))
        XCTAssertNil(HFRepoValidator.validationError("a-b_c.d/e-f_g.h"))
    }

    // MARK: - Iter 24: M43 — publishWithProgress stream

    @MainActor
    func test_publishWithProgress_rejects_empty_token() async {
        // Stream variant mirrors the non-streaming missingToken contract.
        // Empty token must throw PublishServiceError.missingToken on the
        // FIRST iteration of the stream — no subprocess spawn at all.
        let stream = PublishService.publishWithProgress(
            modelPath: URL(fileURLWithPath: "/tmp"),
            repo: "x/y", isPrivate: false, token: "")
        do {
            for try await _ in stream {
                XCTFail("stream should have thrown before yielding any event")
            }
            XCTFail("stream completed without throwing — expected missingToken")
        } catch let e as PublishServiceError {
            switch e {
            case .missingToken: break // expected
            default: XCTFail("wrong PublishServiceError case \(e)")
            }
        } catch {
            XCTFail("wrong error type \(error)")
        }
    }

    // MARK: - Iter 30: M96 ProcessHandle cancellation semantics

    func test_processHandle_wasCancelled_defaults_false() {
        let h = ProcessHandle()
        XCTAssertFalse(h.wasCancelled)
    }

    func test_processHandle_cancel_sets_flag() {
        let h = ProcessHandle()
        h.cancel()
        XCTAssertTrue(h.wasCancelled,
                      "wasCancelled must be true after cancel() so _streamPublish can distinguish user-cancel from real failure")
    }

    func test_processHandle_cancel_before_set_is_safe() {
        // Race case: cancel() fires before the Process reference lands. Must
        // not crash. set(process:) later should terminate immediately.
        let h = ProcessHandle()
        h.cancel()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/sleep")
        proc.arguments = ["1000"]
        try? proc.run()
        h.set(process: proc)
        // If the race handling works, set() should have called terminate()
        // because _wasCancelled was already true. Give it a beat.
        Thread.sleep(forTimeInterval: 0.5)
        XCTAssertFalse(proc.isRunning,
                       "Process spawned after cancel() must be terminated by set()")
    }

    func test_processHandle_cancel_terminates_running_process() {
        let h = ProcessHandle()
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/bin/sleep")
        proc.arguments = ["1000"]
        try? proc.run()
        h.set(process: proc)
        XCTAssertTrue(proc.isRunning)
        h.cancel()
        // SIGTERM should land within a second on /bin/sleep.
        let deadline = Date().addingTimeInterval(3)
        while proc.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        XCTAssertFalse(proc.isRunning, "cancel() must terminate the held process")
    }

    @MainActor
    func test_publishWithProgress_is_async_stream() {
        // Type-level pin: the API returns AsyncThrowingStream<ProgressEvent, Error>
        // — matches PythonRunner.run()'s shape, which is the invariant
        // iter-24's UI wiring depends on. If the return type ever changes
        // (e.g. to an actor-based alternative), this test will fail to
        // compile — forcing an intentional migration.
        let stream: AsyncThrowingStream<ProgressEvent, Error> =
            PublishService.publishWithProgress(
                modelPath: URL(fileURLWithPath: "/tmp"),
                repo: "x/y", isPrivate: false, token: "tok")
        _ = stream   // silence unused-warning
    }
}
