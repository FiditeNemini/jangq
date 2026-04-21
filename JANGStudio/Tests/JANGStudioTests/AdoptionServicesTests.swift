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

    // MARK: - Iter 91 M168: context-aware remediation in publish-error description

    @MainActor
    func test_publish_cliError_401_suggests_token_check() {
        // Common HF response for invalid/expired token. The error description
        // must tell the user HOW to fix it (verify token / check settings),
        // not just show the raw stderr. Mirrors iter-90 M167's "remediation
        // command, not just symptom" pattern.
        let err = PublishServiceError.cliError(
            code: 1,
            stderr: "HfHubHTTPError: 401 Client Error: Unauthorized for url: https://huggingface.co/api/…"
        )
        let desc = err.errorDescription ?? ""
        XCTAssertTrue(desc.contains("token"),
            "401 error must mention the token as the likely cause. Got: \(desc)")
        XCTAssertTrue(desc.contains("huggingface.co/settings/tokens"),
            "401 error must link to the HF tokens settings page. Got: \(desc)")
    }

    @MainActor
    func test_publish_cliError_403_suggests_permission_check() {
        // Forbidden means token is valid but lacks scope for the target repo.
        let err = PublishServiceError.cliError(
            code: 1,
            stderr: "HfHubHTTPError: 403 Client Error: Forbidden — you don't have write access to this repo"
        )
        let desc = err.errorDescription ?? ""
        XCTAssertTrue(desc.contains("permission") || desc.contains("write access"),
            "403 must mention permission / write access. Got: \(desc)")
    }

    @MainActor
    func test_publish_cliError_network_suggests_retry() {
        // Connection errors during upload — user fix is "check network, retry."
        let err = PublishServiceError.cliError(
            code: 1,
            stderr: "ConnectionError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded"
        )
        let desc = err.errorDescription ?? ""
        XCTAssertTrue(desc.contains("network") || desc.contains("retry"),
            "Connection error must suggest network-check + retry. Got: \(desc)")
    }

    @MainActor
    func test_publish_cliError_generic_falls_back_to_generic_hint() {
        // Unknown error shape — fall back to a generic "check token or retry"
        // hint so the user still gets a next-action, not just the raw stderr.
        let err = PublishServiceError.cliError(
            code: 99,
            stderr: "some weird error we've never seen before"
        )
        let desc = err.errorDescription ?? ""
        XCTAssertTrue(desc.contains("weird error we've never seen"),
            "generic fallback must still preserve the raw stderr context")
        // Some form of next-action hint should still appear.
        XCTAssertTrue(desc.contains("token") || desc.contains("network") || desc.contains("retry"),
            "generic fallback should suggest at least one of: token / network / retry. Got: \(desc)")
    }

    @MainActor
    func test_publish_cliError_preserves_stderr_in_all_branches() {
        // Regression guard: the remediation hint is ADDED, not replacing the
        // stderr. The user still needs to see what actually failed.
        let stderr = "HfHubHTTPError: 401 Client Error: Unauthorized"
        let err = PublishServiceError.cliError(code: 1, stderr: stderr)
        let desc = err.errorDescription ?? ""
        XCTAssertTrue(desc.contains(stderr),
            "stderr must still appear in the description — remediation is appended, not substituted")
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

    // MARK: - M214 (iter 144): sanitizeForRepoName — unicode/space prefill
    //
    // Pre-M214 PublishToHuggingFaceSheet's init prefilled the repo
    // field with modelPath.lastPathComponent AS-IS. A user whose source
    // was `~/café-model/` got a prefill that failed HF's ASCII-only
    // regex with a cryptic error. M214's sanitizer turns arbitrary
    // basenames into something the validator accepts.

    @MainActor
    func test_sanitize_ascii_preserves_unchanged() {
        // Already HF-valid — no sanitization needed.
        let r = HFRepoValidator.sanitizeForRepoName("Qwen3-0.6B-Base")
        XCTAssertEqual(r.sanitized, "Qwen3-0.6B-Base")
        XCTAssertFalse(r.wasChanged,
            "ASCII-valid basename must pass through unchanged; "
            + "wasChanged would trigger a misleading 'we sanitized' banner.")
    }

    @MainActor
    func test_sanitize_accents_stripped() {
        // NFD decomposition + ASCII-only filter drops combining marks.
        let r = HFRepoValidator.sanitizeForRepoName("café-model")
        XCTAssertEqual(r.sanitized, "cafe-model",
            "`é` must decompose (NFD) then strip the combining acute, "
            + "leaving `e`. 'café-model' → 'cafe-model'.")
        XCTAssertTrue(r.wasChanged)
        XCTAssertNil(HFRepoValidator.validationError("org/" + r.sanitized),
            "Sanitized output MUST pass validation as a name segment.")
    }

    @MainActor
    func test_sanitize_emoji_replaced() {
        // Emoji isn't in allowed set → replaced with `-`. Leading `-`
        // stripped in trim step.
        let r = HFRepoValidator.sanitizeForRepoName("🍕-pizza")
        XCTAssertFalse(r.sanitized.contains("🍕"))
        XCTAssertEqual(r.sanitized, "pizza",
            "Emoji → `-`; leading-`-` trimmed; result is 'pizza'.")
        XCTAssertNil(HFRepoValidator.validationError("org/" + r.sanitized))
    }

    @MainActor
    func test_sanitize_spaces_to_dashes() {
        let r = HFRepoValidator.sanitizeForRepoName("my model (final)")
        XCTAssertFalse(r.sanitized.contains(" "))
        XCTAssertNil(HFRepoValidator.validationError("org/" + r.sanitized),
            "Spaces + parens must sanitize to something that passes "
            + "validation. Got: \(r.sanitized)")
    }

    @MainActor
    func test_sanitize_collapses_double_dashes() {
        // Consecutive disallowed chars would turn into `--` / `..`, both
        // rejected by the validator. Collapse step must fire.
        let r = HFRepoValidator.sanitizeForRepoName("a  b")   // two spaces
        XCTAssertFalse(r.sanitized.contains("--"),
            "Consecutive replaced chars must be collapsed to a single `-`.")
        XCTAssertNil(HFRepoValidator.validationError("org/" + r.sanitized))
    }

    @MainActor
    func test_sanitize_leading_non_alphanum_prefixed() {
        // Segment regex requires first char [A-Za-z0-9]. If sanitized
        // would start with `.` or `-` or `_`, prefix with `m-`.
        let r = HFRepoValidator.sanitizeForRepoName("_underscore-start")
        XCTAssertTrue(r.sanitized.first!.isLetter || r.sanitized.first!.isNumber,
            "First char must be a letter or digit after sanitization.")
        XCTAssertNil(HFRepoValidator.validationError("org/" + r.sanitized))
    }

    @MainActor
    func test_sanitize_all_invalid_returns_fallback() {
        // Input with NO ASCII alphanumerics → fallback "model".
        let r = HFRepoValidator.sanitizeForRepoName("🍕🍕🍕")
        XCTAssertEqual(r.sanitized, "model")
        XCTAssertTrue(r.wasChanged)
    }

    @MainActor
    func test_sanitize_empty_input_returns_fallback() {
        let r = HFRepoValidator.sanitizeForRepoName("")
        XCTAssertEqual(r.sanitized, "model")
        XCTAssertTrue(r.wasChanged)
    }
}
