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
}
