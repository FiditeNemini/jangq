import XCTest
@testable import JANGStudio

final class ChatMessageTests: XCTestCase {
    func test_chat_message_encodes_and_decodes() throws {
        let msg = ChatMessage(role: .user, text: "Hello", tokensPerSec: 42.0, elapsedS: 0.1)
        let data = try JSONEncoder().encode(msg)
        let decoded = try JSONDecoder().decode(ChatMessage.self, from: data)
        XCTAssertEqual(decoded.role, .user)
        XCTAssertEqual(decoded.text, "Hello")
        XCTAssertEqual(decoded.tokensPerSec, 42.0)
    }

    func test_role_raw_values() {
        XCTAssertEqual(ChatMessage.Role.user.rawValue, "user")
        XCTAssertEqual(ChatMessage.Role.assistant.rawValue, "assistant")
        XCTAssertEqual(ChatMessage.Role.system.rawValue, "system")
    }
}
