import Foundation

struct ChatMessage: Identifiable, Equatable, Codable {
    enum Role: String, Codable { case user, assistant, system }

    let id: UUID
    let role: Role
    let text: String
    let tokensPerSec: Double?
    let elapsedS: Double?
    let imagePath: String?
    let timestamp: Date

    init(role: Role, text: String,
         tokensPerSec: Double? = nil,
         elapsedS: Double? = nil,
         imagePath: String? = nil) {
        self.id = UUID()
        self.role = role
        self.text = text
        self.tokensPerSec = tokensPerSec
        self.elapsedS = elapsedS
        self.imagePath = imagePath
        self.timestamp = Date()
    }
}
