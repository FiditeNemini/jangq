// JANGStudio/JANGStudio/Models/ProgressEvent.swift
import Foundation

enum EventType: String, Codable {
    case phase, tick, info, warn, error, done
}

struct ProgressEvent: Equatable {
    enum Payload: Equatable {
        case phase(n: Int, total: Int, name: String)
        case tick(done: Int, total: Int, label: String?)
        case message(level: String, text: String)
        case done(ok: Bool, output: String?, error: String?)
        case versionMismatch(Int)
        case parseError(String)
    }
    let ts: TimeInterval
    let type: EventType
    let payload: Payload
}
