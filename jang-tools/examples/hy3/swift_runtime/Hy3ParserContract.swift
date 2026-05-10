import Foundation

// Standalone parser contract for Hy3's reasoning/tool text surface.

struct Hy3ParsedOutput: Codable {
    var content: String
    var reasoningContent: String?
    var toolCalls: [Hy3ToolCall]
}

struct Hy3ToolCall: Codable {
    var name: String
    var arguments: [String: String]
}

func takeBetween(_ text: String, _ start: String, _ end: String) -> (String, Range<String.Index>)? {
    guard let startRange = text.range(of: start) else { return nil }
    guard let endRange = text.range(of: end, range: startRange.upperBound..<text.endIndex) else { return nil }
    let value = String(text[startRange.upperBound..<endRange.lowerBound])
    return (value, startRange.lowerBound..<endRange.upperBound)
}

func parseHy3Output(_ text: String) -> Hy3ParsedOutput {
    var working = text
    var reasoning: String?
    if let (value, range) = takeBetween(working, "<think>", "</think>") {
        reasoning = value
        working.removeSubrange(range)
    }

    var calls: [Hy3ToolCall] = []
    if let (block, range) = takeBetween(working, "<tool_calls>", "</tool_calls>") {
        var cursor = block[block.startIndex..<block.endIndex]
        while let start = cursor.range(of: "<tool_call>"),
              let end = cursor.range(of: "</tool_call>", range: start.upperBound..<cursor.endIndex) {
            let raw = String(cursor[start.upperBound..<end.lowerBound])
            if let sep = raw.range(of: "<tool_sep>") {
                let name = String(raw[..<sep.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
                let argsBlob = String(raw[sep.upperBound...])
                calls.append(Hy3ToolCall(name: name, arguments: parseArguments(argsBlob)))
            }
            cursor = cursor[end.upperBound..<cursor.endIndex]
        }
        working.removeSubrange(range)
    }

    return Hy3ParsedOutput(
        content: working.trimmingCharacters(in: .whitespacesAndNewlines),
        reasoningContent: reasoning,
        toolCalls: calls
    )
}

func parseArguments(_ text: String) -> [String: String] {
    var result: [String: String] = [:]
    var cursor = text[text.startIndex..<text.endIndex]
    while let keyStart = cursor.range(of: "<arg_key>"),
          let keyEnd = cursor.range(of: "</arg_key>", range: keyStart.upperBound..<cursor.endIndex),
          let valueStart = cursor.range(of: "<arg_value>", range: keyEnd.upperBound..<cursor.endIndex),
          let valueEnd = cursor.range(of: "</arg_value>", range: valueStart.upperBound..<cursor.endIndex) {
        let key = String(cursor[keyStart.upperBound..<keyEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        let value = String(cursor[valueStart.upperBound..<valueEnd.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        result[key] = value
        cursor = cursor[valueEnd.upperBound..<cursor.endIndex]
    }
    return result
}

let sample = "<think></think><tool_calls><tool_call>search<tool_sep><arg_key>q</arg_key><arg_value>hy3</arg_value></tool_call></tool_calls>"
let parsed = parseHy3Output(CommandLine.arguments.dropFirst().first ?? sample)
let data = try JSONEncoder().encode(parsed)
print(String(data: data, encoding: .utf8)!)

