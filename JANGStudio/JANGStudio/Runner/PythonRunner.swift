// JANGStudio/JANGStudio/Runner/PythonRunner.swift
import Foundation

struct ProcessError: Error, Equatable {
    let code: Int32
    let lastStderr: String
}

actor PythonRunner {
    // `let` properties are safe to read from any concurrency domain.
    nonisolated let executable: URL
    nonisolated let extraArgs: [String]
    private var process: Process?
    private var cancelled = false

    init(executableOverride: URL? = nil, extraArgs: [String]) {
        self.executable = executableOverride ?? BundleResolver.pythonExecutable
        self.extraArgs = extraArgs
    }

    // `nonisolated` — only reads immutable `let` properties, safe from any context.
    nonisolated func run() -> AsyncThrowingStream<ProgressEvent, Error> {
        let executable = self.executable
        let extraArgs = self.extraArgs
        return AsyncThrowingStream { continuation in
            Task.detached {
                await PythonRunner.launch(
                    executable: executable,
                    extraArgs: extraArgs,
                    continuation: continuation
                )
            }
        }
    }

    // Static helper — no actor isolation needed, all params are Sendable values.
    private static func launch(
        executable: URL,
        extraArgs: [String],
        continuation: AsyncThrowingStream<ProgressEvent, Error>.Continuation
    ) async {
        let proc = Process()
        proc.executableURL = executable
        proc.arguments = extraArgs
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        // Drain stdout (not forwarded as ProgressEvent).
        let stdoutTask = Task.detached {
            for try await _ in outPipe.fileHandleForReading.bytes.lines {}
        }

        // Drain stderr — parse each line into ProgressEvents.
        // Parser and lastErrTail are owned exclusively by this task, avoiding
        // the Swift 6 shared-mutable-state error from the plan's original layout.
        let stderrTask = Task.detached {
            let parser = JSONLProgressParser()
            var lastErrTail = ""
            for try await line in errPipe.fileHandleForReading.bytes.lines {
                lastErrTail = String(line.suffix(256))
                if let ev = parser.parse(line: line) {
                    continuation.yield(ev)
                }
            }
            return lastErrTail
        }

        do {
            try proc.run()
        } catch {
            continuation.finish(throwing: error)
            return
        }

        proc.waitUntilExit()
        _ = await stdoutTask.result
        let lastErrTail = (try? await stderrTask.value) ?? ""

        if proc.terminationStatus == 0 {
            continuation.finish()
        } else {
            continuation.finish(
                throwing: ProcessError(
                    code: proc.terminationStatus,
                    lastStderr: lastErrTail
                )
            )
        }
    }

    func cancel() {
        cancelled = true
        guard let proc = process, proc.isRunning else { return }
        proc.terminate()   // SIGTERM
        Task.detached {
            try? await Task.sleep(for: .seconds(3))
            if proc.isRunning { kill(proc.processIdentifier, SIGKILL) }
        }
    }
}
