// JANGStudio/JANGStudio/Runner/PythonRunner.swift
import Foundation

struct ProcessError: Error, Equatable {
    let code: Int32
    let lastStderr: String
}

actor PythonRunner {
    nonisolated let executable: URL
    nonisolated let extraArgs: [String]
    private var process: Process?
    private var cancelled = false

    init(executableOverride: URL? = nil, extraArgs: [String]) {
        self.executable = executableOverride ?? BundleResolver.pythonExecutable
        self.extraArgs = extraArgs
    }

    // `nonisolated` — can be called from any context; the detached Task
    // inside hops back to the actor via `await self.launch(...)`.
    nonisolated func run() -> AsyncThrowingStream<ProgressEvent, Error> {
        AsyncThrowingStream { continuation in
            Task.detached {
                await self.launch(continuation: continuation)
            }
        }
    }

    private func launch(
        continuation: AsyncThrowingStream<ProgressEvent, Error>.Continuation
    ) async {
        let proc = Process()
        proc.executableURL = executable
        proc.arguments = extraArgs
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        // Merge user-settings-driven env vars (tick throttle, thread count,
        // PYTHONPATH prepend). These flow from AppSettings → UserDefaults
        // leaf keys → BundleResolver.childProcessEnvAdditions. See M62 in
        // the Ralph audit log.
        for (k, v) in BundleResolver.childProcessEnvAdditions(inherited: env) {
            env[k] = v
        }
        proc.environment = env

        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        self.process = proc

        // Drain stdout (logs, not yielded as ProgressEvents).
        let stdoutTask = Task.detached {
            for try await _ in outPipe.fileHandleForReading.bytes.lines {}
        }

        // Drain stderr — parser + lastErrTail are owned exclusively by this task
        // (avoids the Swift 6 shared-mutable-state error).
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

        // Wait for termination off the actor thread so cancel() can acquire
        // the actor and call proc.terminate() while we are waiting.
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            proc.terminationHandler = { _ in done.resume() }
        }

        _ = await stdoutTask.result
        let lastErrTail = (try? await stderrTask.value) ?? ""

        if proc.terminationStatus == 0 {
            continuation.finish()
        } else if cancelled {
            continuation.finish()    // cancellation is a clean exit
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
