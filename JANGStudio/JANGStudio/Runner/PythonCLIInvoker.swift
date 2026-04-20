// JANGStudio/JANGStudio/Runner/PythonCLIInvoker.swift
//
// M153 (iter 76): shared subprocess-invocation helper for the 5 adoption
// services (RecommendationService, ExamplesService, ModelCardService,
// CapabilitiesService, ProfilesService).
//
// Each of those services had its own nearly-identical private
// `invokeCLI(args:)` — 31-37 lines of the same ProcessHandle +
// withTaskCancellationHandler + DispatchQueue + waitUntilExit dance.
// Iter-51 M129 aligned their typed-error shapes. Iter-75 M152 extracted
// the Python-side read-side-loader helper after 5 local copies crossed
// the extract threshold. This iter applies the same crystallization
// Swift-side.
//
// Design:
//   - Caller passes an `errorFactory` closure so the helper stays
//     service-agnostic (each service's typed error enum is captured
//     at the call site, not leaked into the helper).
//   - Matches the iter-33 M101 Task-cancel pattern and iter-59 M137
//     intent-vs-outcome semantics.
//
// Ref: ralph_runner/INVESTIGATION_LOG.md iter 76.

import Foundation

/// Mutable box for carrying captured pipe output between the drain
/// blocks and the reader, while satisfying Swift 6 concurrency.
/// Each `DataBox` instance is written by exactly one drain block and
/// read only AFTER that block's `DispatchSemaphore.signal()` has been
/// awaited — the happens-before from the semaphore makes the read safe.
/// Swift 6 can't infer that ordering statically, so the class is
/// `@unchecked Sendable` to silence it. (M160, iter 83.)
private final class DataBox: @unchecked Sendable {
    var data: Data = Data()
}

enum PythonCLIInvoker {
    /// Invoke the bundled Python CLI with the given args and return stdout
    /// as Data. On non-zero exit, calls `errorFactory(code, stderr)` to
    /// produce a typed error and throws it.
    ///
    /// Cancellation: wrapped in `withTaskCancellationHandler` +
    /// `ProcessHandle` so a consumer-Task cancel propagates SIGTERM
    /// (with SIGKILL escalation) to the subprocess. Matches iter-33 M101
    /// and the cross-layer sweep in iters 30-34.
    ///
    /// - Parameters:
    ///   - args: full argv to pass to the python binary (including `-m jang_tools …`).
    ///   - executableOverride: test-only override for the executable URL.
    ///     Production code passes nil (defaults to `BundleResolver.pythonExecutable`);
    ///     tests pass a short-sleep / echo shell-script URL to exercise
    ///     cancel, error-factory invocation, and stdout-round-trip paths
    ///     without needing an actual Python + model. Mirrors the
    ///     `executableOverride` pattern on PythonRunner / InferenceRunner
    ///     (iter-31 M98 / iter-32 M100).
    ///   - env: optional environment-variable overrides merged on top of
    ///     the inherited parent env. Used by PublishService (iter-79 M156)
    ///     to thread HF_HUB_TOKEN into the subprocess without leaking via
    ///     ``ps aux``-visible argv. When nil, subprocess inherits parent
    ///     env unchanged (same behavior as iter-76 M153's original shape).
    ///   - errorFactory: closure invoked on non-zero exit. Receives the
    ///     terminationStatus and captured stderr; should return the
    ///     service-specific typed error. Callers that need stderr
    ///     sanitization (e.g., token redaction) do it inside this
    ///     closure before wrapping the typed error.
    static func invoke(
        args: [String],
        executableOverride: URL? = nil,
        env: [String: String]? = nil,
        errorFactory: @escaping @Sendable (Int32, String) -> Error
    ) async throws -> Data {
        let handle = ProcessHandle()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { cont in
                DispatchQueue.global().async {
                    do {
                        let proc = Process()
                        proc.executableURL = executableOverride ?? BundleResolver.pythonExecutable
                        proc.arguments = args
                        if let env = env {
                            proc.environment = env
                        }
                        let out = Pipe()
                        let err = Pipe()
                        proc.standardOutput = out
                        proc.standardError = err

                        // M160 (iter 83): drain both pipes in parallel on
                        // separate dispatch threads BEFORE `proc.run()`.
                        // Before this fix, `waitUntilExit()` ran on the
                        // same thread that then read the pipes — so a
                        // subprocess that wrote more than the 64 KB macOS
                        // pipe buffer blocked on write(2), couldn't exit,
                        // and `waitUntilExit()` deadlocked forever. Seven
                        // callers run the bug risk: 5 adoption services
                        // (discover, examples, capabilities, recommend,
                        // profiles, model-card), SourceDetector, and
                        // PublishService.invoke. Most emit small output,
                        // but Python tracebacks and `examples --list` on
                        // large registries can cross 64 KB.
                        //
                        // Matches iter-81 M158 (runJangValidate) and
                        // iter-82 M159 (InferenceRunner). Drain-pattern #2
                        // from the three-pattern rule: whole-buffer readers
                        // on a separate thread, synchronized via semaphore.
                        let outBox = DataBox()
                        let errBox = DataBox()
                        let outDone = DispatchSemaphore(value: 0)
                        let errDone = DispatchSemaphore(value: 0)
                        DispatchQueue.global().async {
                            outBox.data = out.fileHandleForReading.readDataToEndOfFile()
                            outDone.signal()
                        }
                        DispatchQueue.global().async {
                            errBox.data = err.fileHandleForReading.readDataToEndOfFile()
                            errDone.signal()
                        }

                        try proc.run()
                        handle.set(process: proc)
                        proc.waitUntilExit()
                        // Wait for both drains to complete. EOF fires as
                        // soon as the subprocess's write-end of each pipe
                        // is closed (which happens at exit), so these
                        // waits return shortly after waitUntilExit.
                        outDone.wait()
                        errDone.wait()
                        if proc.terminationStatus != 0 {
                            let stderr = String(data: errBox.data, encoding: .utf8) ?? ""
                            cont.resume(throwing: errorFactory(proc.terminationStatus, stderr))
                            return
                        }
                        cont.resume(returning: outBox.data)
                    } catch {
                        cont.resume(throwing: error)
                    }
                }
            }
        } onCancel: {
            handle.cancel()
        }
    }
}
