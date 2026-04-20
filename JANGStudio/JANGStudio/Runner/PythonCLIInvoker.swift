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
    ///   - errorFactory: closure invoked on non-zero exit. Receives the
    ///     terminationStatus and captured stderr; should return the
    ///     service-specific typed error.
    static func invoke(
        args: [String],
        executableOverride: URL? = nil,
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
                        let out = Pipe()
                        let err = Pipe()
                        proc.standardOutput = out
                        proc.standardError = err
                        try proc.run()
                        handle.set(process: proc)
                        proc.waitUntilExit()
                        if proc.terminationStatus != 0 {
                            let stderr = String(
                                data: err.fileHandleForReading.readDataToEndOfFile(),
                                encoding: .utf8
                            ) ?? ""
                            cont.resume(throwing: errorFactory(proc.terminationStatus, stderr))
                            return
                        }
                        cont.resume(returning: out.fileHandleForReading.readDataToEndOfFile())
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
