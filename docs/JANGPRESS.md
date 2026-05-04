# JangPress — cold-tier MoE memory policy

JangPress is the load-time memory policy that lets routed-MoE bundles
larger than RAM still serve from a single Apple-Silicon Mac. It works by
combining three OS-level techniques on top of the standard MLX +
`safetensors` load path:

1. **mmap-backed safetensors** — every tensor is a page-aligned no-copy
   `MTLBuffer` mapped from the on-disk file. Resident set is whatever
   the kernel decides to keep paged in, not the full bundle.
2. **`madvise(MADV_DONTNEED)` over canonical routed-expert pages** — at
   load time and per-token, JangPress tells the kernel which expert
   pages are cold, freeing physical memory while the file mapping
   stays addressable. A re-fault simply re-reads from disk.
3. **Router-aware hot set** (optional) — for runtimes that surface
   selected expert ids each token, JangPress maintains a per-layer hot
   set and trims experts beyond the budget to cold.

The result: a 167 GB Kimi-K2.6 JANGTQ bundle loads on a 128 GB Mac with
post-load resident around 1 GB. Decode is slower under heavy eviction
(seconds-per-token under stress), but the bundle is *runnable*, not
just *loadable*. JangPress is OS mmap/page reclaim — not custom
compressed expert blobs — so there is no proprietary on-disk format
involved.

## Where the implementation lives

The Swift implementation is upstream in
[`osaurus-ai/vmlx-swift-lm`](https://github.com/osaurus-ai/vmlx-swift-lm)
under `Libraries/MLXLMCommon/Cache/JangPress*.swift`. The full typed-API
surface — `LoadConfiguration`, `JangPressPolicy`, `JangPressRuntime`,
status reporting, env-var fallbacks, advanced router-advice knobs —
is documented in
[`vmlx-swift-lm/docs/JANGPRESS.md`](https://github.com/osaurus-ai/vmlx-swift-lm/blob/main/docs/JANGPRESS.md).

This repository ships the **JANG-side runtime tooling** that pairs with
JangPress: the Python serve/MMLU harness for Kimi-K2.6 JANGTQ bundles
and the shadow-config builder that lets older `vmlxctl` releases load
text-only Kimi without needing a full Swift rebuild.

## Swift quickstart

```swift
import MLXLMCommon

// Production default — auto-detect, env-fallback on, 70% resident cap.
var config = LoadConfiguration.default

// Or pin the policy explicitly from settings:
config.jangPress = .enabled(coldFraction: 0.70)

// Or disable for byte-compat with pre-JangPress behaviour:
config.jangPress = .disabled
config.maxResidentBytes = .unlimited

let (context, runtime) = try await loadModel(
    from: bundleURL,
    using: tokenizerLoader,
    loadConfiguration: config)

let status = runtime.status()
print("JangPress: enabled=\(status.enabled) cold=\(status.coldFraction ?? 0)")
```

## Python / `vmlxctl` quickstart

The JANG repo ships ready-to-use scripts under
[`scripts/jangpress/`](../scripts/jangpress/) for serving a Kimi-K2.6
JANGTQ bundle and benchmarking it via the `/v1/chat/completions` API.

Two terminals (set `VMLXCTL` if `vmlxctl` is not on `$PATH`):

```bash
# Terminal 1 — serve a 167 GB bundle on a 128 GB Mac
cd scripts/jangpress
./kimi_serve.sh ~/.mlxstudio/models/JANGQ-AI/Kimi-K2.6-Med-JANGTQ 100 8082

# Terminal 2 — MMLU 200q (after the serve says "ready")
./kimi_mmlu.sh Kimi-K2.6-Med-JANGTQ chat 8082
```

See [`scripts/jangpress/README.md`](../scripts/jangpress/README.md) for
the full env-var reference (`VMLXCTL`, `PY`, `SHADOW_ROOT`,
`VMLX_MEMORY_BUDGET_OVERRIDE`, `JANGPRESS_PRESTACK`, `KIMI_LOW_RAM`,
`KIMI_JANGPRESS_FORCE_MODE`, `KIMI_ROUTER_ADVICE`, …) and the
troubleshooting cheatsheet.

## When JangPress helps

| Scenario | JangPress impact |
|---|---|
| Routed-MoE bundle > RAM (Kimi-K2.6, MiniMax M2.7, DSV4-Flash 2L/JANGTQ) | Required to fit; otherwise OOM. |
| Routed-MoE bundle ≈ RAM (DSV4-Flash JANG_2L on 192 GB) | Helpful — keeps idle RSS around 1 GB so the rest of the system stays responsive. |
| Routed-MoE bundle ≪ RAM | Optional — `JANGPRESS=disabled` is byte-compatible with the pre-JangPress path; both work. |
| Dense JANGTQ bundle | Not the target. JangPress' cold-tier ABI is per-routed-expert; dense weights are loaded the standard MLX way. |

## How it interacts with JANG bundles

JANG bundles already pack routed experts as JANGTQ tiles aligned for
direct mmap (no `MLX.stacked` materialization). The cache-side prestack
overlay generated on first load wires the on-disk tile layout to the
shape Metal expects without forcing the routed weights resident.

`JANGPRESS_PRESTACK_CACHE_DIR` controls where this overlay lives. It is
~150 GB per Kimi variant — point it at an internal SSD with headroom,
or `~/Library/Caches/jangpress` if you have the space.

## Validation status (2026-05-04)

- `Kimi-K2.6-Small-JANGTQ` (153 GB) — load-validates on M4 Max 128 GB
  with 12,660 routed expert tiles / 129.8 GB under JangPress
  management and about 0.7 GB post-load footprint.
- `Kimi-K2.6-Med-JANGTQ` (167 GB) — same load path; first-inference
  refault on 128 GB hosts is sensitive to memory pressure (use
  `KIMI_LOW_RAM=1` and `JANGPRESS_PRESTACK=1` for the first pass).
- `DSV4-Flash JANG_2L` (96.6 GB) — fully resident on 128 GB+ hosts;
  JangPress is opt-in for these and primarily reduces idle RSS.

For the current upstream validation matrix see the JangPress doc in
`vmlx-swift-lm`. This page is a JANG-side index, not the source of
truth for the Swift API.
