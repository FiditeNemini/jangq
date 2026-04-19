# JSONL Progress Protocol v1

`jang-tools` emits one JSON object per line on **stderr** when invoked with `--progress=json`. Stdout still carries human-readable output unless `--quiet-text` is also passed.

All numbers are JSON numbers. `ts` is unix seconds with subsecond precision. `v` is the protocol version (1 today); clients must refuse `v` values they don't recognize and prompt the user to upgrade.

## Event types

### `phase`

```json
{"v":1,"type":"phase","n":1,"total":5,"name":"detect","ts":1700000000.123}
```

Fires at the start of each top-level phase. `n/total` drives the coarse progress bar.

### `tick`

```json
{"v":1,"type":"tick","done":1234,"total":2630,"label":"layer.5.gate_proj","ts":...}
```

Per-tensor progress inside long-running phases. Throttled to ~10/s; the final tick with `done == total` is always emitted.

### `info` / `warn` / `error`

```json
{"v":1,"type":"warn","msg":"No chat template found","ts":...}
```

Human-readable messages. `warn` and `error` lines also reach stdout even under `--quiet-text`.

### `done`

```json
{"v":1,"type":"done","ok":true,"output":"/path/to/out","elapsed_s":712.4,"ts":...}
{"v":1,"type":"done","ok":false,"error":"OOM while loading experts","ts":...}
```

Exactly one `done` event per run. `ok:false` runs include `error` (human-readable). Success runs include `output` (final directory) and `elapsed_s`.

## Invoking

```
python -m jang_tools --progress=json --quiet-text convert <src> -o <out> -p JANG_4K
python -m jang_tools.convert_qwen35_jangtq --progress=json --quiet-text <src> <out> JANGTQ2
python -m jang_tools.convert_minimax_jangtq --progress=json --quiet-text <src> <out> JANGTQ2
```

## Extending

Additive: new optional fields can be added without a version bump. Any breaking change increments `v`. Clients parsing unknown `type` values should treat them as `info`-level events.

Created by Jinho Jang (`eric@jangq.ai`).
