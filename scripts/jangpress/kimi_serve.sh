#!/usr/bin/env bash
# Serve a Kimi-K2.6 JANGTQ bundle via vmlxctl with JANGPress mmap eviction.
# Builds a shadow dir on the fly so vMLX's LLM factory sees a flat config
# (works around the kimi_k2 / text_config wrapper bug — see
# kimi_jangpress_agent_fix.patch for the proper Swift fix).
#
# Usage:
#   ./kimi_serve.sh <bundle-dir> [compress-pct=100] [port=8082]
#
# Examples:
#   ./kimi_serve.sh ~/.mlxstudio/models/JANGQ-AI/Kimi-K2.6-Small-JANGTQ 70 8082
set -euo pipefail

BUNDLE="${1:?bundle dir required}"
PCT="${2:-100}"
PORT="${3:-8082}"

if [[ ! -d "$BUNDLE" ]]; then
    echo "FATAL: $BUNDLE not a directory" >&2
    exit 2
fi

VMLXCTL="${VMLXCTL:-$HOME/vmlx/swift/.build/arm64-apple-macosx/release/vmlxctl}"
DEFAULT_PY="$HOME/jang/.venv/bin/python"
PY="${KIMI_PY:-${PY:-$DEFAULT_PY}}"
if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
    PY="${DEFAULT_PY:-python3}"
fi
if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
    PY="python3"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v "$VMLXCTL" >/dev/null 2>&1 && [[ ! -x "$VMLXCTL" ]]; then
    echo "FATAL: \$VMLXCTL ('$VMLXCTL') not found." >&2
    echo "  Build vmlxctl from osaurus-ai/vmlx-swift-lm or set VMLXCTL=<path>." >&2
    exit 2
fi
if ! command -v "$PY" >/dev/null 2>&1 && [[ ! -x "$PY" ]]; then
    echo "FATAL: python ('$PY') not found." >&2
    echo "  Set KIMI_PY=<path> or PY=<path> to a working Python." >&2
    exit 2
fi

# Build shadow dir (config flattened, jang_config.has_vision=false).
SHADOW_ROOT="${SHADOW_ROOT:-/tmp/kimi-shadow}"
echo "[kimi-serve] building shadow dir for $(basename "$BUNDLE")..."
"$PY" "$SCRIPT_DIR/build_kimi_shadow.py" "$BUNDLE" "$SHADOW_ROOT"
SHADOW="$SHADOW_ROOT/$(basename "$BUNDLE")"

# Memory budget override — bypass the load gate's 1.3× peak check
# (167 GB Kimi on 128 GB box; JANGPress mmap makes this fine).
export VMLX_MEMORY_BUDGET_OVERRIDE="${VMLX_MEMORY_BUDGET_OVERRIDE:-274877906944}"  # 256 GB

# Force JANGPress prestack regen (avoids stale cache from a different shape).
export JANGPRESS_PRESTACK="${JANGPRESS_PRESTACK:-1}"

# Keep large rebuildable overlays visible and grouped with local models
# instead of hiding 100+ GB files under macOS "System Data".
export JANGPRESS_PRESTACK_CACHE_DIR="${JANGPRESS_PRESTACK_CACHE_DIR:-$HOME/models/_runtime-cache/jangpress-prestack}"
mkdir -p "$JANGPRESS_PRESTACK_CACHE_DIR"

# Kimi's first prompt touches a very large MoE graph. The generic
# JANGPress route telemetry path synchronously reads router top-k indices
# back to CPU (`MLXArray.asArray`) during every MoE forward; on Kimi this
# has produced 100+ GB physical footprint before the first token. Keep it
# off by default for bring-up and benchmarks; set KIMI_ROUTE_TELEMETRY=1
# for targeted diagnostics.
export VMLX_JANGPRESS_ROUTE_TELEMETRY="${VMLX_JANGPRESS_ROUTE_TELEMETRY:-${KIMI_ROUTE_TELEMETRY:-0}}"

# Avoid inheriting ad-hoc router-advice env from prior experiments. Use
# KIMI_ROUTER_ADVICE=1 to turn it on explicitly for this wrapper.
export JANGPRESS_ROUTER_ADVICE="${KIMI_ROUTER_ADVICE:-0}"

# Experimental diagnostic: force layer-bounded prefill inside the Swift
# DeepseekV3/Kimi JANGTQ model. It did not solve Kimi Small on the 128 GB
# host (same 130 GB footprint), so keep it opt-in.
export VMLX_JANGTQ_LAYER_EVAL="${VMLX_JANGTQ_LAYER_EVAL:-${KIMI_LAYER_EVAL:-0}}"
export VMLX_JANGTQ_LAYER_RECLAIM="${VMLX_JANGTQ_LAYER_RECLAIM:-${KIMI_LAYER_RECLAIM:-1}}"

# Kimi K2.6 JANGTQ needs the Python-runtime prefill contract: 16-token
# chunks, materialization/synchronization between chunks, and a cold JIT
# warmup before the first real request. The Swift model defaults match this;
# keep wrapper knobs here so benchmark runs record the active runtime mode.
export VMLX_JANGTQ_PREFILL_STEP="${VMLX_JANGTQ_PREFILL_STEP:-${KIMI_PREFILL_STEP:-16}}"
export VMLX_JANGTQ_PREFILL_SYNC="${VMLX_JANGTQ_PREFILL_SYNC:-${KIMI_PREFILL_SYNC:-1}}"
export VMLX_JANGTQ_WARMUP="${VMLX_JANGTQ_WARMUP:-${KIMI_WARMUP:-1}}"

LOG="${LOG:-/tmp/kimi_serve_$(basename "$BUNDLE").log}"
echo "[kimi-serve] vmlxctl serve  port=$PORT  pct=$PCT  log=$LOG"
echo "[kimi-serve] model=$SHADOW"
echo ""

EXTRA_ARGS=()
FORCE_MODE="${KIMI_JANGPRESS_FORCE_MODE:-soft}"
if [[ "${KIMI_LOW_RAM:-1}" == "1" ]]; then
    # MMLU is 200 independent short prompts, so prefix/memory/disk KV
    # caches buy little and can retain pages across questions. Keep the
    # first bring-up path lean on 128 GB machines.
    EXTRA_ARGS+=(
        --disable-prefix-cache
        --disable-memory-cache
        --disable-disk-cache
        --kv-cache-quantization none
        --disable-idle
        --default-enable-thinking false
    )
fi
if [[ "${KIMI_ROUTER_ADVICE:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--enable-jangpress-router-advice)
fi

exec "$VMLXCTL" serve \
    --model "$SHADOW" \
    --enable-jangpress \
    --jang-press-compress-pct "$PCT" \
    --jang-press-backend mmap \
    --jang-press-force-mode "$FORCE_MODE" \
    --port "$PORT" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG"
