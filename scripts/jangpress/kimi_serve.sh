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
