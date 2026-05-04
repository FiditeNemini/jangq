#!/usr/bin/env bash
# Run MMLU 200q (10 subjects × 20 questions) against a Kimi JANGTQ bundle
# served via JANGPress mmap. Assumes vmlxctl serve is already running on
# $PORT (default 8082) — start it with kimi_serve.sh in another terminal.
#
# Usage:
#   ./kimi_mmlu.sh <model-id> [thinking|chat] [port=8082]
#
# model-id is the slug vmlxctl uses for the loaded model (the bundle dir
# basename — e.g. `Kimi-K2.6-Med-JANGTQ`). The /v1/chat/completions
# endpoint accepts whatever model name was loaded.
#
# Output:
#   $OUT_DIR/<bundle>_<mode>_<ts>.log
#   (full prompt+answer JSONL also lives in that directory)
set -euo pipefail

MODEL_ID="${1:?model id required (bundle name vmlxctl loaded)}"
MODE="${2:-chat}"
PORT="${3:-8082}"

PY="${PY:-python3}"
JANG_ROOT="${JANG_ROOT:-$(pwd)}"
OUT_DIR="${OUT_DIR:-$JANG_ROOT/jangpress-mmlu-runs}"
mkdir -p "$OUT_DIR"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$OUT_DIR/${MODEL_ID}_${MODE}_${TS}.log"
JSONL="$OUT_DIR/${MODEL_ID}_${MODE}_${TS}.jsonl"

# Wait until the server is responsive (model load can take 60-90s).
echo "[mmlu] waiting for vmlxctl on :$PORT..." | tee -a "$LOG"
for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "[mmlu] server up" | tee -a "$LOG"
        break
    fi
    sleep 5
done
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo "FATAL: vmlxctl not responding on :$PORT after 5 min" | tee -a "$LOG"
    exit 1
fi

THINKING_FLAG=""
if [[ "$MODE" == "thinking" ]]; then
    THINKING_FLAG="--thinking"
fi

echo "[mmlu] running 200q against $MODEL_ID  (mode=$MODE, port=$PORT)" | tee -a "$LOG"
echo "[mmlu] log:   $LOG" | tee -a "$LOG"
echo "[mmlu] jsonl: $JSONL" | tee -a "$LOG"

cd "$JANG_ROOT"
export MMLU_API_BASE="${MMLU_API_BASE:-http://127.0.0.1:$PORT/v1}"
export MMLU_HTTP_TIMEOUT="${MMLU_HTTP_TIMEOUT:-600}"
export MMLU_PROCESS_MATCH="${MMLU_PROCESS_MATCH:-vmlxctl serve}"
MMLU_JSONL_OUT="$JSONL" \
    "$PY" benchmark_mmlu_api.py $THINKING_FLAG "$MODEL_ID" 2>&1 | tee -a "$LOG"

echo "[mmlu] done. summary in $LOG"
