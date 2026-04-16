#!/bin/bash
# End-to-end JANGTQ pipeline for the Qwen 3.6 artifact.
# Runs: verify structure → build Swift sidecar → Python decode smoke
#       → Swift decode smoke → report
#
# Usage: bash /tmp/qwen36_pipeline.sh [MODEL_DIR]
#   default MODEL_DIR: /Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L
set -e

MODEL=${1:-/Users/eric/models/Qwen3.6-35B-A3B-JANGTQ_2L}
echo "=== Pipeline target: $MODEL ==="

if [ ! -d "$MODEL" ]; then
  echo "FATAL: $MODEL does not exist"
  exit 1
fi

# Step 1: structural verification
echo
echo "[1/5] Structural verification"
python3 /tmp/verify_qwen36_artifact.py "$MODEL" || {
  echo "FAIL: artifact verification failed"
  exit 1
}

# Step 2: build Swift sidecar (only if missing)
echo
if [ -f "$MODEL/jangtq_runtime.safetensors" ]; then
  echo "[2/5] Sidecar already present, skipping"
else
  echo "[2/5] Building Swift jangtq_runtime.safetensors sidecar"
  python3 -m jang_tools.build_jangtq_sidecar "$MODEL" || {
    echo "FAIL: sidecar build failed"
    exit 1
  }
fi

# Step 3: Python decode smoke
echo
echo "[3/5] Python decode smoke"
python3 /tmp/test_qwen36_python.py "$MODEL" || {
  echo "FAIL: Python smoke failed"
  exit 1
}

# Step 4: Swift decode smoke (chat REPL — auto-quit after one prompt via stdin pipe)
echo
echo "[4/5] Swift decode smoke (one prompt via vmlxctl chat REPL)"
VMLXCTL=/Users/eric/vmlx/swift/.build/arm64-apple-macosx/debug/vmlxctl
if [ ! -x "$VMLXCTL" ]; then
  echo "  vmlxctl not built — skipping"
else
  printf "What is 2+2?\n/quit\n" | timeout 600 "$VMLXCTL" chat -m "$MODEL" || true
fi

# Step 5: report
echo
echo "[5/5] Done — artifact $MODEL ready for use"
echo "  Python: from jang_tools.load_jangtq import load_jangtq_model; m, t = load_jangtq_model('$MODEL')"
echo "  Swift : $VMLXCTL chat -m $MODEL"
