#!/bin/bash
# JANG Studio — deep-signs the bundled Python runtime + the outer .app.
# Only runs in CI (Phase 7) with APPLE_DEV_ID_APP set to a real Developer ID.
#
# Usage: codesign-runtime.sh path/to/JANGStudio.app
# Env:   APPLE_DEV_ID_APP="Developer ID Application: Jinho Jang (TEAMID)"
# Created by Jinho Jang (eric@jangq.ai)

set -euo pipefail

APP="${1:-build/Debug/JANGStudio.app}"
ID="${APPLE_DEV_ID_APP:?APPLE_DEV_ID_APP not set}"
ENTITLEMENTS="$(cd "$(dirname "$0")/.." && pwd)/JANGStudio/Resources/JANGStudio.entitlements"

if [ ! -d "$APP" ]; then
    echo "[sign] FAIL — app bundle not found: $APP" >&2
    exit 1
fi
if [ ! -f "$ENTITLEMENTS" ]; then
    echo "[sign] FAIL — entitlements not found: $ENTITLEMENTS" >&2
    exit 1
fi

echo "[sign] deep-signing inner dylibs + python binaries under Contents/Resources/python"
PYTHON_ROOT="$APP/Contents/Resources/python"
if [ -d "$PYTHON_ROOT" ]; then
    # Sign in leaf-first order so nested deps are signed before parents.
    find "$PYTHON_ROOT" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 \
        | xargs -0 -n1 codesign --force --options runtime --timestamp --sign "$ID"
    # Sign the interpreter binaries themselves
    for exe in "$PYTHON_ROOT/bin/python3" "$PYTHON_ROOT/bin/python3.11"; do
        [ -e "$exe" ] && codesign --force --options runtime --timestamp --sign "$ID" "$exe" || true
    done
else
    echo "[sign] WARN — no bundled python at $PYTHON_ROOT (continuing; app-only sign)"
fi

echo "[sign] signing outer .app with entitlements"
codesign --force --deep --options runtime --timestamp \
    --entitlements "$ENTITLEMENTS" --sign "$ID" "$APP"

echo "[sign] verifying signature"
codesign --verify --deep --strict --verbose=2 "$APP"

# Gatekeeper check — pre-notarization it will warn; that's expected.
spctl --assess --type execute --verbose "$APP" || true

echo "[sign] OK — $APP"
