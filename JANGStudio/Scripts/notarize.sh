#!/bin/bash
# JANG Studio — notarizes a signed .app via Apple's notarytool, waits for approval,
# and staples the ticket.
#
# Usage: notarize.sh path/to/JANGStudio.app
# Env:   APPLE_ID              — Apple account email
#        APPLE_TEAM_ID         — 10-char team ID
#        APPLE_APP_PASSWORD    — app-specific password (NOT the account password)
#
# Call AFTER codesign-runtime.sh has signed the app. Only runs in CI.
# Created by Jinho Jang (eric@jangq.ai)

set -euo pipefail

APP="${1:-build/Debug/JANGStudio.app}"
AI="${APPLE_ID:?APPLE_ID not set}"
TID="${APPLE_TEAM_ID:?APPLE_TEAM_ID not set}"
PW="${APPLE_APP_PASSWORD:?APPLE_APP_PASSWORD not set}"
ZIP="$(dirname "$APP")/$(basename "$APP" .app).zip"

if [ ! -d "$APP" ]; then
    echo "[notarize] FAIL — app bundle not found: $APP" >&2
    exit 1
fi

echo "[notarize] zipping $(basename "$APP") → $(basename "$ZIP")"
rm -f "$ZIP"
(cd "$(dirname "$APP")" && ditto -c -k --keepParent "$(basename "$APP")" "$(basename "$ZIP")")

echo "[notarize] submitting to Apple (timeout 30 min)"
xcrun notarytool submit "$ZIP" \
    --apple-id "$AI" --team-id "$TID" --password "$PW" \
    --wait --timeout 30m

echo "[notarize] stapling ticket to .app"
xcrun stapler staple "$APP"

echo "[notarize] final Gatekeeper check"
spctl --assess --type execute --verbose "$APP"

rm -f "$ZIP"
echo "[notarize] OK — $APP"
