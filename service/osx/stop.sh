#!/usr/bin/env bash
# stop.sh — stop the running service without uninstalling it.
# It will restart automatically on next login (or use start.sh to restart now).
set -euo pipefail

LABEL="com.granite-speech.serve"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

[[ "$(uname -s)" == "Darwin" ]] || { echo "ERROR: macOS only." >&2; exit 1; }

if launchctl list "$LABEL" &>/dev/null 2>&1; then
  launchctl unload "$PLIST_DST"
  echo "→ Service stopped. Run start.sh to restart."
else
  echo "→ Service is not currently running."
fi
