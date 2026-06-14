#!/usr/bin/env bash
# start.sh — start (or restart) the service without reinstalling.
# Requires install.sh to have been run at least once.
set -euo pipefail

LABEL="com.granite-speech.serve"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

[[ "$(uname -s)" == "Darwin" ]] || { echo "ERROR: macOS only." >&2; exit 1; }
[[ -f "$PLIST_DST" ]] || { echo "ERROR: plist not found — run install.sh first." >&2; exit 1; }

if launchctl list "$LABEL" &>/dev/null 2>&1; then
  echo "→ Restarting service..."
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
else
  launchctl load "$PLIST_DST"
  echo "→ Service started."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "   Logs: tail -f $SCRIPT_DIR/service.log"
echo "   Status: launchctl list $LABEL"
