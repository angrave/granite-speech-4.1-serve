#!/usr/bin/env bash
# uninstall.sh — remove the granite-speech LaunchAgent
set -euo pipefail

LABEL="com.granite-speech.serve"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

[[ "$(uname -s)" == "Darwin" ]] || die "macOS only."

if launchctl list "$LABEL" &>/dev/null 2>&1; then
  launchctl unload "$PLIST_DST" 2>/dev/null || true
  info "Service stopped and unloaded."
else
  info "Service was not loaded."
fi

if [[ -f "$PLIST_DST" ]]; then
  rm "$PLIST_DST"
  info "Removed $PLIST_DST"
fi

info "Done. Servers will no longer start at login."
echo "(Your .env, models, and logs are untouched.)"
