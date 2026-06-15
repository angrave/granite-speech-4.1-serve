#!/usr/bin/env bash
# install.sh — register granite-speech as a macOS LaunchAgent
# Runs automatically at login; restarts if it crashes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"   # repo root
LABEL="com.granite-speech.serve"
PLIST_SRC="$SCRIPT_DIR/$LABEL.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$LABEL.plist"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

# ── Pre-flight checks ────────────────────────────────────────────────────────

[[ "$(uname -s)" == "Darwin" ]] || die "macOS only."
[[ -f "$INSTALL_DIR/scripts/start_apple_dockerless.sh" ]] \
  || die "scripts/start_apple_dockerless.sh not found at $INSTALL_DIR"
[[ -f "$INSTALL_DIR/.env" ]] \
  || die ".env not found at $INSTALL_DIR — copy .env.example and fill in API keys before installing the service."

info "Install directory : $INSTALL_DIR"
info "Plist destination : $PLIST_DST"

# ── Expand placeholders ──────────────────────────────────────────────────────

mkdir -p "$HOME/Library/LaunchAgents"
sed \
  -e "s|INSTALL_DIR|$INSTALL_DIR|g" \
  -e "s|HOME_DIR|$HOME|g" \
  "$PLIST_SRC" > "$PLIST_DST"

info "Plist written to $PLIST_DST"

# ── Load (or reload) ─────────────────────────────────────────────────────────

if launchctl list "$LABEL" &>/dev/null 2>&1; then
  info "Service already loaded — reloading..."
  launchctl unload "$PLIST_DST" 2>/dev/null || true
fi

launchctl load "$PLIST_DST"
info "Service loaded. Servers will start now and on every login."
echo ""
echo "Useful commands:"
echo "  Status  : launchctl list $LABEL"
echo "  Logs    : tail -f $INSTALL_DIR/service/osx/service.log"
echo "  Stop    : $SCRIPT_DIR/stop.sh"
echo "  Uninstall: $SCRIPT_DIR/uninstall.sh"
