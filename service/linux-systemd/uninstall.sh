#!/usr/bin/env bash
# uninstall.sh — stop, disable, and remove the granite-speech systemd service.
# Does NOT remove .env, models, logs, or docker images.
set -euo pipefail

UNIT="granite-speech"
UNIT_FILE="/etc/systemd/system/${UNIT}.service"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

[[ "$(uname -s)" == "Linux" ]] || die "Linux only."
[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"

if systemctl is-active --quiet "$UNIT" 2>/dev/null; then
  systemctl stop "$UNIT" 2>/dev/null || true
  info "Service stopped."
else
  info "Service was not running."
fi

if systemctl is-enabled --quiet "$UNIT" 2>/dev/null; then
  systemctl disable "$UNIT" 2>/dev/null || true
  info "Service disabled."
fi

if [[ -f "$UNIT_FILE" ]]; then
  rm "$UNIT_FILE"
  systemctl daemon-reload
  info "Removed $UNIT_FILE"
else
  info "Unit file not found — nothing to remove."
fi

info "Done. Stack will no longer start on boot."
echo "(Your .env, models, service.log, and docker images are untouched.)"
