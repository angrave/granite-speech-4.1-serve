#!/usr/bin/env bash
# start.sh — start or restart the service without reinstalling.
# Requires install.sh to have been run at least once.
set -euo pipefail

UNIT="granite-speech"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: Linux only." >&2; exit 1; }
[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"

[[ -f "/etc/systemd/system/${UNIT}.service" ]] \
  || { echo "ERROR: unit file not found — run install.sh first." >&2; exit 1; }

if systemctl is-active --quiet "$UNIT" 2>/dev/null; then
  echo "→ Restarting service..."
  systemctl restart "$UNIT"
else
  systemctl start "$UNIT"
  echo "→ Service started."
fi

echo "   Status : systemctl status $UNIT"
echo "   Logs   : journalctl -u $UNIT -f"
