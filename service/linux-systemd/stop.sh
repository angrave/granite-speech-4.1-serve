#!/usr/bin/env bash
# stop.sh — stop the running service without uninstalling it.
# Containers are brought down cleanly via ExecStop in the unit file.
# The service (and containers) will restart on next boot, or use start.sh now.
set -euo pipefail

UNIT="granite-speech"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[[ "$(uname -s)" == "Linux" ]] || { echo "ERROR: Linux only." >&2; exit 1; }
[[ $EUID -eq 0 ]] || exec sudo "$0" "$@"

if systemctl is-active --quiet "$UNIT" 2>/dev/null; then
  systemctl stop "$UNIT"
  echo "→ Service stopped. Containers have been brought down."
  echo "   Run $SCRIPT_DIR/start.sh to restart."
else
  echo "→ Service is not currently running."
fi
