#!/usr/bin/env bash
# install.sh — register granite-speech as a Linux systemd system service.
# Starts the docker compose stack at boot; containers restart themselves via
# their own restart: unless-stopped policy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
UNIT_NAME="granite-speech"
UNIT_SRC="$SCRIPT_DIR/${UNIT_NAME}.service"
UNIT_FILE="/etc/systemd/system/${UNIT_NAME}.service"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

usage() {
  cat >&2 <<EOF
Usage: sudo $0 --mode <ghcr|local> [--log <journald|file>]

  --mode ghcr    Use pre-built ghcr.io images (recommended — no local build step)
  --mode local   Build images locally from source (Dockerfile)

  --log journald Write logs to the systemd journal (default)
                   View with: journalctl -u $UNIT_NAME -f
  --log file     Write logs to $SCRIPT_DIR/service.log
EOF
  exit 1
}

MODE=""
LOG_MODE="journald"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="${2:-}"; shift 2 ;;
    --log)  LOG_MODE="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -z "$MODE" ]] && { echo "ERROR: --mode is required." >&2; echo ""; usage; }
[[ "$MODE" == "ghcr" || "$MODE" == "local" ]] \
  || die "--mode must be 'ghcr' or 'local'"
[[ "$LOG_MODE" == "journald" || "$LOG_MODE" == "file" ]] \
  || die "--log must be 'journald' or 'file'"
[[ "$(uname -s)" == "Linux" ]] || die "Linux only."
[[ $EUID -eq 0 ]] \
  || die "Must be run as root.  Try: sudo bash $0 --mode $MODE${LOG_MODE:+ --log $LOG_MODE}"

# ── Resolve identities ────────────────────────────────────────────────────────

# Run the service as the user who called sudo, not as root.
RUN_USER="${SUDO_USER:-root}"
HOME_DIR="$(getent passwd "$RUN_USER" | cut -d: -f6)"

# ── Resolve start script ──────────────────────────────────────────────────────

if [[ "$MODE" == "ghcr" ]]; then
  START_SCRIPT="scripts/start_ghcr.sh"
else
  START_SCRIPT="scripts/start_local_docker.sh"
fi

# ── Pre-flight checks ─────────────────────────────────────────────────────────

[[ -f "$INSTALL_DIR/$START_SCRIPT" ]] \
  || die "$START_SCRIPT not found at $INSTALL_DIR"
[[ -f "$INSTALL_DIR/.env" ]] \
  || die ".env not found at $INSTALL_DIR — copy .env.example and fill in API keys before installing."
command -v docker >/dev/null 2>&1 \
  || die "docker not found — install Docker before continuing."

info "Install directory : $INSTALL_DIR"
info "Start script      : $START_SCRIPT"
info "Run as user       : $RUN_USER"
info "Log mode          : $LOG_MODE"
info "Unit file         : $UNIT_FILE"

# ── Ensure RUN_USER is in the docker group ────────────────────────────────────

if [[ "$RUN_USER" != "root" ]] && ! id -nG "$RUN_USER" | grep -qw docker; then
  info "Adding $RUN_USER to the docker group..."
  usermod -aG docker "$RUN_USER"
  info "NOTE: $RUN_USER must log out and back in for the docker group to take effect."
  info "      The service will use the group immediately since it is launched by root."
fi

# ── Expand placeholders into the unit file ────────────────────────────────────

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

sed \
  -e "s|INSTALL_DIR|$INSTALL_DIR|g" \
  -e "s|START_SCRIPT|$START_SCRIPT|g" \
  -e "s|RUN_USER|$RUN_USER|g" \
  -e "s|HOME_DIR|$HOME_DIR|g" \
  "$UNIT_SRC" > "$TMP"

if [[ "$LOG_MODE" == "file" ]]; then
  LOG_FILE="$SCRIPT_DIR/service.log"
  touch "$LOG_FILE"
  chown "$RUN_USER" "$LOG_FILE"
  sed -i \
    "s|LOGGING_CONFIG_PLACEHOLDER|StandardOutput=append:$LOG_FILE\nStandardError=append:$LOG_FILE|" \
    "$TMP"
else
  sed -i '/LOGGING_CONFIG_PLACEHOLDER/d' "$TMP"
fi

cp "$TMP" "$UNIT_FILE"
info "Unit file written to $UNIT_FILE"

# ── Enable and start ──────────────────────────────────────────────────────────

systemctl daemon-reload

if systemctl is-active --quiet "$UNIT_NAME" 2>/dev/null; then
  info "Service already running — restarting..."
  systemctl restart "$UNIT_NAME"
else
  systemctl enable --now "${UNIT_NAME}.service"
fi

info "Service installed and started."
echo ""
echo "Useful commands:"
echo "  Status    : systemctl status $UNIT_NAME"
echo "  Logs      : journalctl -u $UNIT_NAME -f"
if [[ "$LOG_MODE" == "file" ]]; then
  echo "  File logs : tail -f $SCRIPT_DIR/service.log"
fi
echo "  Stop      : sudo $SCRIPT_DIR/stop.sh"
echo "  Uninstall : sudo $SCRIPT_DIR/uninstall.sh"
