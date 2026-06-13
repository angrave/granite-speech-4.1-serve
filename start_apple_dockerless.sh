#!/usr/bin/env bash
# start-services.sh — Native Apple Silicon launcher for granite-speech-4.1
# Lazy-installs all dependencies and starts all three servers with MPS/Metal acceleration.
# Server logs: base.log, plus.log, nar.log  (tail -f *.log to monitor)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "→ $*"; }

# ── Platform ────────────────────────────────────────────────────────────────────

[[ "$(uname -s)" == "Darwin" ]] || die "This script is for macOS only."
[[ "$(uname -m)" == "arm64"  ]] || die "Apple Silicon (arm64) required for MPS acceleration."

# ── Homebrew ────────────────────────────────────────────────────────────────────

command -v brew &>/dev/null \
  || die "Homebrew not found. Install from https://brew.sh then re-run."

# ── llama.cpp ───────────────────────────────────────────────────────────────────

if ! command -v llama-server &>/dev/null; then
  info "Installing llama.cpp via Homebrew..."
  brew install llama.cpp
fi

# ── Python 3.9+ ─────────────────────────────────────────────────────────────────

command -v python3 &>/dev/null \
  || die "python3 not found. Install Python 3.9+ (e.g. brew install python) and re-run."

python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)' \
  || die "Python 3.9+ required (found $(python3 --version))."

# ── Virtual environment ──────────────────────────────────────────────────────────

if [[ ! -d "$VENV" ]]; then
  info "Creating virtual environment at .venv ..."
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

# ── PyTorch — arm64 wheel includes MPS ──────────────────────────────────────────

if ! python3 -c 'import torch' &>/dev/null; then
  info "Installing torch and torchaudio (arm64 + MPS)..."
  pip install --quiet torch torchaudio
fi

if ! python3 -c 'import torch; assert torch.backends.mps.is_available()' &>/dev/null; then
  echo "WARNING: MPS not available — inference will fall back to CPU." >&2
fi

# ── Python dependencies ──────────────────────────────────────────────────────────
# Re-install only when requirements.txt has changed (tracked by md5 hash).

HASH_FILE="$VENV/.reqs_hash"
CURRENT_HASH=$(md5 -q "$SCRIPT_DIR/requirements.txt")
if [[ ! -f "$HASH_FILE" ]] || [[ "$(<"$HASH_FILE")" != "$CURRENT_HASH" ]]; then
  info "Installing Python requirements..."
  pip install --quiet -r "$SCRIPT_DIR/requirements.txt"
  echo "$CURRENT_HASH" > "$HASH_FILE"
fi

# ── Environment variables ────────────────────────────────────────────────────────

if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
  set +a
fi

for var in GRANITE_API_KEY LLAMA_API_KEY; do
  val="${!var:-}"
  [[ -n "$val" ]] \
    || die "$var is not set. Copy .env.example to .env and fill in your keys."
  [[ "$val" != "change-me-"* ]] \
    || die "$var still has its placeholder value. Set a real key in .env."
done

# ── Launch ───────────────────────────────────────────────────────────────────────

cd "$SCRIPT_DIR"

cleanup() {
  echo ""
  echo "Stopping servers..."
  # shellcheck disable=SC2317
  kill "${PIDS[@]}" 2>/dev/null || true
  wait "${PIDS[@]}" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT INT TERM

PIDS=()

echo ""
echo "Starting servers (logs: base.log, plus.log, nar.log — use 'tail -f *.log' to monitor)"
echo "  :9797  granite-base  (llama.cpp + Metal)"
echo "  :8001  granite-plus  (PyTorch + MPS)"
echo "  :8002  granite-nar   (PyTorch + MPS)"
echo ""
echo "Note: models are downloaded from HuggingFace on first run (several GB each)."
echo "Press Ctrl+C to stop all servers."
echo ""

llama-server \
  -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 \
  --port 9797 --host 127.0.0.1 \
  --api-key "$LLAMA_API_KEY" \
  >> "$SCRIPT_DIR/base.log" 2>&1 &
PIDS+=($!)

uvicorn serve_plus:app --port 8001 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/plus.log" 2>&1 &
PIDS+=($!)

uvicorn serve_nar:app --port 8002 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/nar.log" 2>&1 &
PIDS+=($!)

wait "${PIDS[@]}"
