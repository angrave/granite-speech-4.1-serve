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

# ── llama-server (brew first; source build if it lacks granite_speech support) ──

BREW_PREFIX="$(brew --prefix)"
LLAMA_LOCAL="$SCRIPT_DIR/.llama_build/build/bin/llama-server"

llama_supports_granite_speech() {
  local bin="$1"
  # Check for the granite_speech projector type string in the binary.
  strings "$bin" 2>/dev/null | grep -q "granite_speech"
}

if [[ -x "$LLAMA_LOCAL" ]] && llama_supports_granite_speech "$LLAMA_LOCAL"; then
  info "Using locally-built llama-server (granite_speech support confirmed)"
  LLAMA_SERVER="$LLAMA_LOCAL"
else
  # Try Homebrew
  if ! command -v llama-server &>/dev/null; then
    info "Installing llama.cpp via Homebrew..."
    brew install llama.cpp
  fi

  BREW_BIN="$(command -v llama-server)"
  if llama_supports_granite_speech "$BREW_BIN"; then
    info "Homebrew llama-server supports granite_speech"
    LLAMA_SERVER="$BREW_BIN"
  else
    echo "→ Homebrew llama-server lacks granite_speech support — building from source"
    echo "  (this takes ~10 minutes on first run; result is cached in .llama_build/)"
    if ! command -v cmake &>/dev/null; then
      info "Installing cmake via Homebrew..."
      brew install cmake
    fi
    LLAMA_SRC="$SCRIPT_DIR/.llama_build/src"
    if [[ ! -d "$LLAMA_SRC/.git" ]]; then
      git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
    else
      info "Updating llama.cpp source..."
      git -C "$LLAMA_SRC" pull --ff-only
    fi
    cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
      -DGGML_METAL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=OFF \
      -DLLAMA_BUILD_SERVER=ON \
      > "$SCRIPT_DIR/llama_build.log" 2>&1
    cmake --build "$LLAMA_SRC/build" --target llama-server \
      -j"$(sysctl -n hw.logicalcpu)" \
      >> "$SCRIPT_DIR/llama_build.log" 2>&1
    # Copy to canonical cache location
    mkdir -p "$(dirname "$LLAMA_LOCAL")"
    cp "$LLAMA_SRC/build/bin/llama-server" "$LLAMA_LOCAL"
    info "llama-server built and cached at .llama_build/build/bin/llama-server"
    LLAMA_SERVER="$LLAMA_LOCAL"
  fi
fi

# ── Python 3.10+ ────────────────────────────────────────────────────────────────
# The NAR model's remote code uses Python 3.10+ union-type syntax (int | None).
# Prefer a versioned interpreter from Homebrew; lazy-install python@3.11 if needed.

find_python() {
  local dirs=("$BREW_PREFIX/bin" /usr/local/bin /usr/bin)
  for dir in "${dirs[@]}"; do
    for ver in 3.14 3.13 3.12 3.11 3.10; do
      local p="$dir/python$ver"
      if [[ -x "$p" ]] && "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        echo "$p"; return 0
      fi
    done
  done
  return 1
}

if ! PYTHON=$(find_python); then
  info "No Python 3.10+ found — installing python@3.11 via Homebrew..."
  brew install python@3.11
  PYTHON=$(find_python) || die "Python 3.10+ still not found after install. Please install manually: brew install python@3.11"
fi
info "Using $PYTHON ($(${PYTHON} --version))"

# ── Virtual environment ──────────────────────────────────────────────────────────

if [[ ! -d "$VENV" ]]; then
  info "Creating virtual environment at .venv ..."
  "$PYTHON" -m venv "$VENV"
elif ! "$VENV/bin/python3" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
  info "Existing .venv is Python <3.10 — recreating with $PYTHON ..."
  rm -rf "$VENV"
  "$PYTHON" -m venv "$VENV"
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

PIDS=()
NAMES=()
LOGS=()

cleanup() {
  echo ""
  echo "Stopping servers..."
  [[ ${#PIDS[@]} -gt 0 ]] && kill "${PIDS[@]}" 2>/dev/null || true
  [[ ${#PIDS[@]} -gt 0 ]] && wait "${PIDS[@]}" 2>/dev/null || true
  echo "Done."
}
trap cleanup INT TERM

echo ""
echo "Starting servers (logs: base.log, plus.log, nar.log — use 'tail -f *.log' to monitor)"
echo "  :9797  granite-base  (llama.cpp + Metal)"
echo "  :8001  granite-plus  (PyTorch + MPS)"
echo "  :8002  granite-nar   (PyTorch + MPS)"
echo ""
echo "Note: models are downloaded from HuggingFace on first run (several GB each)."
echo "Press Ctrl+C to stop all servers."
echo ""

"$LLAMA_SERVER" \
  -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 \
  --port 9797 --host 127.0.0.1 \
  --api-key "$LLAMA_API_KEY" \
  >> "$SCRIPT_DIR/base.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-base"); LOGS+=("base.log")

uvicorn serve_plus:app --port 8001 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/plus.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-plus"); LOGS+=("plus.log")

uvicorn serve_nar:app --port 8002 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/nar.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-nar"); LOGS+=("nar.log")

# Monitor: report if a server exits unexpectedly but keep the others running.
while true; do
  sleep 3
  alive=(); alive_names=(); alive_logs=()
  for i in "${!PIDS[@]}"; do
    if kill -0 "${PIDS[$i]}" 2>/dev/null; then
      alive+=("${PIDS[$i]}")
      alive_names+=("${NAMES[$i]}")
      alive_logs+=("${LOGS[$i]}")
    else
      echo "WARNING: ${NAMES[$i]} exited — check ${LOGS[$i]}"
    fi
  done
  PIDS=("${alive[@]+"${alive[@]}"}")
  NAMES=("${alive_names[@]+"${alive_names[@]}"}")
  LOGS=("${alive_logs[@]+"${alive_logs[@]}"}")
  [[ ${#PIDS[@]} -gt 0 ]] || { echo "All servers have exited."; break; }
done
