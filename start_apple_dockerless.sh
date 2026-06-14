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

# ── llama-server (cache → brew → release download → source build) ───────────────

BREW_PREFIX="$(brew --prefix)"
LLAMA_LOCAL="$SCRIPT_DIR/.llama_build/build/bin/llama-server"    # source-build cache
LLAMA_RELEASE="$SCRIPT_DIR/.llama_build/release/llama-server"    # downloaded release cache

llama_supports_granite_speech() {
  local bin="$1"
  # grep -q exits as soon as it finds a match, causing strings to get SIGPIPE.
  # With set -o pipefail that 141 exit propagates as failure even when the string
  # was found.  Run each check in a subshell with pipefail disabled.
  _gs_grep() { (set +o pipefail; strings "$1" 2>/dev/null | grep -q "granite_speech"); }

  # Check the binary itself.
  if _gs_grep "$bin"; then
    return 0
  fi
  # Check dylibs co-located with the binary (downloaded release layout).
  if (set +o pipefail; strings "$(dirname "$bin")"/libmtmd*.dylib 2>/dev/null | grep -q "granite_speech"); then
    return 0
  fi
  # The source build links granite_speech into dylibs (libmtmd) loaded via @rpath.
  # Extract the first LC_RPATH entry and check libmtmd there.
  local rpath
  rpath=$(otool -l "$bin" 2>/dev/null \
    | awk '/LC_RPATH/{f=1} f && /path /{print $2; f=0}' \
    | head -1)
  if [[ -n "$rpath" ]] && (set +o pipefail; strings "$rpath"/libmtmd*.dylib 2>/dev/null | grep -q "granite_speech"); then
    return 0
  fi
  return 1
}

if [[ -x "$LLAMA_RELEASE" ]] && llama_supports_granite_speech "$LLAMA_RELEASE"; then
  info "Using downloaded llama-server (granite_speech support confirmed)"
  LLAMA_SERVER="$LLAMA_RELEASE"
elif [[ -x "$LLAMA_LOCAL" ]] && llama_supports_granite_speech "$LLAMA_LOCAL"; then
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
    # Try downloading the pre-built release binary (saves ~10 min vs. source build).
    RELEASE_URL="https://github.com/angrave/granite-speech-4.1-serve/releases/latest/download/llama-server-macos-arm64.tar.gz"
    TMP_TAR=$(mktemp /tmp/llama-release-XXXXXX.tar.gz)
    info "Downloading pre-built llama-server from GitHub Releases..."
    if curl -fsSL --max-time 120 -o "$TMP_TAR" "$RELEASE_URL" 2>/dev/null; then
      mkdir -p "$(dirname "$LLAMA_RELEASE")"
      tar -xzf "$TMP_TAR" --strip-components=1 -C "$(dirname "$LLAMA_RELEASE")"
      rm -f "$TMP_TAR"
      chmod +x "$LLAMA_RELEASE"
      if llama_supports_granite_speech "$LLAMA_RELEASE"; then
        info "Downloaded llama-server supports granite_speech"
        LLAMA_SERVER="$LLAMA_RELEASE"
      else
        echo "→ Downloaded binary lacks granite_speech support — building from source"
        rm -rf "$(dirname "$LLAMA_RELEASE")"
      fi
    else
      rm -f "$TMP_TAR"
      echo "→ Release download failed — building from source"
    fi

    if [[ -z "${LLAMA_SERVER:-}" ]]; then
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
echo "  :19797 granite-base-llama  (llama.cpp + Metal, internal)"
echo "  :9797  granite-base        (chunking proxy)"
echo "  :18001 granite-plus        (PyTorch + MPS, internal)"
echo "  :8001  granite-plus-proxy  (chunking proxy with timestamp/speaker stitching)"
echo "  :8002  granite-nar         (PyTorch + MPS)"
echo ""
echo "Note: models are downloaded from HuggingFace on first run (several GB each)."
echo "Press Ctrl+C to stop all servers."
echo ""

# Ensure dylibs co-located with the binary are found (needed for downloaded release builds).
export DYLD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER")${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

"$LLAMA_SERVER" \
  -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 \
  --port 19797 --host 127.0.0.1 \
  --api-key "$LLAMA_API_KEY" \
  >> "$SCRIPT_DIR/base.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-base-llama"); LOGS+=("base.log")

echo "Waiting for llama-server on :19797..."
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:19797/health > /dev/null 2>&1 && break
  sleep 2
done

uvicorn serve_base:app --port 9797 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/base.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-base"); LOGS+=("base.log")

uvicorn serve_plus:app --port 18001 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/plus.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-plus"); LOGS+=("plus.log")

echo "Waiting for granite-plus model on :18001..."
for _ in $(seq 1 90); do
  curl -sf http://127.0.0.1:18001/health > /dev/null 2>&1 && break
  sleep 2
done

uvicorn serve_plus_proxy:app --port 8001 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/plus.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-plus-proxy"); LOGS+=("plus.log")

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
