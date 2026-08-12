#!/bin/bash
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT/.."

if [ ! -f .env ]; then
  echo "ERROR: .env not found." >&2
  echo "       Copy .env.example to .env and fill in your keys." >&2
  echo "       See README.md for details." >&2
  exit 1
fi

pick_cuda_wheels() {
  local ver major minor
  ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
  [ -z "$ver" ] && { echo "cpu 2.6.0"; return; }
  major=${ver%%.*}; minor=${ver##*.}
  if   [ "$major" -ge 13 ];                              then echo "cu130 2.11.0"
  elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ];        then echo "cu128 2.11.0"
  # cu124 + torch 2.6.0 (the previous pairing here) hard-pins
  # nvidia-cudnn-cu12==9.1.0.70, and that exact build is absent from
  # PyTorch's cu124 wheel index (it's still on PyPI proper, just never
  # mirrored/kept on pytorch.org's cu124 channel), so
  # `pip install torch==2.6.0 --index-url .../cu124` now fails with
  # "No matching distribution found for nvidia-cudnn-cu12==9.1.0.70".
  # cu126 + torch 2.7.1 is the last PyTorch release that still ships
  # Pascal (sm_60/61) kernels — matching this tier's "Pascal -> Ada"
  # promise below — and its cudnn pin (9.5.1.17) resolves cleanly as of
  # 2026-08. If this breaks again, check whether the pinned
  # nvidia-cudnn-cu12 version for the chosen torch release is still listed
  # at https://download.pytorch.org/whl/<tag>/nvidia-cudnn-cu12/ before
  # bumping further.
  else                                                         echo "cu126 2.7.1"
  fi
}

if nvidia-smi &>/dev/null || [ -e /dev/nvidia0 ]; then
  read -r cu_tag pt_ver <<< "$(pick_cuda_wheels)"
  export PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${cu_tag}"
  export PYTORCH_VERSION="${pt_ver}"
  echo "NVIDIA GPU detected (CUDA ${cu_tag#cu}) — building with ${cu_tag} wheels (PyTorch ${pt_ver}) + GPU passthrough"
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
else
  export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
  export PYTORCH_VERSION=2.6.0
  echo "No NVIDIA GPU detected — building with CPU wheels (PyTorch ${PYTORCH_VERSION})"
  COMPOSE_FILES="-f docker-compose.yml"
fi

if [ -f docker-compose.local.yml ]; then
  COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.local.yml"
  echo "Applying local overrides from docker-compose.local.yml"
fi

docker compose $COMPOSE_FILES pull
docker compose $COMPOSE_FILES up -d --build
