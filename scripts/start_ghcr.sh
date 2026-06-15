#!/bin/bash
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT/.."

if [ ! -f .env ]; then
  echo "ERROR: .env not found." >&2
  echo "       Copy .env.example to .env and fill in your keys." >&2
  echo "       See README.md for details." >&2
  exit 1
fi

pick_cuda_tag() {
  local ver major minor
  ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
  [ -z "$ver" ] && { echo "cpu"; return; }
  major=${ver%%.*}; minor=${ver##*.}
  if   [ "$major" -ge 13 ];                              then echo "cuda130"
  elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ];        then echo "cuda128"
  else                                                         echo "cuda"
  fi
}

if nvidia-smi &>/dev/null || [ -e /dev/nvidia0 ]; then
  export GHCR_TAG=$(pick_cuda_tag)
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.ghcr.yml -f docker-compose.gpu.yml"
  echo "NVIDIA GPU detected — using ghcr.io image :${GHCR_TAG} + GPU passthrough"
else
  export GHCR_TAG=latest
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.ghcr.yml"
  echo "No NVIDIA GPU detected — using ghcr.io image :latest (CPU)"
fi

if [ -f docker-compose.local.yml ]; then
  COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.local.yml"
  echo "Applying local overrides from docker-compose.local.yml"
fi

docker compose $COMPOSE_FILES pull
docker compose $COMPOSE_FILES up -d
