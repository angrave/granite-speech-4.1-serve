#!/bin/bash
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT/.."

if nvidia-smi &>/dev/null || [ -e /dev/nvidia0 ]; then
  export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
  echo "NVIDIA GPU detected — building with CUDA 12.4 wheels + GPU passthrough"
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
else
  export PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
  echo "No NVIDIA GPU detected — building with CPU wheels"
  COMPOSE_FILES="-f docker-compose.yml"
fi

if [ -f docker-compose.local.yml ]; then
  COMPOSE_FILES="$COMPOSE_FILES -f docker-compose.local.yml"
  echo "Applying local overrides from docker-compose.local.yml"
fi

docker compose $COMPOSE_FILES pull
docker compose $COMPOSE_FILES up -d --build
