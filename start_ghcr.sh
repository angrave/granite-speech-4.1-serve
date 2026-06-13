#!/bin/bash
SCRIPT=`dirname $BASH_SOURCE`
cd $SCRIPT

if nvidia-smi &>/dev/null || [ -e /dev/nvidia0 ]; then
  export GHCR_TAG=cuda
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.ghcr.yml -f docker-compose.gpu.yml"
  echo "NVIDIA GPU detected — using ghcr.io image :cuda + GPU passthrough"
else
  export GHCR_TAG=latest
  COMPOSE_FILES="-f docker-compose.yml -f docker-compose.ghcr.yml"
  echo "No NVIDIA GPU detected — using ghcr.io image :latest (CPU)"
fi

docker compose $COMPOSE_FILES pull
docker compose $COMPOSE_FILES up -d
