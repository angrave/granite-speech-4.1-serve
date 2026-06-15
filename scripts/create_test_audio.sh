#!/usr/bin/env bash
# Creates test10s.wav — a 10-second mono 16 kHz clip used by direct-model tests.
# Requires the llama-base container to be running (provides ffmpeg).
set -euo pipefail

SRC="${1:-test.mp3}"
DST="${2:-test10s.wav}"
CONTAINER="${FFMPEG_CONTAINER:-granite-speech-41-serve-llama-base-1}"

docker cp "$SRC" "$CONTAINER:/tmp/_src_audio"
docker exec "$CONTAINER" ffmpeg -y -i /tmp/_src_audio -t 10 -ar 16000 -ac 1 /tmp/_test10s.wav
docker cp "$CONTAINER:/tmp/_test10s.wav" "$DST"
echo "Created $DST"
