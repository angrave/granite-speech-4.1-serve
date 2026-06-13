#!/usr/bin/env bash
# Benchmark: wall-clock time for one transcription request per model.
# Usage: ./benchmark.sh [audio.wav] [N_RUNS]
# Defaults: test.wav, 3 runs
# Requires: source .env first

set -euo pipefail

AUDIO="${1:-test.wav}"
N="${2:-3}"

: "${LLAMA_API_KEY:?LLAMA_API_KEY not set. Run: source .env}"
: "${GRANITE_API_KEY:?GRANITE_API_KEY not set. Run: source .env}"

echo "Audio file : ${AUDIO}"
echo "Runs each  : ${N}"
echo "Date       : $(date)"
echo ""

audio_duration_s() {
  python3 -c "
import wave
with wave.open('${AUDIO}', 'r') as f:
    print(f.getnframes() / f.getframerate())
" 2>/dev/null || echo "unknown"
}
AUDIO_DUR=$(audio_duration_s)
echo "Audio duration: ${AUDIO_DUR}s"
echo ""

run_benchmark() {
  local label="$1"
  local url="$2"
  local auth_header="$3"
  local model_field="$4"
  local extra_fields="$5"

  echo "--- ${label} ---"
  local total=0
  for i in $(seq 1 "${N}"); do
    start=$(python3 -c "import time; print(time.monotonic())")
    curl -s "${url}" \
      -H "${auth_header}" \
      -F "file=@${AUDIO}" \
      -F "model=${model_field}" \
      ${extra_fields} \
      -o /dev/null
    end=$(python3 -c "import time; print(time.monotonic())")
    elapsed=$(python3 -c "print(f'{${end} - ${start}:.2f}')")
    echo "  Run ${i}: ${elapsed}s"
    total=$(python3 -c "print(${total} + ${end} - ${start})")
  done
  avg=$(python3 -c "print(f'{${total} / ${N}:.2f}')")
  if [ "${AUDIO_DUR}" != "unknown" ]; then
    rtfx=$(python3 -c "print(f'{float(\"${AUDIO_DUR}\") / (${total} / ${N}):.1f}')")
    echo "  Average: ${avg}s  |  RTFx (real-time factor): ${rtfx}x"
  else
    echo "  Average: ${avg}s"
  fi
  echo ""
}

run_benchmark \
  "Base model (llama.cpp, port 9797)" \
  "http://127.0.0.1:9797/v1/audio/transcriptions" \
  "Authorization: Bearer ${LLAMA_API_KEY}" \
  "ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  "-F 'prompt=transcribe with punctuation and capitalization.'"

run_benchmark \
  "Plus model (FastAPI, port 8001)" \
  "http://127.0.0.1:8001/v1/audio/transcriptions" \
  "Authorization: Bearer ${GRANITE_API_KEY}" \
  "plus" \
  ""

run_benchmark \
  "NAR model (FastAPI, port 8002)" \
  "http://127.0.0.1:8002/v1/audio/transcriptions" \
  "Authorization: Bearer ${GRANITE_API_KEY}" \
  "nar" \
  ""

echo "Benchmark complete."
