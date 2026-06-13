#!/usr/bin/env bash
set -euo pipefail

AUDIO="${1:-test.wav}"

# Auto-source .env if keys aren't already in the environment
if [[ -z "${LLAMA_API_KEY:-}" || -z "${GRANITE_API_KEY:-}" ]] && [[ -f .env ]]; then
  # shellcheck source=.env
  # shellcheck disable=SC1091
  source .env
fi

# Keys must be in environment
: "${LLAMA_API_KEY:?LLAMA_API_KEY not set. Run: source .env}"
: "${GRANITE_API_KEY:?GRANITE_API_KEY not set. Run: source .env}"

echo "=== Pre-flight: server health (no auth required) ==="
for port in 9797 8001 8002; do
  result=$(curl -sf "http://127.0.0.1:${port}/health" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'), '| auth:', d.get('auth','n/a'))" \
    2>/dev/null || echo "UNREACHABLE")
  echo "  Port ${port}: ${result}"
done

echo ""
echo "=== Auth rejection check (expect 401 from all three) ==="
for port in 9797 8001 8002; do
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -F "file=@${AUDIO}" \
    -F "model=test" \
    "http://127.0.0.1:${port}/v1/audio/transcriptions" 2>/dev/null || echo "000")
  echo "  Port ${port} without key: HTTP ${code}"
done

echo ""
echo "=== Base (llama.cpp, port 9797) ==="
curl -s http://127.0.0.1:9797/v1/audio/transcriptions \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -F "model=ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  -F "file=@${AUDIO}" \
  -F "prompt=transcribe with punctuation and capitalization." \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== Plus — plain ASR (port 8001) ==="
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== Plus — word-level timestamps (port 8001) ==="
# Timestamps use format [T:N] where N is end-time in centiseconds mod 1000.
# Note: use --form-string (not -F) for prompts that start with <|audio|> — curl's
# -F flag treats values starting with < as file-content references.
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== Plus — speaker attribution (port 8001) ==="
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns." \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== Plus — punctuated ASR (port 8001) ==="
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> transcribe the speech with proper punctuation and capitalization." \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== Plus — keyword biasing (port 8001) ==="
# Add keywords to bias recognition towards domain-specific terms.
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> can you transcribe the speech into a written format? Keywords: timothy, velvet, hearth" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== Plus — combined (punct + timestamps + speakers) on single-speaker (port 8001) ==="
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Timestamps and Speaker attribution: Transcribe the speech with proper punctuation and capitalization. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]. Denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns." \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== NAR (FastAPI, port 8002) ==="
curl -s http://127.0.0.1:8002/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=nar" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"

echo ""
echo "=== Plus — speaker attribution on multi-speaker audio (port 8001) ==="
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@test_multi_speaker.wav" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns." \
  | python3 -c "
import sys, json
text = json.load(sys.stdin)['text']
print(text)
assert '[Speaker 1]:' in text and '[Speaker 2]:' in text, \
    'FAIL: expected both [Speaker 1]: and [Speaker 2]: tags'
print('PASS: speaker split detected')
"

echo ""
echo "=== Plus — combined (punct + timestamps + speakers) on multi-speaker audio (port 8001) ==="
# Tests whether the model can produce punctuation, word-level timestamps, and
# speaker tags all from a single prompt.
curl -s http://127.0.0.1:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@test_multi_speaker.wav" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Timestamps and Speaker attribution: Transcribe the speech with proper punctuation and capitalization. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]. Denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns." \
  | python3 -c "
import sys, json
text = json.load(sys.stdin)['text']
print(text)
has_speakers = '[Speaker 1]:' in text and '[Speaker 2]:' in text
has_timestamps = '[T:' in text
print('PASS' if has_speakers else 'WARN: speaker tags missing',
      '| speakers:', has_speakers, '| timestamps:', has_timestamps)
"

echo ""
echo "=== Done ==="
