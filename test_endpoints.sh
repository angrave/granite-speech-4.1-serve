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
for port in 8700 8701 8702; do
  result=$(curl -sf "http://127.0.0.1:${port}/health" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'), '| auth:', d.get('auth','n/a'))" \
    2>/dev/null || echo "UNREACHABLE")
  echo "  Port ${port}: ${result}"
done

echo ""
echo "=== Auth rejection check (expect 401 from all three) ==="
for port in 8700 8701 8702; do
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -F "file=@${AUDIO}" \
    -F "model=test" \
    "http://127.0.0.1:${port}/v1/audio/transcriptions" 2>/dev/null || echo "000")
  echo "  Port ${port} without key: HTTP ${code}"
done

echo ""
echo "=== Base (llama.cpp proxy, port 8700) ==="
raw=$(curl -s http://127.0.0.1:8700/v1/audio/transcriptions \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -F "model=ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  -F "file=@${AUDIO}" \
  -F "prompt=transcribe with punctuation and capitalization.")
echo "  raw: ${raw}"
python3 -c "
import sys, json
d = json.loads(sys.argv[1])
print('  chunks:', d['usage'].get('chunks', 1))
print('  text:  ', d['text'])
assert d['text'].strip(), 'FAIL: text is empty'
print('  PASS: non-empty')
" "${raw}"

echo ""
echo "=== Plus — plain ASR (port 8701) ==="
raw=$(curl -s http://127.0.0.1:8701/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus")
echo "  raw: ${raw}"
python3 -c "
import sys, json
raw = sys.argv[1]
d = json.loads(raw)
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
print('  PASS: non-empty')
" "${raw}"

echo ""
echo "=== Plus — word-level timestamps (port 8701) ==="
# Timestamps use format [T:N] where N is end-time in centiseconds mod 1000.
# Note: use --form-string (not -F) for prompts that start with <|audio|> — curl's
# -F flag treats values starting with < as file-content references.
raw=$(curl -s http://127.0.0.1:8701/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]")
echo "  raw: ${raw}"
python3 -c "
import sys, json
raw = sys.argv[1]
d = json.loads(raw)
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
print('  PASS: non-empty')
" "${raw}"

echo ""
echo "=== Plus — speaker attribution (port 8701) ==="
raw=$(curl -s http://127.0.0.1:8701/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.")
echo "  raw: ${raw}"
python3 -c "
import sys, json
raw = sys.argv[1]
d = json.loads(raw)
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
print('  PASS: non-empty')
" "${raw}"


echo ""
echo "=== Plus — keyword biasing (port 8701) ==="
# Add keywords to bias recognition towards domain-specific terms.
raw=$(curl -s http://127.0.0.1:8701/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> can you transcribe the speech into a written format? Keywords: timothy, velvet, hearth,uh-huhhh")
echo "  raw: ${raw}"
python3 -c "
import sys, json
raw = sys.argv[1]
d = json.loads(raw)
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
print('  PASS: non-empty')
" "${raw}"


echo ""
echo "=== NAR (FastAPI, port 8702) ==="
raw=$(curl -s http://127.0.0.1:8702/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=nar")
echo "  raw: ${raw}"
python3 -c "
import sys, json
raw = sys.argv[1]
d = json.loads(raw)
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
print('  PASS: non-empty')
" "${raw}"

echo ""
echo "=== Plus — speaker attribution on multi-speaker audio (port 8701) ==="
raw=$(curl -s http://127.0.0.1:8701/v1/audio/transcriptions \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.")
echo "  raw: ${raw}"
python3 -c "
import sys, json
raw = sys.argv[1]
d = json.loads(raw)
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
assert '[Speaker 1]:' in text and '[Speaker 2]:' in text, \
    'FAIL: expected both [Speaker 1]: and [Speaker 2]: tags'
print('  PASS: speaker split detected')
" "${raw}"

echo "=== Done ==="
