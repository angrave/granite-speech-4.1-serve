#!/usr/bin/env bash
set -euo pipefail

AUDIO="${1:-test.mp3}"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found." >&2
  echo "       Copy .env.example to .env and fill in your keys." >&2
  echo "       See README.md for details." >&2
  exit 1
fi

# Auto-source .env if keys aren't already in the environment
if [[ -z "${LLAMA_API_KEY:-}" || -z "${GRANITE_API_KEY:-}" ]]; then
  # shellcheck source=.env
  # shellcheck disable=SC1091
  source .env
fi

# Keys must be in environment
: "${LLAMA_API_KEY:?LLAMA_API_KEY not set. Run: source .env}"
: "${GRANITE_API_KEY:?GRANITE_API_KEY not set. Run: source .env}"

GRANITE_BASE_DIRECT_PORT="${GRANITE_BASE_DIRECT_PORT:-8700}"
GRANITE_PLUS_DIRECT_PORT="${GRANITE_PLUS_DIRECT_PORT:-8701}"
GRANITE_NAR_DIRECT_PORT="${GRANITE_NAR_DIRECT_PORT:-8702}"
GRANITE_BASE_PROXY_PORT="${GRANITE_BASE_PROXY_PORT:-18700}"
GRANITE_PLUS_PROXY_PORT="${GRANITE_PLUS_PROXY_PORT:-18701}"
AUDIO_SHORT="${AUDIO_SHORT:-test10s.wav}"

: "${AUDIO_SHORT:?}"
[[ -f "$AUDIO_SHORT" ]] || { echo "ERROR: $AUDIO_SHORT not found. Run: scripts/create_test_audio.sh" >&2; exit 1; }

# Returns 0 if the named profile is active (or COMPOSE_PROFILES is unset → all active).
profile_active() {
  local profiles="${COMPOSE_PROFILES:-base,plus,nar}"
  [[ ",$profiles," == *",$1,"* ]]
}

echo "=== Pre-flight: server health (no auth required) ==="
for port in "${GRANITE_BASE_DIRECT_PORT}" "${GRANITE_PLUS_DIRECT_PORT}" "${GRANITE_NAR_DIRECT_PORT}"; do
  case $port in
    "${GRANITE_BASE_DIRECT_PORT}") profile="base" ;;
    "${GRANITE_PLUS_DIRECT_PORT}") profile="plus" ;;
    "${GRANITE_NAR_DIRECT_PORT}")  profile="nar"  ;;
  esac
  if ! profile_active "$profile"; then
    echo "  Port ${port}: SKIPPED (profile '$profile' not in COMPOSE_PROFILES)"
    continue
  fi
  result=$(curl -sf "http://127.0.0.1:${port}/health" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'), '| auth:', d.get('auth','n/a'))" \
    2>/dev/null || echo "UNREACHABLE")
  echo "  Port ${port}: ${result}"
done

echo ""
echo "=== Auth rejection check (expect 401 from active endpoints) ==="
for port in "${GRANITE_BASE_DIRECT_PORT}" "${GRANITE_PLUS_DIRECT_PORT}" "${GRANITE_NAR_DIRECT_PORT}"; do
  case $port in
    "${GRANITE_BASE_DIRECT_PORT}") profile="base" ;;
    "${GRANITE_PLUS_DIRECT_PORT}") profile="plus" ;;
    "${GRANITE_NAR_DIRECT_PORT}")  profile="nar"  ;;
  esac
  if ! profile_active "$profile"; then
    echo "  Port ${port} without key: SKIPPED"
    continue
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" \
    -F "file=@${AUDIO}" \
    -F "model=test" \
    "http://127.0.0.1:${port}/v1/audio/transcriptions" 2>/dev/null || echo "000")
  echo "  Port ${port} without key: HTTP ${code}"
done

if profile_active base; then
echo ""
echo "=== Base (llama.cpp proxy, port ${GRANITE_BASE_DIRECT_PORT}) ==="
raw=$(curl -s "http://127.0.0.1:${GRANITE_BASE_DIRECT_PORT}/v1/audio/transcriptions" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -F "model=ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  -F "file=@${AUDIO}" \
  --form-string "prompt=<|audio|>transcribe the speech with proper punctuation and capitalization.")
echo "  raw: ${raw}"
python3 -c "
import sys, json
d = json.loads(sys.argv[1])
print('  chunks:', d['usage'].get('chunks', 1))
print('  text:  ', d['text'])
assert d['text'].strip(), 'FAIL: text is empty'
print('  PASS: non-empty')
" "${raw}"
fi # profile: base

if profile_active plus; then
echo ""
echo "=== Plus — plain ASR (port ${GRANITE_PLUS_DIRECT_PORT}) ==="
raw=$(curl -s "http://127.0.0.1:${GRANITE_PLUS_DIRECT_PORT}/v1/audio/transcriptions" \
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
echo "=== Plus — word-level timestamps (port ${GRANITE_PLUS_DIRECT_PORT}) ==="
# Timestamps use format [T:N] where N is end-time in centiseconds mod 1000.
# Note: use --form-string (not -F) for prompts that start with <|audio|> — curl's
# -F flag treats values starting with < as file-content references.
raw=$(curl -s "http://127.0.0.1:${GRANITE_PLUS_DIRECT_PORT}/v1/audio/transcriptions" \
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
echo "=== Plus — speaker attribution (port ${GRANITE_PLUS_DIRECT_PORT}) ==="
raw=$(curl -s "http://127.0.0.1:${GRANITE_PLUS_DIRECT_PORT}/v1/audio/transcriptions" \
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
echo "=== Plus — keyword biasing (port ${GRANITE_PLUS_DIRECT_PORT}) ==="
# Add keywords to bias recognition towards domain-specific terms.
raw=$(curl -s "http://127.0.0.1:${GRANITE_PLUS_DIRECT_PORT}/v1/audio/transcriptions" \
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
fi # profile: plus


if profile_active nar; then
echo ""
echo "=== NAR (FastAPI, port ${GRANITE_NAR_DIRECT_PORT}) ==="
raw=$(curl -s "http://127.0.0.1:${GRANITE_NAR_DIRECT_PORT}/v1/audio/transcriptions" \
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
fi # profile: nar

if profile_active plus; then
echo ""
echo "=== Plus — speaker attribution on multi-speaker audio (port ${GRANITE_PLUS_DIRECT_PORT}) ==="
raw=$(curl -s "http://127.0.0.1:${GRANITE_PLUS_DIRECT_PORT}/v1/audio/transcriptions" \
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
fi # profile: plus

if profile_active base; then
echo ""
echo "=== Base — direct llama-server (port ${GRANITE_BASE_PROXY_PORT}, 10s clip) ==="
raw=$(curl -s "http://127.0.0.1:${GRANITE_BASE_PROXY_PORT}/v1/audio/transcriptions" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -F "model=ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  -F "file=@${AUDIO_SHORT}" \
  --form-string "prompt=<|audio|>transcribe the speech with proper punctuation and capitalization.")
echo "  raw: ${raw}"
python3 -c "
import sys, json, re
d = json.loads(sys.argv[1])
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
assert re.search(r'[A-Z]', text), 'FAIL: no capitalization found'
assert re.search(‘[.!?,\x27]’, text), ‘FAIL: no punctuation found’
print('  PASS: non-empty, capitalized, punctuated')
" "${raw}"
fi # profile: base

if profile_active plus; then
echo ""
echo "=== Plus — direct model with timestamps (port ${GRANITE_PLUS_PROXY_PORT}, 10s clip) ==="
raw=$(curl -s "http://127.0.0.1:${GRANITE_PLUS_PROXY_PORT}/v1/audio/transcriptions" \
  -H "Authorization: Bearer ${GRANITE_API_KEY}" \
  -F "file=@${AUDIO_SHORT}" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]")
echo "  raw: ${raw}"
python3 -c "
import sys, json, re
d = json.loads(sys.argv[1])
text = d['text']
print('  text:', text)
assert text.strip(), 'FAIL: text is empty'
assert re.search(r'\[T:\d+\]', text), 'FAIL: no timestamp tags found'
print('  PASS: timestamps present')
" "${raw}"
fi # profile: plus

echo "=== Done ==="
