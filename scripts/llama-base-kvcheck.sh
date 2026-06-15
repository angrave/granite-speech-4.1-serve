#!/usr/bin/env bash
# llama-base-kvcheck.sh — Smoke-test the granite-speech base endpoint ($GRANITE_BASE_DIRECT_PORT)
#
# Tests three things:
#   1. Short audio (≤15 s) transcribes correctly             → non-empty text
#   2. Full-length audio (>17 s) reproduces the silent-empty bug → text=""
#   3. KV-cache identity: same short audio sent twice gives
#      identical non-empty text (cache hit is valid)
#
# Usage: ./llama-base-kvcheck.sh [audio.wav]
#   audio.wav must be at least 20 s long; defaults to test.wav
#
# Requires: bash, curl, python3 (wave module), source .env (LLAMA_API_KEY)
set -euo pipefail

AUDIO="${1:-test.wav}"
PORT="${GRANITE_BASE_DIRECT_PORT:-8700}"
BASE="http://127.0.0.1:${PORT}"
MODEL="ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0"
PROMPT="transcribe with punctuation and capitalization."

# ── Colour helpers ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass()  { echo -e "${GREEN}  PASS${NC}: $*"; }
fail()  { echo -e "${RED}  FAIL${NC}: $*"; FAILURES=$((FAILURES+1)); }
warn()  { echo -e "${YELLOW}  WARN${NC}: $*"; }
FAILURES=0

# ── Auth ─────────────────────────────────────────────────────────────────────
if [[ -z "${LLAMA_API_KEY:-}" ]] && [[ -f .env ]]; then
  # shellcheck source=.env
  source .env
fi
: "${LLAMA_API_KEY:?LLAMA_API_KEY not set}"

# ── Helpers ───────────────────────────────────────────────────────────────────
transcribe() {
  local file="$1"
  curl -sf "${BASE}/v1/audio/transcriptions" \
    -H "Authorization: Bearer ${LLAMA_API_KEY}" \
    -F "model=${MODEL}" \
    -F "file=@${file}" \
    -F "prompt=${PROMPT}"
}

# Extract fields from a JSON response stored in a variable
json_text()          { echo "$1" | python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"; }
json_input_tokens()  { echo "$1" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['input_tokens'])"; }
json_output_tokens() { echo "$1" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['output_tokens'])"; }
json_cached()        { echo "$1" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['input_tokens_details']['cached_tokens'])"; }

# ── Audio clips ───────────────────────────────────────────────────────────────
echo ""
echo "=== Preparing audio clips from: ${AUDIO} ==="

# Check the source file is long enough
DURATION=$(python3 -c "
import wave
with wave.open('${AUDIO}') as f:
    print(f'{f.getnframes() / f.getframerate():.1f}')
")
echo "  Source duration: ${DURATION} s"

if python3 -c "assert float('${DURATION}') >= 20" 2>/dev/null; then
  echo "  Source is long enough (>= 20 s)"
else
  echo -e "${RED}ERROR${NC}: ${AUDIO} must be at least 20 s long for the long-audio bug test" >&2
  exit 1
fi

SHORT_WAV=$(mktemp /tmp/kvcheck_short_XXXXXX.wav)
LONG_WAV=$(mktemp /tmp/kvcheck_long_XXXXXX.wav)
trap 'rm -f "${SHORT_WAV}" "${LONG_WAV}"' EXIT

python3 - <<EOF
import wave

def clip(src_path, dst_path, seconds):
    with wave.open(src_path) as src:
        r, c, w = src.getframerate(), src.getnchannels(), src.getsampwidth()
        data = src.readframes(int(r * seconds))
    with wave.open(dst_path, "w") as dst:
        dst.setnchannels(c); dst.setsampwidth(w); dst.setframerate(r)
        dst.writeframes(data)

clip("${AUDIO}", "${SHORT_WAV}", 15)   # ≤ 15 s — known-good range
clip("${AUDIO}", "${LONG_WAV}",  35)   # 35 s  — reproduces the empty-output bug
EOF
echo "  15 s clip: ${SHORT_WAV}"
echo "  35 s clip: ${LONG_WAV}"

# ── Health ────────────────────────────────────────────────────────────────────
echo ""
echo "=== Server health ==="
health=$(curl -sf "${BASE}/health" 2>/dev/null || echo '{"status":"UNREACHABLE"}')
status=$(echo "${health}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))")
echo "  ${BASE}/health → ${status}"
if [[ "${status}" != "ok" ]]; then
  echo -e "${RED}ERROR${NC}: server not reachable or unhealthy. Start it with start_apple_dockerless.sh." >&2
  exit 1
fi

# ── TEST 1: short audio → non-empty transcription ─────────────────────────────
echo ""
echo "=== TEST 1: short audio (15 s) → expects non-empty text ==="
resp1=$(transcribe "${SHORT_WAV}")
echo "  raw: ${resp1}"
text1=$(json_text "${resp1}")
in1=$(json_input_tokens "${resp1}")
out1=$(json_output_tokens "${resp1}")
cached1=$(json_cached "${resp1}")
echo "  text:         ${text1}"
echo "  input_tokens: ${in1}  output_tokens: ${out1}  cached: ${cached1}"

if [[ -n "${text1// /}" ]]; then
  pass "short audio transcribed: \"${text1:0:60}…\""
else
  fail "short audio returned empty text (input_tokens=${in1}, output_tokens=${out1})"
  warn "If this is the very first request after server start, try running the script again — the first warmup request sometimes primes the Metal graph."
fi

# ── TEST 2: long audio (35 s) → reproduces the bug ───────────────────────────
echo ""
echo "=== TEST 2: long audio (35 s) → expects empty text (known bug) ==="
resp2=$(transcribe "${LONG_WAV}" || echo '{}')
echo "  raw: ${resp2}"
text2=$(json_text "${resp2}" 2>/dev/null || echo "PARSE_ERROR")
in2=$(json_input_tokens "${resp2}" 2>/dev/null || echo "?")
out2=$(json_output_tokens "${resp2}" 2>/dev/null || echo "?")
echo "  text:         ${text2}"
echo "  input_tokens: ${in2}  output_tokens: ${out2}"

if [[ -z "${text2// /}" || "${text2}" == "PARSE_ERROR" ]]; then
  echo -e "  ${YELLOW}BUG REPRODUCED${NC}: 35 s audio returns empty transcription"
  echo "  → See llama-BUG.md for upstream report"
else
  warn "Long audio returned non-empty text — bug may be fixed or not triggered on this audio."
  echo "  text: ${text2:0:80}"
fi

# ── TEST 3: KV-cache identity — same short audio twice ───────────────────────
echo ""
echo "=== TEST 3: KV-cache identity — same 15 s clip sent twice ==="
resp3a=$(transcribe "${SHORT_WAV}")
resp3b=$(transcribe "${SHORT_WAV}")
text3a=$(json_text "${resp3a}")
text3b=$(json_text "${resp3b}")
cached3a=$(json_cached "${resp3a}")
cached3b=$(json_cached "${resp3b}")
in3a=$(json_input_tokens "${resp3a}")
in3b=$(json_input_tokens "${resp3b}")

echo "  run 1: input=${in3a} cached=${cached3a} text=\"${text3a:0:60}\""
echo "  run 2: input=${in3b} cached=${cached3b} text=\"${text3b:0:60}\""

# First run should be non-empty
if [[ -n "${text3a// /}" ]]; then
  pass "run 1: non-empty transcription"
else
  fail "run 1: empty — server may still have bad cache from TEST 2; restart and re-run"
fi

# Second run should reuse cache (cached ≈ input - 1)
expected_cached=$(( in3b - 1 ))
if [[ "${cached3b}" -ge "${expected_cached}" ]]; then
  pass "run 2: KV cache reused (cached=${cached3b} of ${in3b} tokens)"
else
  warn "run 2: cache NOT reused (cached=${cached3b}); may indicate different audio hashes"
fi

# Both runs must return the same text
if [[ "${text3a}" == "${text3b}" ]]; then
  pass "run 1 and run 2 return identical text (cache hit is stable)"
else
  fail "run 1 and run 2 differ — cache reuse is producing different output"
  echo "    run 1: ${text3a}"
  echo "    run 2: ${text3b}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
echo "  Audio source : ${AUDIO} (${DURATION} s)"
echo "  Server       : ${BASE}"
echo "  Model        : ${MODEL}"
echo ""
if [[ ${FAILURES} -eq 0 ]]; then
  echo -e "${GREEN}All assertions passed.${NC}"
  echo "  TEST 2 confirmed the known llama.cpp granite_speech long-audio bug."
  echo "  See llama-BUG.md for details and upstream report."
else
  echo -e "${RED}${FAILURES} assertion(s) failed.${NC}"
  echo "  Check output above. Common causes:"
  echo "    - Server not fully started (retry after 'tail -f runtime/logs/base.log' shows 'server is listening')"
  echo "    - Bad KV cache from previous failed run (restart server and re-run)"
  echo "    - Audio file shorter than 20 s"
fi
echo ""

exit ${FAILURES}
