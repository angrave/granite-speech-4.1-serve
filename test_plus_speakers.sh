#!/usr/bin/env bash
# test_plus_speakers.sh — Speaker attribution tests for the plus proxy.
#
# Tests speaker diarization through the proxy (:8001) and verifies results
# against the internal endpoint (:18001) as ground truth.
#
# Fixtures in test_audio/ (regenerate with: ./generate_test_audio.sh):
#   Source: 4404.mp3 (2 female), 4074.mp3 (2 male)
#
#   2spk_short_30s.wav      — 2 female speakers (4404), 30s
#   2spk_medium_90s.wav     — 2 female speakers (4404), 90s
#   2spk_long_150s.wav      — 2 female speakers (4404), 150s (forces chunking)
#   2spk_male_short_30s.wav — 2 male speakers (4074), 30s
#   2spk_male_long_180s.wav — 2 male speakers (4074), 180s (forces chunking)
#   4spk_short_62s.wav      — 4 voices: female pair + male pair interleaved, 62s
#   4spk_long_153s.wav      — 4 voices: female 75s + male 75s sequential, 153s
#
# Usage: ./test_plus_speakers.sh [--internal-only] [--proxy-only]
#
# Requires: bash, curl, python3, source .env (GRANITE_API_KEY)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Options ──────────────────────────────────────────────────────────────────
RUN_INTERNAL=true
RUN_PROXY=true
for arg in "$@"; do
  case "$arg" in
    --internal-only) RUN_PROXY=false ;;
    --proxy-only)    RUN_INTERNAL=false ;;
  esac
done

# ── Colour helpers ───────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass()  { echo -e "  ${GREEN}PASS${NC}: $*"; }
fail()  { echo -e "  ${RED}FAIL${NC}: $*"; FAILURES=$((FAILURES+1)); }
warn()  { echo -e "  ${YELLOW}WARN${NC}: $*"; }
info()  { echo -e "  $*"; }
FAILURES=0

# ── Auth ─────────────────────────────────────────────────────────────────────
if [[ -z "${GRANITE_API_KEY:-}" ]] && [[ -f .env ]]; then
  source .env
fi
: "${GRANITE_API_KEY:?GRANITE_API_KEY not set}"

PROXY_URL="http://127.0.0.1:8001"
INTERNAL_URL="http://127.0.0.1:18001"
SPK_PROMPT='<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.'
COMBINED_PROMPT='<|audio|> Timestamps and Speaker attribution: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]. Denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.'

# ── Helpers ──────────────────────────────────────────────────────────────────
transcribe() {
  local url="$1" file="$2" prompt="${3:-$SPK_PROMPT}"
  curl -sf "${url}/v1/audio/transcriptions" \
    -H "Authorization: Bearer ${GRANITE_API_KEY}" \
    -F "file=@${file}" \
    -F "model=plus" \
    --form-string "prompt=${prompt}" \
    --max-time 300
}

count_speakers() {
  python3 -c "
import sys, re
speakers = set(re.findall(r'\[Speaker (\d+)\]:', sys.argv[1]))
print(len(speakers))
" "$1"
}

extract_text() {
  python3 -c "import sys,json; print(json.load(sys.stdin)['text'])"
}

extract_chunks() {
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('usage',{}).get('chunks','n/a'))"
}

# ── Health check ─────────────────────────────────────────────────────────────
echo ""
echo "=== Health check ==="
for label_url in "proxy:${PROXY_URL}" "internal:${INTERNAL_URL}"; do
  label="${label_url%%:*}"
  url="${label_url#*:}"
  status=$(curl -sf "${url}/health" 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','?'))" \
    2>/dev/null || echo "UNREACHABLE")
  if [[ "$status" == "ok" ]]; then
    pass "${label} (${url}) is healthy"
  else
    fail "${label} (${url}) is ${status}"
  fi
done

if [[ $FAILURES -gt 0 ]]; then
  echo -e "\n${RED}Cannot continue — servers not healthy.${NC}"
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: 2 female speakers, short (30s) — baseline
#   Internal: ground truth (should always find 2 speakers)
#   Proxy: currently chunks at 14s → loses Speaker 2 for similar-voiced pairs
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 1: 2 female speakers, short 30s (4404) — baseline ==="
FILE="test_audio/2spk_short_30s.wav"

if $RUN_INTERNAL; then
  echo "  --- internal (:18001) ---"
  resp=$(transcribe "$INTERNAL_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  n=$(count_speakers "$text")
  info "speakers: $n  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "internal: $n speakers" || fail "internal: only $n speaker(s)"
fi

if $RUN_PROXY; then
  echo "  --- proxy (:8001) ---"
  resp=$(transcribe "$PROXY_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  chunks=$(echo "$resp" | extract_chunks)
  n=$(count_speakers "$text")
  info "speakers: $n  chunks: $chunks  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "proxy: $n speakers" || fail "proxy: only $n speaker(s) — chunking lost Speaker 2"
  [[ "$chunks" == "1" ]] && pass "proxy: single chunk" || warn "proxy: $chunks chunks for 30s"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: 2 male speakers, short (30s) — distinct-voice control
#   Male voices in 4074 are more distinct; even chunked output may retain both.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 2: 2 male speakers, short 30s (4074) — distinct-voice control ==="
FILE="test_audio/2spk_male_short_30s.wav"

if $RUN_INTERNAL; then
  echo "  --- internal (:18001) ---"
  resp=$(transcribe "$INTERNAL_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  n=$(count_speakers "$text")
  info "speakers: $n  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "internal: $n speakers" || fail "internal: only $n speaker(s)"
fi

if $RUN_PROXY; then
  echo "  --- proxy (:8001) ---"
  resp=$(transcribe "$PROXY_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  chunks=$(echo "$resp" | extract_chunks)
  n=$(count_speakers "$text")
  info "speakers: $n  chunks: $chunks  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "proxy: $n speakers (distinct voices survived chunking)" \
                    || fail "proxy: only $n speaker(s)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: 2 female speakers, medium (90s) — within unchunked threshold
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 3: 2 female speakers, medium 90s (4404) ==="
FILE="test_audio/2spk_medium_90s.wav"

if $RUN_INTERNAL; then
  echo "  --- internal (:18001) ---"
  resp=$(transcribe "$INTERNAL_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  n=$(count_speakers "$text")
  info "speakers: $n  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "internal: $n speakers" || fail "internal: only $n speaker(s)"
fi

if $RUN_PROXY; then
  echo "  --- proxy (:8001) ---"
  resp=$(transcribe "$PROXY_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  chunks=$(echo "$resp" | extract_chunks)
  n=$(count_speakers "$text")
  info "speakers: $n  chunks: $chunks  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "proxy: $n speakers" || fail "proxy: only $n speaker(s)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: 2 female speakers, long (150s) — over 120s, forces Phase 2 chunking
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 4: 2 female speakers, long 150s (4404) — over threshold ==="
FILE="test_audio/2spk_long_150s.wav"

if $RUN_INTERNAL; then
  echo "  --- internal (:18001) ---"
  resp=$(transcribe "$INTERNAL_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  n=$(count_speakers "$text")
  info "speakers: $n  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "internal: $n speakers (ground truth)" \
                    || warn "internal: only $n — may exceed context window"
fi

if $RUN_PROXY; then
  echo "  --- proxy (:8001) ---"
  resp=$(transcribe "$PROXY_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  chunks=$(echo "$resp" | extract_chunks)
  n=$(count_speakers "$text")
  info "speakers: $n  chunks: $chunks  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "proxy: $n speakers across $chunks chunks" \
                    || fail "proxy: only $n speaker(s) across $chunks chunks — needs Phase 2 preamble"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: 2 male speakers, long (180s) — over 120s, forces Phase 2 chunking
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 5: 2 male speakers, long 180s (4074) — over threshold ==="
FILE="test_audio/2spk_male_long_180s.wav"

if $RUN_PROXY; then
  echo "  --- proxy (:8001) ---"
  resp=$(transcribe "$PROXY_URL" "$FILE")
  text=$(echo "$resp" | extract_text)
  chunks=$(echo "$resp" | extract_chunks)
  n=$(count_speakers "$text")
  info "speakers: $n  chunks: $chunks  text: ${text:0:120}..."
  [[ "$n" -ge 2 ]] && pass "proxy: $n speakers across $chunks chunks" \
                    || fail "proxy: only $n speaker(s) across $chunks chunks — needs Phase 2 preamble"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: 4-voice short (62s) — model speaker-count limit test
#   Female pair (4404) + male pair (4074), interleaved 15s segments.
#   Model currently maxes out at 2 speaker labels — this test documents that.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 6: 4-voice interleaved 62s (female+male) — model limit test ==="
FILE="test_audio/4spk_short_62s.wav"
SPK4_PROMPT='<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]:, [Speaker 2]:, [Speaker 3]:, and [Speaker 4]: tags before speaker turns.'

if $RUN_INTERNAL; then
  echo "  --- internal (:18001) with 4-speaker prompt ---"
  resp=$(transcribe "$INTERNAL_URL" "$FILE" "$SPK4_PROMPT")
  text=$(echo "$resp" | extract_text)
  n=$(count_speakers "$text")
  info "speakers: $n  text: ${text:0:150}..."
  if [[ "$n" -ge 4 ]]; then
    pass "internal: all 4 speakers detected"
  elif [[ "$n" -ge 3 ]]; then
    warn "internal: $n/4 speakers — partial detection"
  else
    warn "internal: $n speakers — model appears to max out at 2 (known limitation)"
  fi
fi

if $RUN_PROXY; then
  echo "  --- proxy (:8001) with 4-speaker prompt ---"
  resp=$(transcribe "$PROXY_URL" "$FILE" "$SPK4_PROMPT")
  text=$(echo "$resp" | extract_text)
  chunks=$(echo "$resp" | extract_chunks)
  n=$(count_speakers "$text")
  info "speakers: $n  chunks: $chunks  text: ${text:0:150}..."
  if [[ "$n" -ge 4 ]]; then
    pass "proxy: all 4 speakers detected"
  elif [[ "$n" -ge 2 ]]; then
    warn "proxy: $n speakers — model limitation (not a proxy bug)"
  else
    fail "proxy: only $n speaker(s)"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: 4-voice long (153s) — over threshold + model limit
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 7: 4-voice sequential 153s (female+male) — over threshold ==="
FILE="test_audio/4spk_long_153s.wav"

if $RUN_PROXY; then
  echo "  --- proxy (:8001) with 4-speaker prompt ---"
  resp=$(transcribe "$PROXY_URL" "$FILE" "$SPK4_PROMPT")
  text=$(echo "$resp" | extract_text)
  chunks=$(echo "$resp" | extract_chunks)
  n=$(count_speakers "$text")
  info "speakers: $n  chunks: $chunks  text: ${text:0:150}..."
  if [[ "$n" -ge 2 ]]; then
    warn "proxy: $n speakers across $chunks chunks (model maxes at 2)"
  else
    fail "proxy: only $n speaker(s) across $chunks chunks"
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Combined speaker + timestamps, female 30s
#
# KNOWN LIMITATION: The model collapses to 1 speaker when given the combined
# (speaker + timestamps) prompt with similar-sounding female voices, even when
# sent as a single full-audio request (no chunking).  The internal endpoint
# (:18001) exhibits the same behaviour — this is a model limitation, not a
# proxy bug.  Male voices (Test 9) are distinct enough to survive the combined
# prompt.  Failure here is expected and does not indicate a regression.
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 8: Combined speaker + timestamps, female 30s (4404) ==="
FILE="test_audio/2spk_short_30s.wav"

if $RUN_PROXY; then
  echo "  --- proxy (:8001) ---"
  resp=$(transcribe "$PROXY_URL" "$FILE" "$COMBINED_PROMPT")
  text=$(echo "$resp" | extract_text)
  n=$(count_speakers "$text")
  has_ts=$(python3 -c "import sys,re; print('yes' if re.search(r'\[T:\d+\]', sys.argv[1]) else 'no')" "$text")
  info "speakers: $n  timestamps: $has_ts"
  info "text: ${text:0:200}..."
  [[ "$n" -ge 2 ]] && pass "combined: $n speakers" || warn "combined: only $n speaker(s) — known model limitation with female voices + combined prompt"
  [[ "$has_ts" == "yes" ]] && pass "combined: timestamps present" || fail "combined: no timestamps"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Combined speaker + timestamps, male 30s
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 9: Combined speaker + timestamps, male 30s (4074) ==="
FILE="test_audio/2spk_male_short_30s.wav"

if $RUN_PROXY; then
  echo "  --- proxy (:8001) ---"
  resp=$(transcribe "$PROXY_URL" "$FILE" "$COMBINED_PROMPT")
  text=$(echo "$resp" | extract_text)
  n=$(count_speakers "$text")
  has_ts=$(python3 -c "import sys,re; print('yes' if re.search(r'\[T:\d+\]', sys.argv[1]) else 'no')" "$text")
  info "speakers: $n  timestamps: $has_ts"
  info "text: ${text:0:200}..."
  [[ "$n" -ge 2 ]] && pass "combined: $n speakers" || fail "combined: only $n speaker(s)"
  [[ "$has_ts" == "yes" ]] && pass "combined: timestamps present" || fail "combined: no timestamps"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: Proxy vs internal consistency — female (30s)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 10: Proxy vs internal consistency — female 30s ==="
FILE="test_audio/2spk_short_30s.wav"

if $RUN_INTERNAL && $RUN_PROXY; then
  resp_i=$(transcribe "$INTERNAL_URL" "$FILE")
  text_i=$(echo "$resp_i" | extract_text)
  resp_p=$(transcribe "$PROXY_URL" "$FILE")
  text_p=$(echo "$resp_p" | extract_text)

  similarity=$(python3 -c "
import re, sys
def words(t):
    t = re.sub(r'\[Speaker \d+\]:', '', t)
    return set(t.lower().split())
w_i = words(sys.argv[1])
w_p = words(sys.argv[2])
if not w_i: print('0.0')
else: print(f'{len(w_i & w_p) / len(w_i | w_p):.2f}')
" "$text_i" "$text_p")
  info "word overlap (Jaccard): $similarity"
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.70 else 1)" "$similarity"; then
    pass "proxy consistent with internal (${similarity})"
  else
    fail "proxy diverges from internal (${similarity})"
    info "internal: ${text_i:0:100}..."
    info "proxy:    ${text_p:0:100}..."
  fi
else
  warn "skipped — needs both endpoints"
fi

# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: Proxy vs internal consistency — male (30s)
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo "=== TEST 11: Proxy vs internal consistency — male 30s ==="
FILE="test_audio/2spk_male_short_30s.wav"

if $RUN_INTERNAL && $RUN_PROXY; then
  resp_i=$(transcribe "$INTERNAL_URL" "$FILE")
  text_i=$(echo "$resp_i" | extract_text)
  resp_p=$(transcribe "$PROXY_URL" "$FILE")
  text_p=$(echo "$resp_p" | extract_text)

  similarity=$(python3 -c "
import re, sys
def words(t):
    t = re.sub(r'\[Speaker \d+\]:', '', t)
    return set(t.lower().split())
w_i = words(sys.argv[1])
w_p = words(sys.argv[2])
if not w_i: print('0.0')
else: print(f'{len(w_i & w_p) / len(w_i | w_p):.2f}')
" "$text_i" "$text_p")
  info "word overlap (Jaccard): $similarity"
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.70 else 1)" "$similarity"; then
    pass "proxy consistent with internal (${similarity})"
  else
    fail "proxy diverges from internal (${similarity})"
    info "internal: ${text_i:0:100}..."
    info "proxy:    ${text_p:0:100}..."
  fi
else
  warn "skipped — needs both endpoints"
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
echo "  Fixtures:  test_audio/  (regenerate: ./generate_test_audio.sh)"
echo "  Sources:   4404.mp3 (female pair), 4074.mp3 (male pair)"
echo "  Proxy:     ${PROXY_URL}"
echo "  Internal:  ${INTERNAL_URL}"
echo ""
echo "  Known model limitations:"
echo "    - Only 2 speaker labels used regardless of how many distinct voices are present."
echo "    - Combined (speaker + timestamps) prompt collapses female speakers to 1 label."
echo "      Internal endpoint (:18001) has the same behaviour — not a proxy bug."
echo "      Male voices are distinct enough to survive the combined prompt (Test 9)."
echo "  See PLUS-SPEAKER-REWORK.md for full analysis."
echo ""
if [[ ${FAILURES} -eq 0 ]]; then
  echo -e "${GREEN}All assertions passed.${NC}"
else
  echo -e "${RED}${FAILURES} assertion(s) failed.${NC}"
  echo "  Tests 1,3,4 proxy failures expected before Phase 1 (skip chunking for speaker mode)."
  echo "  Tests 4-5 proxy failures expected before Phase 2 (speaker preamble for >120s)."
fi
echo ""
exit ${FAILURES}
