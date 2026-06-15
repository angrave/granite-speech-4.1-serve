#!/usr/bin/env bash
# test_long_duration.sh — Send a long looped audio file to all three services.
#
# Usage:  ./test_long_duration.sh [DURATION_MINUTES] [SOURCE_WAV]
#   DURATION_MINUTES  How many minutes of audio to generate (default: 33, range: 1-120)
#                     33 minutes exceeds the plus-model encoder context, forcing at least
#                     one chunk split on the plus proxy.
#   SOURCE_WAV        WAV file to loop (default: test.wav)
#
# The script loops SOURCE_WAV with ffmpeg to produce exactly DURATION_MINUTES of audio,
# sends it to all services, and checks:
#   - word count >= 60 wpm (detects token-limit truncation)
#   - chunk count > 1 (confirms chunking proxy is active)
#   - timestamps monotonically increasing (plus timestamps mode only)
set -euo pipefail

DURATION_MIN="${1:-33}"
SOURCE_WAV="${2:-test.wav}"

# ── Validate arguments ──────────────────────────────────────────────────────────

if ! [[ "$DURATION_MIN" =~ ^[0-9]+(\.[0-9]+)?$ ]] || \
   (( $(echo "$DURATION_MIN < 1"  | bc -l) )) || \
   (( $(echo "$DURATION_MIN > 120" | bc -l) )); then
  echo "ERROR: duration must be a number between 1 and 120 (minutes). Got: $DURATION_MIN" >&2
  exit 1
fi

[[ -f "$SOURCE_WAV" ]] || { echo "ERROR: $SOURCE_WAV not found" >&2; exit 1; }

command -v ffmpeg &>/dev/null || { echo "ERROR: ffmpeg not found (brew install ffmpeg)" >&2; exit 1; }

# ── Load env / keys ─────────────────────────────────────────────────────────────

if [[ -z "${LLAMA_API_KEY:-}" || -z "${GRANITE_API_KEY:-}" ]] && [[ -f .env ]]; then
  # shellcheck source=.env
  # shellcheck disable=SC1091
  source .env
fi
: "${LLAMA_API_KEY:?LLAMA_API_KEY not set. Run: source .env}"
: "${GRANITE_API_KEY:?GRANITE_API_KEY not set. Run: source .env}"

GRANITE_BASE_DIRECT_PORT="${GRANITE_BASE_DIRECT_PORT:-8700}"
GRANITE_PLUS_DIRECT_PORT="${GRANITE_PLUS_DIRECT_PORT:-8701}"
GRANITE_NAR_DIRECT_PORT="${GRANITE_NAR_DIRECT_PORT:-8702}"

# ── Build looped WAV ─────────────────────────────────────────────────────────────

DURATION_S=$(echo "$DURATION_MIN * 60" | bc | awk '{printf "%d", $1}')
TMPFILE=$(mktemp /tmp/long_audio_XXXXXX.wav)
trap 'rm -f "$TMPFILE"' EXIT

echo "=== Generating ${DURATION_MIN}-minute (${DURATION_S}s) looped audio from $SOURCE_WAV ==="
# -stream_loop -1: loop input indefinitely; -t: stop at DURATION_S seconds
ffmpeg -y -loglevel error \
  -stream_loop -1 -i "$SOURCE_WAV" \
  -t "$DURATION_S" \
  -ar 16000 -ac 1 -c:a pcm_s16le \
  "$TMPFILE"
SIZE_MB=$(du -m "$TMPFILE" | cut -f1)
echo "  Generated: $TMPFILE (${SIZE_MB} MB, ${DURATION_S}s @ 16kHz mono PCM)"
echo ""

# ── Helper ───────────────────────────────────────────────────────────────────────

run_test() {
  local label="$1"; local port="$2"; local key="$3"; shift 3
  echo "=== $label (port $port) ==="

  local start_ms
  start_ms=$(python3 -c "import time; print(int(time.time()*1000))")

  local raw
  raw=$(curl -s --max-time 7200 \
    "http://127.0.0.1:${port}/v1/audio/transcriptions" \
    -H "Authorization: Bearer ${key}" \
    -F "file=@${TMPFILE}" \
    "$@") || { echo "  ERROR: curl failed"; echo ""; return; }

  local end_ms elapsed_ms
  end_ms=$(python3 -c "import time; print(int(time.time()*1000))")
  elapsed_ms=$(( end_ms - start_ms ))

  echo "  elapsed: ${elapsed_ms}ms"

  python3 - "$raw" "$elapsed_ms" "$DURATION_S" <<'PYEOF'
import sys, json

raw, elapsed_ms, dur_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
elapsed = elapsed_ms / 1000
dur_min = dur_s / 60
try:
    d = json.loads(raw)
except json.JSONDecodeError:
    print(f"  ERROR: invalid JSON: {raw[:200]}")
    sys.exit(0)

text   = d.get("text", "")
words  = text.split()
usage  = d.get("usage", {})
chunks = usage.get("chunks", 1)

speed = dur_s / elapsed if elapsed else float("inf")
wpm   = len(words) / dur_min if dur_min else 0

print(f"  chunks : {chunks}")
print(f"  words  : {len(words)}  ({wpm:.0f} wpm)")
print(f"  chars  : {len(text)}")
print(f"  speed  : {speed:.1f}x realtime  ({elapsed:.1f}s to process {dur_s}s of audio)")
print(f"  text   : {text[:200]}{'...' if len(text)>200 else ''}")

ok = True

# Word-count sanity check: expect ≥ 60 wpm (very conservative; looped
# speech with pauses typically produces 100–130 wpm).  Failure almost
# always means the model stopped early due to token-limit truncation.
min_words = int(dur_min * 60)
if len(words) >= min_words:
    print(f"  PASS: word count {len(words)} >= {min_words} (60 wpm floor)")
else:
    print(f"  FAIL: word count {len(words)} < {min_words} — likely truncation "
          f"(expected ~{int(dur_min * 120)} words at 120 wpm)")
    ok = False

# Chunk-count sanity check: for audio longer than 14 s the proxy must split.
# A chunk count of 1 on long audio means chunking did not fire.
if dur_s > 14 and chunks <= 1:
    print(f"  FAIL: chunks={chunks} — expected >1 for {dur_min:.0f}-minute audio "
          f"(chunking proxy may not be running)")
    ok = False
elif dur_s > 14:
    print(f"  PASS: chunks={chunks} > 1 (chunking confirmed)")

if not text.strip():
    print("  FAIL: text is empty")
    ok = False

if ok and text.strip():
    print("  PASS")
PYEOF

  echo ""
}

# ── Health check ─────────────────────────────────────────────────────────────────

echo "=== Health check ==="
for port in "${GRANITE_BASE_DIRECT_PORT}" "${GRANITE_PLUS_DIRECT_PORT}" "${GRANITE_NAR_DIRECT_PORT}"; do
  result=$(curl -sf "http://127.0.0.1:${port}/health" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'), '| auth:', d.get('auth','n/a'))" \
    2>/dev/null || echo "UNREACHABLE")
  echo "  Port ${port}: ${result}"
done
echo ""

# ── Send to each service ──────────────────────────────────────────────────────────

run_test "Base — llama.cpp chunking proxy" "${GRANITE_BASE_DIRECT_PORT}" "$LLAMA_API_KEY" \
  -F "model=ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  -F "prompt=transcribe with punctuation and capitalization."

run_test "Plus — plain ASR" "${GRANITE_PLUS_DIRECT_PORT}" "$GRANITE_API_KEY" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> can you transcribe the speech into a written format?"

run_test_timestamps() {
  local label="$1"; local port="$2"; local key="$3"; shift 3
  echo "=== $label (port $port) ==="

  local start_ms
  start_ms=$(python3 -c "import time; print(int(time.time()*1000))")

  local raw
  raw=$(curl -s --max-time 7200 \
    "http://127.0.0.1:${port}/v1/audio/transcriptions" \
    -H "Authorization: Bearer ${key}" \
    -F "file=@${TMPFILE}" \
    "$@") || { echo "  ERROR: curl failed"; echo ""; return; }

  local end_ms elapsed_ms
  end_ms=$(python3 -c "import time; print(int(time.time()*1000))")
  elapsed_ms=$(( end_ms - start_ms ))
  echo "  elapsed: ${elapsed_ms}ms"

  python3 - "$raw" "$elapsed_ms" "$DURATION_S" <<'PYEOF'
import sys, json, re

raw, elapsed_ms, dur_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
elapsed = elapsed_ms / 1000
dur_min = dur_s / 60
try:
    d = json.loads(raw)
except json.JSONDecodeError:
    print(f"  ERROR: invalid JSON: {raw[:200]}")
    sys.exit(0)

text   = d.get("text", "")
usage  = d.get("usage", {})
chunks = usage.get("chunks", 1)
tags   = [int(m.group(1)) for m in re.finditer(r'\[T:(\d+)\]', text)]

# Spoken words: strip [T:N] timestamp tags and _ silence markers before counting.
# Raw text.split() includes those tokens and inflates the count ~2.5×.
spoken_text  = re.sub(r'\[T:\d+\]', '', text)   # remove timestamp tags
spoken_text  = re.sub(r'\b_\b', '', spoken_text)  # remove silence markers
spoken_words = spoken_text.split()
wpm          = len(spoken_words) / dur_min if dur_min else 0

print(f"  chunks        : {chunks}")
print(f"  spoken words  : {len(spoken_words)}  ({wpm:.0f} wpm)")
print(f"  timestamps    : {len(tags)}")
print(f"  text          : {text[:200]}{'...' if len(text)>200 else ''}")

ok = True

if not text.strip():
    print("  FAIL: text is empty")
    ok = False

# 60 wpm floor on spoken words (not raw tokens).
min_words = int(dur_min * 60)
if len(spoken_words) >= min_words:
    print(f"  PASS: spoken word count {len(spoken_words)} >= {min_words} (60 wpm floor)")
else:
    print(f"  FAIL: spoken word count {len(spoken_words)} < {min_words} — likely truncation")
    ok = False

if dur_s > 14 and chunks <= 1:
    print(f"  FAIL: chunks={chunks} — expected >1 for {dur_min:.0f}-minute audio")
    ok = False
elif dur_s > 14:
    print(f"  PASS: chunks={chunks} > 1 (chunking confirmed)")

if not tags:
    print("  FAIL: no [T:N] timestamp tags in output")
    ok = False
else:
    # Monotone check
    non_mono = [(i, tags[i-1], tags[i]) for i in range(1, len(tags)) if tags[i] < tags[i-1]]
    if non_mono:
        print(f"  FAIL: {len(non_mono)} non-monotone timestamp(s) — first: "
              f"index {non_mono[0][0]}: [T:{non_mono[0][1]}]→[T:{non_mono[0][2]}]")
        ok = False
    else:
        print(f"  PASS: {len(tags)} timestamps monotonically increasing "
              f"[{tags[0]}..{tags[-1]}] cs")

    # Range check: last timestamp should be within 30% of the audio duration.
    # Catches cases where chunking stopped early or timestamps reset mid-file.
    expected_cs = dur_s * 100
    lo, hi = int(expected_cs * 0.70), int(expected_cs * 1.30)
    last_cs = tags[-1]
    if lo <= last_cs <= hi:
        print(f"  PASS: last timestamp {last_cs} cs within 30% of "
              f"expected {expected_cs} cs ({dur_s}s audio)")
    else:
        print(f"  FAIL: last timestamp {last_cs} cs outside ±30% of "
              f"expected {expected_cs} cs — timestamps may not span full audio")
        ok = False

if ok:
    print("  PASS")
PYEOF

  echo ""
}

run_test_timestamps "Plus — timestamps (monotone check)" "${GRANITE_PLUS_DIRECT_PORT}" "$GRANITE_API_KEY" \
  -F "model=plus" \
  --form-string "prompt=<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"

run_test "NAR" "${GRANITE_NAR_DIRECT_PORT}" "$GRANITE_API_KEY" \
  -F "model=nar"

echo "=== Done ==="
