#!/usr/bin/env bash
# generate_test_audio.sh — Create test WAV fixtures from source MP3 files.
#
# Source files (not committed — too large):
#   4404.mp3  — 30:00, 2 female speakers (cruise/family conversation)
#   4941.mp3  — 20:32, 2 female speakers (one in Israel, language symposium)
#   4074.mp3  — 30:00, 2 male speakers (accents/database conversation)
#
# Output: test_audio/*.wav (16 kHz mono PCM_16)
#
# Requires: ffmpeg
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUT="test_audio"

for src in 4404.mp3 4074.mp3; do
  [[ -f "$src" ]] || { echo "ERROR: $src not found"; exit 1; }
done

mkdir -p "$OUT"

# ── Helper ───────────────────────────────────────────────────────────────────
wav() {
  # wav <input.mp3> <output.wav> <start_s> <duration_s>
  ffmpeg -y -i "$1" -ss "$3" -t "$4" -ar 16000 -ac 1 "$2" 2>/dev/null
  local dur
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$2")
  echo "  $(basename "$2"): ${dur}s"
}

echo "=== Generating test fixtures ==="

# ── 2-speaker tests (female pair from 4404) ──────────────────────────────────
echo ""
echo "-- 2-speaker (4404, female pair) --"
wav 4404.mp3 "$OUT/2spk_short_30s.wav"    0  30
wav 4404.mp3 "$OUT/2spk_medium_90s.wav"   0  90
wav 4404.mp3 "$OUT/2spk_long_150s.wav"    0 150

# ── 2-speaker tests (male pair from 4074) ────────────────────────────────────
echo ""
echo "-- 2-speaker (4074, male pair) --"
wav 4074.mp3 "$OUT/2spk_male_short_30s.wav"   0  30
wav 4074.mp3 "$OUT/2spk_male_long_180s.wav"   0 180

# ── 4-speaker tests (female pair 4404 + male pair 4074) ──────────────────────
# Interleaved: 15s female, 0.5s silence, 15s male, 0.5s silence, repeat
echo ""
echo "-- 4-speaker (4404 female + 4074 male, interleaved) --"

# Extract building blocks
ffmpeg -y -i 4404.mp3 -ss  0 -t 15 -ar 16000 -ac 1 /tmp/gen_f1.wav 2>/dev/null
ffmpeg -y -i 4404.mp3 -ss 15 -t 15 -ar 16000 -ac 1 /tmp/gen_f2.wav 2>/dev/null
ffmpeg -y -i 4404.mp3 -ss 30 -t 15 -ar 16000 -ac 1 /tmp/gen_f3.wav 2>/dev/null
ffmpeg -y -i 4074.mp3 -ss  0 -t 15 -ar 16000 -ac 1 /tmp/gen_m1.wav 2>/dev/null
ffmpeg -y -i 4074.mp3 -ss 15 -t 15 -ar 16000 -ac 1 /tmp/gen_m2.wav 2>/dev/null
ffmpeg -y -i 4074.mp3 -ss 30 -t 15 -ar 16000 -ac 1 /tmp/gen_m3.wav 2>/dev/null

# Short 4-speaker (62s): F(0-15) silence M(0-15) silence F(15-30) silence M(15-30)
ffmpeg -y \
  -i /tmp/gen_f1.wav -i /tmp/gen_m1.wav \
  -i /tmp/gen_f2.wav -i /tmp/gen_m2.wav \
  -filter_complex "
    aevalsrc=0:d=0.5:s=16000:c=mono[s1];
    aevalsrc=0:d=0.5:s=16000:c=mono[s2];
    aevalsrc=0:d=0.5:s=16000:c=mono[s3];
    [0:a][s1][1:a][s2][2:a][s3][3:a]concat=n=7:v=0:a=1[out]" \
  -map "[out]" "$OUT/4spk_short_62s.wav" 2>/dev/null
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/4spk_short_62s.wav")
echo "  4spk_short_62s.wav: ${dur}s"

# Long 4-speaker (153s): F(0-75) silence(3s) M(0-75)
ffmpeg -y -i 4404.mp3 -ss 0 -t 75 -ar 16000 -ac 1 /tmp/gen_f_long.wav 2>/dev/null
ffmpeg -y -i 4074.mp3 -ss 0 -t 75 -ar 16000 -ac 1 /tmp/gen_m_long.wav 2>/dev/null
ffmpeg -y \
  -i /tmp/gen_f_long.wav -i /tmp/gen_m_long.wav \
  -filter_complex "
    aevalsrc=0:d=3:s=16000:c=mono[gap];
    [0:a][gap][1:a]concat=n=3:v=0:a=1[out]" \
  -map "[out]" "$OUT/4spk_long_153s.wav" 2>/dev/null
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT/4spk_long_153s.wav")
echo "  4spk_long_153s.wav: ${dur}s"

# ── Cleanup ──────────────────────────────────────────────────────────────────
rm -f /tmp/gen_f*.wav /tmp/gen_m*.wav

echo ""
echo "=== Done ==="
ls -lh "$OUT"/*.wav
