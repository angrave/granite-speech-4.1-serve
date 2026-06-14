# PLUS-SPEAKER-REWORK: Speaker-aware chunking for serve_plus_proxy.py

## Problem

The plus proxy chunks audio at 14s for all modes. The granite-speech-4.1-2b-plus
model cannot reliably identify multiple speakers in short chunks — each chunk
independently assigns `[Speaker 1]:` to whatever it hears, losing speaker
diversity. A 35s two-speaker conversation returns all `[Speaker 1]:` through the
proxy, while the internal endpoint (unchunked) correctly returns both speakers.

Verified empirically: sending 4 individually-chunked segments of `test.wav` to
`:18701` returns `[Speaker 1]:` only in every chunk. Sending the full 35s returns
correct `[Speaker 1]:` / `[Speaker 2]:` turns.

The severity depends on voice distinctiveness: male pairs (4074.mp3) retain both
speakers even when chunked, but female pairs (4404.mp3) with more similar voices
collapse to a single speaker.

## Model speaker-count limitation

Tested with 4 distinct voices (2 female from 4404.mp3 + 2 male from 4074.mp3)
interleaved with silence gaps (0.5s and 3s) and sequentially (75s+75s). Even
with explicit 4-speaker prompting, the model only ever uses `[Speaker 1]:` and
`[Speaker 2]:`. This appears to be a fundamental model limitation — the
granite-speech-4.1-2b-plus model maxes out at 2 speaker labels regardless of
how many distinct voices are present.

This means the preamble approach (Phase 2) only needs to handle 2 reference
clips, simplifying implementation. The 4-speaker test fixtures document this
limitation and will detect if a future model version adds support.

## Test fixtures

Source files (not committed — `.gitignore`):
- `4404.mp3` — 30:00, 2 female speakers (cruise/family conversation)
- `4074.mp3` — 30:00, 2 male speakers (accents/database conversation)
- `4941.mp3` — 20:32, 2 female speakers (Israel/language symposium) — available but not used

Generated fixtures (`./generate_test_audio.sh`):

| File | Duration | Content | Purpose |
|------|----------|---------|---------|
| `2spk_short_30s.wav` | 30s | 4404 female pair | Baseline, Phase 1 regression |
| `2spk_male_short_30s.wav` | 30s | 4074 male pair | Distinct-voice control |
| `2spk_medium_90s.wav` | 90s | 4404 female pair | Under 120s threshold |
| `2spk_long_150s.wav` | 150s | 4404 female pair | Over threshold, Phase 2 test |
| `2spk_male_long_180s.wav` | 180s | 4074 male pair | Over threshold, Phase 2 test |
| `4spk_short_62s.wav` | 62s | Female+male interleaved | Model limit test |
| `4spk_long_153s.wav` | 153s | Female 75s + male 75s | Model limit + chunking |

Test script: `./test_plus_speakers.sh` (11 tests, `--internal-only` / `--proxy-only`)

## Context budget (4096 max_position_embeddings)

Measured empirically with the real tokenizer and processor:

- Audio tokens: **10 tokens/sec** (constant)
- Prompt overhead: **80 tokens** (system + user prompt with speaker attribution)
- Remaining for audio + output: **4016 tokens**

Output token rates by mode (from 35s test.wav transcription):

| Mode                        | Output tok/s | Total tok/s | Max audio duration |
|-----------------------------|-------------|------------|-------------------|
| Speakers only (2 spk)       | 3.0         | 13.0       | ~308s (~5 min)    |
| 4 speakers, frequent turns  | 9.7         | 19.7       | ~204s (~3.4 min)  |
| Speakers + timestamps       | 15.9        | 25.9       | ~155s (~2.5 min)  |
| 4 spk + timestamps (worst)  | ~20         | ~30        | ~134s (~2.2 min)  |

The model documentation claims up to 4 speakers, but empirical testing shows it
maxes out at 2 (`[Speaker 1]:` and `[Speaker 2]:`). See "Model speaker-count
limitation" above.

## Design

### Phase 1: Skip chunking for short audio in speaker/combined modes

For audio below a safe duration threshold, skip chunking entirely and send to the
internal endpoint as a single request. This is the simplest fix and covers most
real-world use cases.

**Threshold calculation:**
Use a conservative threshold that works for the worst case (4 speakers +
timestamps). With 4016 tokens available and ~30 tok/s total consumption:
`4016 / 30 = ~134s`. Apply a safety margin: **120 seconds (2 min)**.

Add `PLUS_SPEAKER_MAX_UNCHUNKED_S` env var (default `120`). When mode is
`speakers` or `combined` and `dur_s <= PLUS_SPEAKER_MAX_UNCHUNKED_S`, send
the full audio without chunking.

**Files to modify:** `serve_plus_proxy.py` — the `transcribe()` endpoint.

**Changes:**
1. Add the env var and constant at the top (near `MAX_CHUNK_S`).
2. In `transcribe()`, after `mode = detect_mode(prompt)`, add the skip-chunking
   logic:
   ```python
   if mode in ("speakers", "combined") and dur_s <= PLUS_SPEAKER_MAX_UNCHUNKED_S:
       chunks = [(pcm, 0)]
   ```
3. Log when this path is taken:
   ```
   [plus-proxy] 35.0s audio -> 1 chunk(s), mode=speakers (full audio - speaker mode)
   ```

**Test:** Run the speaker attribution test from `test_endpoints.sh` (lines 147-164).
The assertion `'[Speaker 1]:' in text and '[Speaker 2]:' in text` must pass.
Also test with timestamp+speaker combined prompt.

### Phase 2: Speaker-reference preamble for long audio (>120s)

For audio exceeding the unchunked threshold, prepend short speaker reference
clips to each chunk so the model can identify speakers consistently.

#### Step 2a: Bootstrap — extract speaker reference clips

1. Send the first `MAX_CHUNK_S` seconds (14s) to the internal endpoint with the
   speaker attribution prompt. This is the "bootstrap chunk."
2. Parse the result for `[Speaker N]:` tags. If only one speaker is found,
   progressively extend the bootstrap segment (try 20s, 28s, up to
   `PLUS_SPEAKER_MAX_UNCHUNKED_S`) until at least 2 speakers are detected or
   the limit is reached.
3. For each detected speaker, extract a ~3s audio clip from the middle of their
   longest turn. Store as `ref_clips: dict[str, np.ndarray]` mapping speaker
   label to PCM.

**How to find turn boundaries in the audio:** Use the bootstrap chunk's timestamp
output (send with combined speaker+timestamp prompt for the bootstrap). The
`[T:N]` tags give word-level timing. For each speaker turn, the start time is
the previous word's timestamp and the end time is the last word's timestamp in
that turn. Extract audio between those sample offsets.

If timestamps are not available (speaker-only mode), fall back to splitting the
bootstrap audio evenly by speaker turn count — e.g., if Speaker 1 speaks first
and Speaker 2 speaks second, take 3s from the first third and 3s from the second
third of the bootstrap chunk. This is approximate but sufficient for reference.

#### Step 2b: Compose chunk audio with preamble

For each chunk after the bootstrap, build composite audio:

```
[ref_spk1 ~3s] [silence 0.5s] [ref_spk2 ~3s] [silence 0.5s] ... [actual chunk]
```

Since the model maxes at 2 speakers, preamble = 2 * 3.5s = 7s.
Context budget with preamble:
- Preamble: ~7s of audio = ~70 tokens + ~20 output tokens = ~90 tokens
- Target actual content per chunk: ~60s (conservative for worst-case output rate)
- Total audio per request: ~67s = ~670 tokens
- Output for 60s at ~16 tok/s (speakers+timestamps) = 960 tokens
- Total: 80 + 670 + 960 + 90 = 1800 tokens — well within 4096

So set `PLUS_SPEAKER_CHUNK_MAX_S = 60` for the actual content portion when using
preamble mode. The preamble audio is added on top.

**Implementation in `serve_plus_proxy.py`:**

Add a helper:
```python
def compose_with_preamble(
    chunk_pcm: np.ndarray,
    ref_clips: dict[str, np.ndarray],  # {"1": pcm, "2": pcm, ...}
    sr: int = 16_000,
    silence_s: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Prepend speaker reference clips to chunk audio.
    Returns (composite_pcm, preamble_duration_s)."""
    silence = np.zeros(int(silence_s * sr), dtype=np.float32)
    parts = []
    for spk_id in sorted(ref_clips.keys()):
        parts.append(ref_clips[spk_id])
        parts.append(silence)
    parts.append(chunk_pcm)
    return np.concatenate(parts), sum(len(p) for p in parts[:-1]) / sr
```

#### Step 2c: Strip preamble from output

The model's output for a preamble-augmented chunk will look like:
```
[Speaker 1]: <ref1 transcription> [Speaker 2]: <ref2 transcription> [Speaker 1]: <actual chunk text>...
```

**Stripping strategy — tag counting, not text matching:**

Count the number of reference clips (= number of distinct speakers in refs).
The preamble produces exactly one `[Speaker N]:` tag per reference clip. Strip
everything up to and including the Nth speaker tag occurrence.

```python
def strip_preamble(text: str, n_ref_speakers: int) -> str:
    """Remove the first n_ref_speakers speaker-tagged segments."""
    pattern = r"\[Speaker \d+\]:"
    matches = list(re.finditer(pattern, text))
    if len(matches) <= n_ref_speakers:
        return text  # not enough tags — return as-is (fallback)
    cut_pos = matches[n_ref_speakers].start()
    return text[cut_pos:]
```

#### Step 2d: Verify preamble effectiveness

After stripping, check that the preamble portion (the stripped text) contained
all expected speaker labels. If a reference speaker is missing from the preamble
output, log a warning — the reference clip may be too short or ambiguous.

```python
preamble_text = text[:cut_pos]
for spk_id in ref_clips:
    if f"[Speaker {spk_id}]:" not in preamble_text:
        print(f"[plus-proxy] WARNING: Speaker {spk_id} not detected in preamble for chunk {i}")
```

#### Step 2e: Timestamp fixup

When mode is `combined` (speakers + timestamps), timestamps in the chunk output
include the preamble duration. After stripping the preamble text:

1. Calculate `preamble_cs = int(preamble_duration_s * 100)`
2. Subtract `preamble_cs` from all `[T:N]` values in the remaining text
3. Then apply the existing `parse_and_unwrap_timestamps()` with the chunk's
   global `start_cs` offset

```python
def adjust_preamble_timestamps(text: str, preamble_cs: int) -> str:
    """Subtract preamble duration from all timestamp tags."""
    def replace(m):
        n = max(0, int(m.group(1)) - preamble_cs)
        return f"[T:{n}]"
    return re.sub(r"\[T:(\d+)\]", replace, text)
```

Apply this between stripping and the existing unwrap logic:
```python
stripped = strip_preamble(raw_text, n_ref_speakers)
if mode == "combined":
    stripped = adjust_preamble_timestamps(stripped, preamble_cs)
```

#### Step 2f: Speaker stitching across chunks (existing logic)

The existing `stitch_with_speakers()` should now work correctly because:
- Every chunk has seen all speakers in the preamble
- Speaker labels are anchored to the reference clips (Speaker 1 in the preamble
  is always the same physical person)
- The swap heuristic becomes unnecessary — speakers are consistently labeled

**However**, verify this assumption. If the model sometimes reorders speakers
relative to the preamble, keep the existing swap logic as a safety net. Test
with at least 3 chunks of multi-speaker audio to confirm label consistency.

If labels are consistent (expected), simplify `stitch_with_speakers()` to just
concatenate without remapping when preamble mode is active.

### Phase 3: Test infrastructure (DONE)

Test fixtures and script are already in place:

- `generate_test_audio.sh` — regenerates all WAV fixtures from source MP3s
- `test_plus_speakers.sh` — 11 tests covering:
  - 2-speaker female (similar voices — hardest case) at 30s, 90s, 150s
  - 2-speaker male (distinct voices — positive control) at 30s, 180s
  - 4-voice files (documenting model's 2-speaker limit) at 62s, 153s
  - Combined speaker+timestamp mode for both voice types
  - Proxy vs internal consistency (Jaccard word overlap ≥ 0.70)

Expected results before Phase 1: Tests 1,3 proxy fail (female chunking loses
Speaker 2). Tests 4-5 proxy fail (over threshold, no preamble). Male tests
may pass even without fixes (voices distinct enough for per-chunk detection).

Run: `./test_plus_speakers.sh` (both endpoints) or with `--proxy-only`.

## Implementation order

1. **Phase 1 first** — it's a small, safe change that fixes the immediate bug
   for audio under 2 min. Ship and test this before starting Phase 2.
2. **Phase 2a-2b** — bootstrap + preamble composition. Test that preamble chunks
   produce multi-speaker output by sending composed audio to `:18701` directly.
3. **Phase 2c-2d** — stripping + verification. Test with `test.wav` by
   artificially lowering the unchunked threshold to force chunking.
4. **Phase 2e** — timestamp fixup. Test with combined mode prompt.
5. **Phase 2f** — verify stitching. End-to-end test with long audio.
6. **Phase 3** — test updates.

## Files to modify

- `serve_plus_proxy.py` — all code changes live here

## Files NOT to modify

- `serve_plus.py` — the internal endpoint is unchanged; it already handles
  arbitrary-length audio up to the context limit
- `serve_base.py` / `serve_base_proxy.py` — base model doesn't do speaker
  attribution
