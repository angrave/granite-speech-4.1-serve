# PLUS_CHUNKING — Long-audio support for granite-plus (port 8001)

## Problem

`serve_plus.py` sends the entire waveform to the model in one shot.  Two limits
cause truncated output on long audio:

1. **`max_new_tokens=800`** (line 152 of `serve_plus.py`) — the model stops
   generating after 800 output tokens regardless of how much audio is left.
   At ~1.1 tokens/word, this caps output at ~700 words (≈ 4–5 minutes of speech).
   Measured: 10-minute audio produced 725 words, stopping mid-transcript.

2. **Audio encoder input limit** — needs investigation; the processor may silently
   truncate the waveform before encoding.  Even if `max_new_tokens` is raised,
   very long audio may be clipped at the encoder.

---

## Phase 0 — Quick fix: raise `max_new_tokens` (do first)

**File:** `serve_plus.py` line 152

```python
# Before
generated = _model.generate(**inputs, max_new_tokens=800)

# After
generated = _model.generate(**inputs, max_new_tokens=4096)
```

**Impact:** Fixes truncation for audio up to ~40 minutes (plain ASR) or ~15 minutes
(timestamps mode, which emits ~3 tokens/word).  No architectural change needed.
Zero risk of regression — `max_new_tokens` is only an upper bound; the model still
stops at EOS when it finishes.

**Should also expose via env var:**

```python
MAX_NEW_TOKENS = int(os.environ.get("PLUS_MAX_NEW_TOKENS", "4096"))
# ...
generated = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
```

---

## Phase 1 — Audio encoder investigation

Before building a full chunking proxy, confirm whether the audio encoder itself
clips long waveforms:

```python
# In _infer(), add before generate():
print(f"[plus] input audio frames: {inputs.get('audio_values', inputs.get('input_features', torch.tensor([]))).shape}")
```

Run against a 10-minute file and a 35-second file.  If the encoded shape is
identical, the encoder is capping input — chunking is required regardless of
`max_new_tokens`.

---

## Phase 2 — Chunking proxy: `serve_plus_proxy.py`

Same architecture as `serve_base.py`:

```
client → :8001  serve_plus_proxy.py  (FastAPI)
                    ↓  chunks ≤ 14 s, sequential
                :18001  serve_plus (internal)
```

Plain-ASR chunking is identical to the base proxy (reuse `find_split_sample`,
`chunk_audio`, `pcm_to_wav_bytes` from `serve_base.py` or factor into
`audio_utils.py`).

The complications are in the output post-processing.

---

## Phase 3 — Timestamp stitching

### The problem

The model emits timestamps as `[T:N]` where N is end-time in **centiseconds mod
1000** (model design — it wraps at 10 seconds).  Chunk N starts at offset
`chunk_start_cs` centiseconds.  The absolute centisecond time of a token is:

```
absolute_cs = chunk_start_cs + chunk_relative_cs
```

where `chunk_relative_cs` must be **unwrapped** (it can exceed 1000 within one
chunk for audio > 10 s).

### Unwrapping within a chunk

```python
import re

def parse_and_unwrap_timestamps(text: str, chunk_start_cs: int) -> str:
    """
    Replace [T:N] tags with globally-unwrapped centisecond values.
    N wraps at 1000 within each model output; we track rollovers and add
    chunk_start_cs to produce a monotonically increasing global timeline.
    """
    prev_n = 0
    rollover = 0

    def replace(m):
        nonlocal prev_n, rollover
        n = int(m.group(1))
        if n < prev_n - 500:   # crossed 1000 boundary (hysteresis = 500)
            rollover += 1000
        prev_n = n
        absolute = chunk_start_cs + rollover + n
        return f"[T:{absolute}]"

    return re.sub(r"\[T:(\d+)\]", replace, text)
```

### Computing `chunk_start_cs`

Track the cumulative sample offset before each chunk and convert:

```python
chunk_start_cs = (pos_samples // sr) * 100   # samples → seconds → centiseconds
```

### Stitch

```python
def stitch_with_timestamps(chunk_results: list[tuple[str, int]]) -> str:
    """chunk_results: list of (text, chunk_start_cs)"""
    parts = []
    for text, start_cs in chunk_results:
        unwrapped = parse_and_unwrap_timestamps(text, start_cs)
        if unwrapped.strip():
            parts.append(unwrapped.strip())
    return " ".join(parts)
```

---

## Phase 4 — Speaker ID stitching

### The problem

Each chunk independently assigns `[Speaker 1]:` and `[Speaker 2]:`.  The first
speaker in chunk N+1 may be labelled differently from the last speaker in chunk N,
causing phantom speaker flips at every chunk boundary.

### Strategy

At the chunk boundary, use a short **overlap window** (0.5–1 s of audio) fed to
both chunk N and chunk N+1.  The last word(s) of the overlap should appear in both
transcriptions and be attributed to the same physical speaker.

Compare the speaker label on the first word of chunk N+1's output with the speaker
label on the last word of chunk N's output.  If they differ, remap all `[Speaker X]`
and `[Speaker Y]` labels in chunk N+1 by swapping 1↔2.

```python
import re

def last_speaker(text: str) -> str | None:
    """Return the last [Speaker N]: label seen in text, or None."""
    tags = re.findall(r"\[Speaker (\d+)\]:", text)
    return tags[-1] if tags else None

def first_speaker(text: str) -> str | None:
    """Return the first [Speaker N]: label seen in text, or None."""
    m = re.search(r"\[Speaker (\d+)\]:", text)
    return m.group(1) if m else None

def remap_speakers(text: str, swap: bool) -> str:
    """If swap=True, exchange all [Speaker 1] and [Speaker 2] tags."""
    if not swap:
        return text
    # Use a placeholder to avoid double-replacement
    text = text.replace("[Speaker 1]:", "__SPK_A__:")
    text = text.replace("[Speaker 2]:", "[Speaker 1]:")
    text = text.replace("__SPK_A__:", "[Speaker 2]:")
    return text

def stitch_with_speakers(chunk_texts: list[str]) -> str:
    if not chunk_texts:
        return ""
    result = [chunk_texts[0].strip()]
    for i in range(1, len(chunk_texts)):
        prev_spk = last_speaker(result[-1])
        curr_spk = first_speaker(chunk_texts[i])
        swap = (prev_spk is not None and curr_spk is not None
                and prev_spk != curr_spk
                and _same_physical_speaker(result[-1], chunk_texts[i]))
        result.append(remap_speakers(chunk_texts[i].strip(), swap))
    return " ".join(result)
```

`_same_physical_speaker` is the hard part: use the overlap audio to check whether
the first words of chunk N+1 duplicate words at the end of chunk N (from the
overlapping audio).  If the duplicated segment has different speaker labels in the
two chunks, swap the labels for the entire chunk N+1.

### Edge cases

| Case | Behaviour |
|------|-----------|
| Single speaker throughout | No `[Speaker X]:` tags → no remapping needed |
| New speaker enters mid-chunk | Label is correct within chunk; boundary check still applies |
| More than 2 speakers | Current design only handles 2; extend mapping table for N speakers |
| Chunk with no speaker tags | Skip remapping for that chunk |

---

## Phase 5 — Combined timestamps + speakers

When the user requests both timestamps and speaker attribution simultaneously,
apply Phase 3 (timestamp unwrap) first, then Phase 4 (speaker remap) on the
unwrapped text.  The two transforms are independent of each other.

---

## Implementation order

1. **Phase 0** — bump `max_new_tokens` to 4096 + env var.  Ship immediately.
2. **Phase 1** — add encoder shape logging.  Run 3 test cases (35 s / 2 min / 10 min).
   Confirms whether chunking is architecturally required.
3. **Phase 2** — plain-ASR chunking proxy (copy pattern from `serve_base.py`).
   Start llama (internal) on :18001, proxy on :8001.
4. **Phase 3** — add timestamp unwrapping to the proxy.  Test with 2-min audio.
5. **Phase 4** — add speaker stitching.  Test with known 2-speaker audio.
6. **Phase 5** — combined mode.  Regression-test all prompt modes.

---

## Test cases needed

```bash
# Phase 0 smoke test (after bumping max_new_tokens)
bash test_long_duration.sh 5   # 5 min, check word count scales linearly

# Phase 3 timestamp test
# Generate 2-min audio, request timestamps, verify [T:N] values increase
# monotonically across chunk boundaries and match expected offsets

# Phase 4 speaker test
# Use a known 2-speaker recording > 14 s; verify Speaker 1 and Speaker 2
# remain consistent across the full transcript

# Regression: existing test_endpoints.sh must still pass unchanged
bash test_endpoints.sh
```
