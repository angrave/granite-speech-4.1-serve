# CUTNSTICH — Production-grade chunking proxy for granite-base (port 9797)

## Problem

`llama-server` with the `granite_speech` projector silently returns `"text": ""`
for audio longer than ~17 s (see `llama-BUG.md`).  The fix is a thin Python proxy
that sits in front of llama-server, splits long audio at word boundaries, forwards
each chunk, and stitches the results.

---

## Architecture

```
Before:
  client → :9797  llama-server (crashes silently on audio > ~17 s)

After:
  client → :9797  serve_base.py (FastAPI proxy)
                      ↓  chunks ≤14 s each, sequential
                  :19797  llama-server  (internal, loopback only)
```

`serve_base.py` is intentionally modelled on `serve_plus.py`: same auth pattern,
same health endpoint, same JSON response shape.  No external interface changes.

---

## TODO checklist

### Phase 1 — Core (required)

- [ ] **P1-1** Create `serve_base.py`
  - [ ] Auth (same `LLAMA_API_KEY` as current llama-server)
  - [ ] `/health` endpoint — forward to internal llama-server
  - [ ] `load_audio` — soundfile decode + resample to 16 kHz mono → numpy float32
  - [ ] `find_split_sample` — longest-gap splitter (see §Chunking)
  - [ ] `chunk_audio` — loop that calls `find_split_sample` until audio is consumed
  - [ ] `pcm_to_wav_bytes` — write numpy slice back to in-memory WAV
  - [ ] `post_chunk` — async httpx POST with random backoff on 429/503
  - [ ] `stitch` — join chunk texts with a single space, strip edges
  - [ ] `/v1/audio/transcriptions` endpoint — orchestrates the above
  - [ ] Return aggregated `usage` counts (sum input/output tokens across chunks)

- [ ] **P1-2** Update `start_apple_dockerless.sh`
  - [ ] Move llama-server from `--port 9797` to `--port 19797`
  - [ ] Add `uvicorn serve_base:app --port 9797 --host 127.0.0.1` after llama-server

- [ ] **P1-3** Update `README.md`
  - [ ] Note that port 9797 is now the proxy; llama-server is on 19797 (internal)
  - [ ] Add the ~17 s limitation reference and chunking note

- [ ] **P1-4** Update `test_endpoints.sh` base-endpoint assertion
  - [ ] Currently asserts non-empty; will now pass for `test.wav` (35 s)

### Phase 2 — Boundary quality (recommended for production)

- [ ] **P2-1** Overlap stitching: feed last 0.5 s of chunk N as prefix of chunk N+1,
  then strip the duplicated words from the head of N+1's text (see §Stitching)

- [ ] **P2-2** Retry telemetry: log chunk index, audio duration, attempt number, and
  backoff delay at INFO level so slow llama-server slots are visible in `base.log`

- [ ] **P2-3** Hard timeout per chunk (e.g. 90 s) distinct from the backoff retry loop

- [ ] **P2-4** Config via env vars: `LLAMA_CHUNK_MAX_S`, `LLAMA_CHUNK_MIN_SPLIT_S`,
  `LLAMA_BACKOFF_MIN_S`, `LLAMA_BACKOFF_MAX_S`, `LLAMA_MAX_RETRIES`

---

## §Chunking — finding word boundaries

### Concept

```
|←———— chunk budget: 14 s ————→|
|←— 5 s minimum ———→|
                    |←— search window —→|
0s         5s                         14s
           ^                           ^
           search starts here          hard cut if no gap found
```

For each chunk window, compute per-frame RMS energy (20 ms frames) over [5 s, 14 s].
Find the **longest contiguous run of quiet frames** in that window.  Split at the
**midpoint** of that run.  If no run is found, hard-cut at 14 s.

### Why longest gap rather than first/last gap

- **First gap** after 5 s → may split at a single quiet frame between syllables.
- **Last gap** before 14 s → maximises chunk length but risks cutting a word if
  the final seconds are dense speech.
- **Longest gap** → most silence context around the cut; cleanest word boundary.
  The 5 s minimum prevents degenerate short chunks even if the opening is silent.

### Energy threshold — adaptive not absolute

Different audio sources (phone, studio, laptop mic) have wildly different noise
floors.  Using a fixed threshold (e.g. RMS < 0.01) will miss silences in noisy
recordings or falsely trigger in quiet ones.

**Recipe:** within the search window, take the 10th-percentile frame energy and
multiply by 2.0.  This tracks the noise floor of the specific recording.

```python
thresh = np.percentile(rms[lo_frame:hi_frame], 10) * 2.0
```

If the 10th-percentile × 2 still produces no qualifying frames (very dense speech),
the loop falls back to the single quietest frame in the window.

### `find_split_sample` — full implementation

```python
import numpy as np

def find_split_sample(
    pcm: np.ndarray,       # float32, mono, 16 kHz
    sr: int = 16_000,
    min_s: float = 5.0,    # don't split before this
    max_s: float = 14.0,   # hard cut if no gap found
    frame_ms: int = 20,    # energy frame duration
) -> int:
    """Return the sample index of the best split point."""
    frame = int(sr * frame_ms / 1_000)
    lo = int(min_s * sr) // frame
    hi = min(int(max_s * sr) // frame, len(pcm) // frame - 1)

    if lo >= hi:                              # audio shorter than min_s
        return len(pcm)

    # RMS energy per frame
    n_frames = len(pcm) // frame
    rms = np.array([
        np.sqrt(np.mean(pcm[i * frame:(i + 1) * frame] ** 2))
        for i in range(n_frames)
    ])

    window_rms = rms[lo:hi + 1]
    thresh = np.percentile(window_rms, 10) * 2.0

    # Find longest contiguous quiet run in [lo, hi]
    best_start, best_len = None, 0
    cur_start, cur_len  = None, 0
    for i in range(lo, hi + 1):
        if rms[i] <= thresh:
            if cur_start is None:
                cur_start, cur_len = i, 0
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = None, 0

    # Check trailing run
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len

    if best_start is None:
        # No quiet frames at all → fallback: quietest single frame
        best_start = int(lo + np.argmin(window_rms))
        best_len   = 1

    split_frame = best_start + best_len // 2
    return split_frame * frame
```

### `chunk_audio` — the splitting loop

```python
def chunk_audio(
    pcm: np.ndarray,
    sr: int = 16_000,
    max_s: float = 14.0,
) -> list[np.ndarray]:
    """Split pcm into chunks of at most max_s seconds at word boundaries."""
    max_samples = int(max_s * sr)
    chunks = []
    pos = 0
    while pos < len(pcm):
        remaining = pcm[pos:]
        if len(remaining) <= max_samples:
            chunks.append(remaining)
            break
        split = find_split_sample(remaining, sr, min_s=5.0, max_s=max_s)
        chunks.append(remaining[:split])
        pos += split
    return chunks
```

---

## §HTTP client — backoff on busy

llama-server returns **503** when all slots are occupied and **429** if rate-limited.
On either code, wait a random duration in [5, 30] s and retry.  After
`MAX_RETRIES` attempts, propagate a 503 to the caller.

```python
import asyncio, random, httpx

LLAMA_INTERNAL_URL = "http://127.0.0.1:19797/v1/audio/transcriptions"
LLAMA_INTERNAL_KEY = os.environ["LLAMA_API_KEY"]   # same key, loopback
MAX_RETRIES        = 5
BACKOFF_MIN        = 5.0   # seconds
BACKOFF_MAX        = 30.0

async def post_chunk(
    client:    httpx.AsyncClient,
    wav_bytes: bytes,
    model:     str,
    prompt:    str,
    chunk_idx: int,
) -> dict:
    headers = {"Authorization": f"Bearer {LLAMA_INTERNAL_KEY}"}
    data    = {"model": model, "prompt": prompt}

    for attempt in range(MAX_RETRIES + 1):
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        try:
            resp = await client.post(
                LLAMA_INTERNAL_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=90.0,
            )
        except httpx.TimeoutException:
            if attempt >= MAX_RETRIES:
                raise HTTPException(503, f"chunk {chunk_idx}: llama-server timed out")
            delay = random.uniform(BACKOFF_MIN, BACKOFF_MAX)
            print(f"[base] chunk {chunk_idx} timeout, retry {attempt+1} in {delay:.1f}s")
            await asyncio.sleep(delay)
            continue

        if resp.status_code in (429, 503):
            if attempt >= MAX_RETRIES:
                raise HTTPException(503, f"chunk {chunk_idx}: llama-server busy after {MAX_RETRIES} retries")
            delay = random.uniform(BACKOFF_MIN, BACKOFF_MAX)
            print(f"[base] chunk {chunk_idx} HTTP {resp.status_code}, retry {attempt+1} in {delay:.1f}s")
            await asyncio.sleep(delay)
            continue

        resp.raise_for_status()
        return resp.json()

    raise HTTPException(503, f"chunk {chunk_idx}: max retries exceeded")
```

---

## §Stitching

### Phase 1 — simple join (implement first)

```python
def stitch(texts: list[str]) -> str:
    return " ".join(t.strip() for t in texts if t.strip())
```

Caveat: at every chunk boundary the model sees a hard audio cut.  It will likely
end chunk N with a word and begin chunk N+1 mid-word or with a repeated word.
For most use-cases this is acceptable — the boundary artefact is at most one word.

### Phase 2 — overlap stitching (better boundary quality)

Feed the **last 0.5 s of chunk N** as a prefix of chunk N+1's audio input.  After
getting chunk N+1's text, strip words from its head that already appear at the
tail of chunk N's text.

```python
OVERLAP_S = 0.5   # seconds of audio to overlap

def chunk_audio_with_overlap(pcm, sr=16_000, max_s=14.0, overlap_s=OVERLAP_S):
    """Yield (chunk_pcm, is_first) tuples with overlap prefix on non-first chunks."""
    overlap = int(overlap_s * sr)
    chunks  = chunk_audio(pcm, sr, max_s)
    for i, c in enumerate(chunks):
        if i == 0:
            yield c, True
        else:
            prefix = chunks[i - 1][-overlap:]
            yield np.concatenate([prefix, c]), False

def strip_overlap_prefix(prev_text: str, curr_text: str) -> str:
    """
    Remove words from the head of curr_text that already appear
    at the tail of prev_text (duplicate from overlap window).
    Comparison is case-insensitive and punctuation-stripped.
    """
    import re
    clean = lambda s: re.sub(r"[^\w\s]", "", s).lower().split()
    prev_words = clean(prev_text)
    curr_words = curr_text.split()
    curr_clean = clean(curr_text)

    # Find the longest suffix of prev_words that matches a prefix of curr_clean
    for overlap_len in range(min(len(prev_words), len(curr_clean), 8), 0, -1):
        if prev_words[-overlap_len:] == curr_clean[:overlap_len]:
            return " ".join(curr_words[overlap_len:]).strip()
    return curr_text.strip()
```

---

## §FastAPI endpoint — `serve_base.py`

Full wiring of the above pieces:

```python
"""
serve_base.py — Chunking proxy for granite-speech-4.1-2b (llama-server backend).
Listens on :9797. Forwards to llama-server on :19797.
"""
import asyncio, io, os, random
import numpy as np
import soundfile as sf
import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import hmac
from contextlib import asynccontextmanager

_API_KEY           = os.environ.get("LLAMA_API_KEY", "")
LLAMA_INTERNAL_URL = "http://127.0.0.1:19797/v1/audio/transcriptions"
LLAMA_HEALTH_URL   = "http://127.0.0.1:19797/health"
MAX_CHUNK_S        = float(os.environ.get("LLAMA_CHUNK_MAX_S",       "14"))
MIN_SPLIT_S        = float(os.environ.get("LLAMA_CHUNK_MIN_SPLIT_S",  "5"))
BACKOFF_MIN        = float(os.environ.get("LLAMA_BACKOFF_MIN_S",      "5"))
BACKOFF_MAX        = float(os.environ.get("LLAMA_BACKOFF_MAX_S",     "30"))
MAX_RETRIES        = int(  os.environ.get("LLAMA_MAX_RETRIES",         "5"))

_bearer = HTTPBearer(auto_error=False)

def verify_key(creds: HTTPAuthorizationCredentials = Security(_bearer)):
    if not _API_KEY:
        return
    if creds is None or not hmac.compare_digest(creds.credentials, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

_http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient()
    print(f"[base-proxy] → llama-server at {LLAMA_INTERNAL_URL}")
    print(f"[base-proxy]   max_chunk={MAX_CHUNK_S}s  min_split={MIN_SPLIT_S}s  "
          f"backoff=[{BACKOFF_MIN},{BACKOFF_MAX}]s  max_retries={MAX_RETRIES}")
    yield
    await _http_client.aclose()

app = FastAPI(title="Granite Speech Base (proxy)", lifespan=lifespan)


def load_audio(data: bytes, target_sr: int = 16_000) -> np.ndarray:
    """Decode any soundfile-supported format → float32 mono at target_sr."""
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    pcm = audio.mean(axis=1)            # stereo → mono
    if sr != target_sr:
        # torchaudio resample; fall back to simple decimation if not available
        try:
            import torchaudio.functional as AF
            import torch
            t = torch.from_numpy(pcm).unsqueeze(0)
            t = AF.resample(t, sr, target_sr)
            pcm = t.squeeze(0).numpy()
        except ImportError:
            # integer-ratio decimation (rough, fine for speech)
            step = sr // target_sr
            pcm = pcm[::step]
    return pcm.astype(np.float32)


def pcm_to_wav_bytes(pcm: np.ndarray, sr: int = 16_000) -> bytes:
    """Encode float32 mono numpy array as 16-bit WAV in memory."""
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# find_split_sample and chunk_audio from §Chunking (paste here verbatim)


async def _post_chunk(wav_bytes, model, prompt, chunk_idx) -> dict:
    """POST one chunk to internal llama-server with backoff retry."""
    headers = {"Authorization": f"Bearer {_API_KEY}"}
    for attempt in range(MAX_RETRIES + 1):
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        try:
            resp = await _http_client.post(
                LLAMA_INTERNAL_URL,
                headers=headers, files=files,
                data={"model": model, "prompt": prompt},
                timeout=90.0,
            )
        except httpx.TimeoutException:
            if attempt >= MAX_RETRIES:
                raise HTTPException(503, f"chunk {chunk_idx}: llama-server timed out")
            delay = random.uniform(BACKOFF_MIN, BACKOFF_MAX)
            print(f"[base] chunk {chunk_idx} timeout — retry {attempt+1} in {delay:.1f}s")
            await asyncio.sleep(delay)
            continue

        if resp.status_code in (429, 503):
            if attempt >= MAX_RETRIES:
                raise HTTPException(503, f"chunk {chunk_idx}: busy after {MAX_RETRIES} retries")
            delay = random.uniform(BACKOFF_MIN, BACKOFF_MAX)
            print(f"[base] chunk {chunk_idx} HTTP {resp.status_code} — retry {attempt+1} in {delay:.1f}s")
            await asyncio.sleep(delay)
            continue

        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()

    raise HTTPException(503, f"chunk {chunk_idx}: max retries exceeded")  # unreachable


@app.get("/health")
async def health():
    try:
        r = await _http_client.get(LLAMA_HEALTH_URL, timeout=5.0)
        return r.json()
    except Exception:
        raise HTTPException(503, "llama-server unreachable")


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_key)])
async def transcribe(
    file:   UploadFile = File(...),
    model:  str        = Form("ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0"),
    prompt: str        = Form("transcribe with punctuation and capitalization."),
):
    raw   = await file.read()
    pcm   = load_audio(raw)
    sr    = 16_000
    dur_s = len(pcm) / sr

    chunks = chunk_audio(pcm, sr, max_s=MAX_CHUNK_S) if dur_s > MAX_CHUNK_S \
             else [pcm]

    print(f"[base] {dur_s:.1f}s audio → {len(chunks)} chunk(s)")

    texts        = []
    total_in     = 0
    total_out    = 0
    total_cached = 0

    for i, chunk in enumerate(chunks):
        wav_bytes = pcm_to_wav_bytes(chunk, sr)
        result    = await _post_chunk(wav_bytes, model, prompt, i)
        texts.append(result.get("text", ""))
        usage = result.get("usage", {})
        total_in     += usage.get("input_tokens",  0)
        total_out    += usage.get("output_tokens", 0)
        total_cached += usage.get("input_tokens_details", {}).get("cached_tokens", 0)

    full_text = stitch(texts)

    return JSONResponse({
        "type": "transcript.text.done",
        "text": full_text,
        "usage": {
            "type": "tokens",
            "input_tokens":  total_in,
            "output_tokens": total_out,
            "total_tokens":  total_in + total_out,
            "input_tokens_details": {"cached_tokens": total_cached},
            "chunks": len(chunks),
        },
    })
```

---

## §`start_apple_dockerless.sh` changes

Two lines change.  Before:

```bash
"$LLAMA_SERVER" \
  -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 \
  --port 9797 --host 127.0.0.1 \
  --api-key "$LLAMA_API_KEY" \
  >> "$SCRIPT_DIR/base.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-base"); LOGS+=("base.log")
```

After:

```bash
"$LLAMA_SERVER" \
  -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 \
  --port 19797 --host 127.0.0.1 \
  --api-key "$LLAMA_API_KEY" \
  >> "$SCRIPT_DIR/base.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-base-llama"); LOGS+=("base.log")

uvicorn serve_base:app --port 9797 --host 127.0.0.1 \
  >> "$SCRIPT_DIR/base.log" 2>&1 &
PIDS+=($!); NAMES+=("granite-base"); LOGS+=("base.log")
```

Note: both processes append to the same `base.log`.  The uvicorn process should
start after llama-server is ready.  Add a readiness poll before starting the proxy:

```bash
echo "Waiting for llama-server on :19797..."
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:19797/health > /dev/null 2>&1 && break
  sleep 2
done
```

---

## §Edge cases and production notes

| Case | Behaviour |
|------|-----------|
| Audio ≤ 14 s | Single chunk, forwarded as-is — no chunking overhead |
| Chunk returns `text: ""` | Included as empty string; `stitch()` filters it out |
| All chunks empty | Returns `""` with HTTP 200 — caller should assert non-empty |
| llama-server crash | After `MAX_RETRIES` the proxy returns HTTP 503 to the caller |
| Very long silence (> 14 s) | `chunk_audio` splits at 14 s hard cut; no infinite loop |
| Audio < 5 s | `find_split_sample` returns `len(pcm)` → treated as single chunk |
| Concurrent requests | Each request runs its chunks **sequentially** (order matters for stitching). Multiple concurrent requests to serve_base.py will pipeline into llama-server's 4 parallel slots. |
| Prompt forwarding | The `prompt` field is forwarded unchanged to every chunk. The base model does not support timestamps (that is plus-only on port 8001); no special handling needed. |

---

## §Validation — update `test_endpoints.sh`

After implementation, the base test should pass for the full 35 s `test.wav`:

```bash
echo "=== Base (llama.cpp proxy, port 9797) ==="
raw=$(curl -s http://127.0.0.1:9797/v1/audio/transcriptions \
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
```

---

## Implementation order

1. Write `serve_base.py` with Phase 1 stitching only.
2. Update and test `start_apple_dockerless.sh`.
3. Run `test_endpoints.sh` — base endpoint should now pass for `test.wav`.
4. Run `llama-base-kvcheck.sh` — TEST 1 and TEST 3 should pass; TEST 2 (35 s) now
   also passes via chunking.
5. Ship Phase 2 (overlap stitching) as a follow-up once Phase 1 is stable in production.

---

## Capability boundary

| Feature | Base (:9797) | Plus (:8001) | NAR (:8002) |
|---------|-------------|-------------|------------|
| Punctuation / capitalisation | ✅ | ❌ | ❌ |
| Word-level timestamps | ❌ | ✅ | ❌ |
| Speaker attribution | ❌ | ✅ | ❌ |
| Long audio (> 17 s) | ✅ via proxy | ✅ native | ✅ native |

The chunking proxy adds long-audio support to base; it does not add timestamp or
speaker features — those remain plus-only and are unaffected by this work.
