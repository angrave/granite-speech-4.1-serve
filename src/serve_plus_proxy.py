"""
serve_plus_proxy.py — Chunking proxy for granite-speech-4.1-2b-plus.
Listens on :8701 (external). Forwards chunks to serve_plus on :18701 (internal).

Architecture:
  client → :8701  serve_plus_proxy.py  (FastAPI, this file)
                      ↓  chunks ≤ 14 s, sequential
                  :18701  serve_plus.py (internal, loads model)

Prompt modes (pass via the `prompt` form field — same strings as serve_plus.py):
  Plain ASR:   chunks stitched with a space.
  Timestamps:  [T:N] values are unwrapped across chunk boundaries so that
               centisecond offsets are globally monotone.
  Speakers:    [Speaker N]: labels are remapped at chunk boundaries to
               preserve consistent speaker identity.
  Combined:    timestamp unwrapping applied first, then speaker remapping.

Speaker-aware chunking:
  For audio ≤ PLUS_SPEAKER_MAX_UNCHUNKED_S (default 120s) in speakers/combined
  mode, the full audio is sent as a single request to avoid per-chunk speaker
  label drift.

  For audio > PLUS_SPEAKER_MAX_UNCHUNKED_S in speakers/combined mode, a
  preamble of short speaker reference clips (~3s each) is prepended to every
  chunk so the model assigns consistent labels across chunks.

Env vars:
  GRANITE_API_KEY              — shared with serve_plus; Bearer auth.
  PLUS_INTERNAL_URL            — URL of internal serve_plus (default :18701).
  PLUS_INTERNAL_HEALTH_URL     — health endpoint of internal serve_plus.
  PLUS_CHUNK_MAX_S             — max chunk length in seconds (default 14).
  PLUS_CHUNK_MIN_SPLIT_S       — min audio before looking for a split (default 5).
  PLUS_BACKOFF_MIN_S           — retry back-off floor in seconds (default 5).
  PLUS_BACKOFF_MAX_S           — retry back-off ceiling in seconds (default 30).
  PLUS_MAX_RETRIES             — max retries per chunk (default 5).
  PLUS_SPEAKER_MAX_UNCHUNKED_S — max duration sent unchunked in speaker mode
                                  (default 120).
  PLUS_SPEAKER_CHUNK_MAX_S     — chunk size for preamble mode (default 60).
"""
import asyncio
import hmac
import io
import os
import random
import re

import httpx
import numpy as np
import soundfile as sf
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_API_KEY                     = os.environ.get("GRANITE_API_KEY", "")
PLUS_INTERNAL_URL            = os.environ.get("PLUS_INTERNAL_URL",        "http://127.0.0.1:18701/v1/audio/transcriptions")
PLUS_HEALTH_URL              = os.environ.get("PLUS_INTERNAL_HEALTH_URL", "http://127.0.0.1:18701/health")
MAX_CHUNK_S                  = float(os.environ.get("PLUS_CHUNK_MAX_S",              "14"))
MIN_SPLIT_S                  = float(os.environ.get("PLUS_CHUNK_MIN_SPLIT_S",         "5"))
BACKOFF_MIN                  = float(os.environ.get("PLUS_BACKOFF_MIN_S",             "5"))
BACKOFF_MAX                  = float(os.environ.get("PLUS_BACKOFF_MAX_S",            "30"))
MAX_RETRIES                  = int(  os.environ.get("PLUS_MAX_RETRIES",               "5"))
PLUS_SPEAKER_MAX_UNCHUNKED_S = float(os.environ.get("PLUS_SPEAKER_MAX_UNCHUNKED_S", "120"))
PLUS_SPEAKER_CHUNK_MAX_S     = float(os.environ.get("PLUS_SPEAKER_CHUNK_MAX_S",      "60"))
DEFAULT_PROMPT               = "<|audio|> can you transcribe the speech into a written format?"

# Bootstrap prompt: always combined (speaker + timestamp) so we can locate
# speaker turn boundaries in the audio for reference clip extraction.
_BOOTSTRAP_PROMPT = (
    "<|audio|> Timestamps and Speaker attribution: Transcribe the speech. "
    "After each word, add a timestamp tag showing the end time in centiseconds, "
    "e.g. hello [T:45] world [T:82]. "
    "Denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
)

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
    print(f"[plus-proxy] → plus-model at {PLUS_INTERNAL_URL}")
    print(f"[plus-proxy]   max_chunk={MAX_CHUNK_S}s  min_split={MIN_SPLIT_S}s  "
          f"backoff=[{BACKOFF_MIN},{BACKOFF_MAX}]s  max_retries={MAX_RETRIES}")
    print(f"[plus-proxy]   speaker_max_unchunked={PLUS_SPEAKER_MAX_UNCHUNKED_S}s  "
          f"speaker_chunk_max={PLUS_SPEAKER_CHUNK_MAX_S}s")
    yield
    await _http_client.aclose()


app = FastAPI(title="Granite Speech Plus (proxy)", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Audio utilities (mirrors serve_base.py)
# ---------------------------------------------------------------------------

def load_audio(data: bytes, target_sr: int = 16_000) -> np.ndarray:
    """Decode audio bytes → float32 mono at target_sr.

    Tries libsndfile (WAV, FLAC, OGG, …) first; falls back to torchaudio's
    ffmpeg backend for MP3, MP4/AAC, and any other ffmpeg-supported format.
    """
    buf = io.BytesIO(data)
    try:
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
        pcm = audio.mean(axis=1)
    except Exception:
        import torchaudio, torch
        buf.seek(0)
        t, sr = torchaudio.load(buf)          # ffmpeg backend handles MP3, MP4, AAC…
        pcm = t.mean(dim=0).numpy()           # (channels, samples) → mono
    if sr != target_sr:
        try:
            import torchaudio.functional as AF
            import torch
            t = torch.from_numpy(pcm).unsqueeze(0)
            t = AF.resample(t, sr, target_sr)
            pcm = t.squeeze(0).numpy()
        except ImportError:
            step = sr // target_sr
            pcm = pcm[::step]
    return pcm.astype(np.float32)


def pcm_to_wav_bytes(pcm: np.ndarray, sr: int = 16_000) -> bytes:
    """Encode float32 mono numpy array as 16-bit WAV in memory."""
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def find_split_sample(
    pcm: np.ndarray,
    sr: int = 16_000,
    min_s: float = 5.0,
    max_s: float = 14.0,
    frame_ms: int = 20,
) -> int:
    """Return the sample index of the best split point (silence-aware)."""
    frame = int(sr * frame_ms / 1_000)
    lo = int(min_s * sr) // frame
    hi = min(int(max_s * sr) // frame, len(pcm) // frame - 1)

    if lo >= hi:
        return len(pcm)

    n_frames = len(pcm) // frame
    rms = np.array([
        np.sqrt(np.mean(pcm[i * frame:(i + 1) * frame] ** 2))
        for i in range(n_frames)
    ])

    window_rms = rms[lo:hi + 1]
    thresh = np.percentile(window_rms, 10) * 2.0

    best_start, best_len = None, 0
    cur_start, cur_len = None, 0
    for i in range(lo, hi + 1):
        if rms[i] <= thresh:
            if cur_start is None:
                cur_start, cur_len = i, 0
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_start, cur_len = None, 0

    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len

    if best_start is None:
        best_start = int(lo + np.argmin(window_rms))
        best_len = 1

    split_frame = best_start + best_len // 2
    return split_frame * frame


def chunk_audio(
    pcm: np.ndarray,
    sr: int = 16_000,
    max_s: float = 14.0,
) -> list[tuple[np.ndarray, int]]:
    """Split pcm into chunks of at most max_s seconds at word boundaries.

    Returns a list of (chunk_pcm, start_sample) tuples so callers can compute
    the chunk's start offset in any unit they need.
    """
    max_samples = int(max_s * sr)
    chunks: list[tuple[np.ndarray, int]] = []
    pos = 0
    while pos < len(pcm):
        remaining = pcm[pos:]
        if len(remaining) <= max_samples:
            chunks.append((remaining, pos))
            break
        split = find_split_sample(remaining, sr, min_s=MIN_SPLIT_S, max_s=max_s)
        chunks.append((remaining[:split], pos))
        pos += split
    return chunks


# ---------------------------------------------------------------------------
# Timestamp stitching
# ---------------------------------------------------------------------------

def parse_and_unwrap_timestamps(text: str, chunk_start_cs: int) -> str:
    """Replace [T:N] tags with globally-unwrapped centisecond values.

    N wraps at 1000 within each model output (centiseconds mod 1000 per model
    design).  We track rollovers within the chunk and add chunk_start_cs to
    produce a monotonically increasing global timeline across chunks.
    """
    prev_n = 0
    rollover = 0

    def replace(m: re.Match) -> str:
        nonlocal prev_n, rollover
        n = int(m.group(1))
        if n < prev_n - 500:   # crossed the 1000-boundary (hysteresis = 500)
            rollover += 1000
        prev_n = n
        absolute = chunk_start_cs + rollover + n
        return f"[T:{absolute}]"

    return re.sub(r"\[T:(\d+)\]", replace, text)


def stitch_with_timestamps(chunk_results: list[tuple[str, int]]) -> str:
    """Unwrap and stitch timestamped chunk outputs.

    Args:
        chunk_results: list of (text, chunk_start_cs) pairs.
    """
    parts = []
    for text, start_cs in chunk_results:
        unwrapped = parse_and_unwrap_timestamps(text, start_cs)
        if unwrapped.strip():
            parts.append(unwrapped.strip())
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Speaker ID stitching
# ---------------------------------------------------------------------------

def last_speaker(text: str) -> str | None:
    """Return the last [Speaker N]: label seen in text, or None."""
    tags = re.findall(r"\[Speaker (\d+)\]:", text)
    return tags[-1] if tags else None


def first_speaker(text: str) -> str | None:
    """Return the first [Speaker N]: label seen in text, or None."""
    m = re.search(r"\[Speaker (\d+)\]:", text)
    return m.group(1) if m else None


def remap_speakers(text: str, swap: bool) -> str:
    """If swap=True, exchange all [Speaker 1]: and [Speaker 2]: tags."""
    if not swap:
        return text
    # Use a placeholder to avoid double-replacement
    text = text.replace("[Speaker 1]:", "__SPK_A__:")
    text = text.replace("[Speaker 2]:", "[Speaker 1]:")
    text = text.replace("__SPK_A__:", "[Speaker 2]:")
    return text


def stitch_with_speakers(chunk_texts: list[str]) -> str:
    """Stitch speaker-attributed chunks, remapping labels at boundaries.

    Strategy: at each boundary, if the first speaker of chunk N+1 differs from
    the last speaker of chunk N, swap all speaker labels in chunk N+1.  This
    assumes the same physical person is speaking at the chunk boundary, which
    holds when chunks overlap (or when speech is continuous) — a conservative
    heuristic that eliminates phantom speaker flips.

    Note: full overlap-audio verification (feeding 0.5-1 s of audio to both
    adjacent chunks and comparing the duplicated segment) is a future
    enhancement tracked in PLUS_CHUNKING.md Phase 4.
    """
    if not chunk_texts:
        return ""
    result = [chunk_texts[0].strip()]
    for i in range(1, len(chunk_texts)):
        prev_spk = last_speaker(result[-1])
        curr_spk = first_speaker(chunk_texts[i])
        swap = (prev_spk is not None and curr_spk is not None and prev_spk != curr_spk)
        result.append(remap_speakers(chunk_texts[i].strip(), swap))
    return " ".join(result)


# ---------------------------------------------------------------------------
# Combined: timestamps + speakers
# ---------------------------------------------------------------------------

def stitch_combined(chunk_results: list[tuple[str, int]]) -> str:
    """Apply timestamp unwrapping first, then speaker stitching."""
    unwrapped_texts = [
        parse_and_unwrap_timestamps(text, start_cs)
        for text, start_cs in chunk_results
    ]
    return stitch_with_speakers(unwrapped_texts)


# ---------------------------------------------------------------------------
# Mode detection & plain stitch
# ---------------------------------------------------------------------------

def detect_mode(prompt: str) -> str:
    """Return 'combined', 'timestamps', 'speakers', or 'plain'."""
    has_ts  = "[T:" in prompt
    has_spk = "[Speaker" in prompt
    if has_ts and has_spk:
        return "combined"
    if has_ts:
        return "timestamps"
    if has_spk:
        return "speakers"
    return "plain"


def stitch_plain(texts: list[str]) -> str:
    return " ".join(t.strip() for t in texts if t.strip())


# ---------------------------------------------------------------------------
# Speaker preamble helpers (Phase 2)
# ---------------------------------------------------------------------------

def parse_speaker_turns(text: str) -> list[tuple[str, int, int]]:
    """Parse bootstrap text into (speaker_id, start_cs, end_cs) turns.

    Uses [Speaker N]: tags and [T:X] timestamps to find turn boundaries.
    """
    turns: list[tuple[str, int, int]] = []
    current_speaker: str | None = None
    turn_start_cs = 0
    last_cs = 0

    for m in re.finditer(r'\[Speaker (\d+)\]:|(?:\[T:(\d+)\])', text):
        spk = m.group(1)
        ts  = m.group(2)
        if spk is not None:
            if current_speaker is not None:
                turns.append((current_speaker, turn_start_cs, last_cs))
            current_speaker = spk
            turn_start_cs = last_cs
        elif ts is not None:
            last_cs = int(ts)

    if current_speaker is not None:
        turns.append((current_speaker, turn_start_cs, last_cs))

    return turns


def extract_ref_clips(
    pcm: np.ndarray,
    sr: int,
    turns: list[tuple[str, int, int]],
    ref_dur_s: float = 3.0,
) -> dict[str, np.ndarray]:
    """Extract ~ref_dur_s audio clips from the middle of each speaker's longest turn."""
    by_speaker: dict[str, list[tuple[int, int]]] = {}
    for spk_id, start_cs, end_cs in turns:
        by_speaker.setdefault(spk_id, []).append((start_cs, end_cs))

    ref_clips: dict[str, np.ndarray] = {}
    ref_samples = int(ref_dur_s * sr)

    for spk_id, spk_turns in by_speaker.items():
        longest = max(spk_turns, key=lambda t: t[1] - t[0])
        start_cs, end_cs = longest
        dur_cs = end_cs - start_cs
        if dur_cs <= 0:
            continue
        mid_cs = start_cs + dur_cs // 2
        clip_start_cs = max(start_cs, mid_cs - 150)      # 1.5 s before mid
        clip_end_cs   = min(end_cs, clip_start_cs + 300)  # 3 s total
        s = int(clip_start_cs * sr / 100)
        e = min(len(pcm), int(clip_end_cs * sr / 100))
        e = min(e, s + ref_samples)
        if e > s:
            ref_clips[spk_id] = pcm[s:e]

    return ref_clips


def extract_ref_clips_no_ts(
    pcm: np.ndarray,
    sr: int,
    speaker_order: list[str],
    ref_dur_s: float = 3.0,
) -> dict[str, np.ndarray]:
    """Fallback: split audio evenly, extract a center clip per speaker."""
    if not speaker_order:
        return {}
    ref_clips: dict[str, np.ndarray] = {}
    ref_samples = int(ref_dur_s * sr)
    n = len(speaker_order)
    seg = len(pcm) // n
    for i, spk_id in enumerate(speaker_order):
        seg_start = i * seg
        mid = seg_start + seg // 2
        clip_start = max(0, mid - ref_samples // 2)
        clip_end = min(len(pcm), clip_start + ref_samples)
        ref_clips[spk_id] = pcm[clip_start:clip_end]
    return ref_clips


def compose_with_preamble(
    chunk_pcm: np.ndarray,
    ref_clips: dict[str, np.ndarray],
    sr: int = 16_000,
    silence_s: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Prepend speaker reference clips to chunk audio.

    Returns (composite_pcm, preamble_duration_s).
    """
    silence = np.zeros(int(silence_s * sr), dtype=np.float32)
    parts: list[np.ndarray] = []
    for spk_id in sorted(ref_clips.keys()):
        parts.append(ref_clips[spk_id])
        parts.append(silence)
    preamble_dur_s = sum(len(p) for p in parts) / sr
    parts.append(chunk_pcm)
    return np.concatenate(parts), preamble_dur_s


def strip_preamble(text: str, n_ref_speakers: int) -> str:
    """Remove the first n_ref_speakers speaker-tagged segments."""
    matches = list(re.finditer(r"\[Speaker \d+\]:", text))
    if len(matches) <= n_ref_speakers:
        return text  # not enough tags — return as-is (fallback)
    cut_pos = matches[n_ref_speakers].start()
    return text[cut_pos:]


def adjust_preamble_timestamps(text: str, preamble_cs: int) -> str:
    """Subtract preamble duration from all timestamp tags."""
    def replace(m: re.Match) -> str:
        n = max(0, int(m.group(1)) - preamble_cs)
        return f"[T:{n}]"
    return re.sub(r"\[T:(\d+)\]", replace, text)


async def bootstrap_speaker_refs(
    pcm: np.ndarray,
    sr: int,
    model: str,
) -> dict[str, np.ndarray]:
    """Run bootstrap to extract per-speaker reference audio clips.

    Tries progressively longer bootstrap segments until ≥2 speakers are found
    or PLUS_SPEAKER_MAX_UNCHUNKED_S is reached.  Returns a dict mapping
    speaker_id (str) to a ~3 s PCM clip, or empty dict if bootstrap fails.
    """
    boot_durations: list[float] = []
    seen: set[float] = set()
    for d in [MAX_CHUNK_S, 20.0, 28.0, 60.0, PLUS_SPEAKER_MAX_UNCHUNKED_S]:
        if d not in seen:
            boot_durations.append(d)
            seen.add(d)

    for boot_dur_s in boot_durations:
        boot_samples = min(int(boot_dur_s * sr), len(pcm))
        boot_pcm = pcm[:boot_samples]
        boot_wav = pcm_to_wav_bytes(boot_pcm, sr)

        print(f"[plus-proxy] bootstrap: sending {len(boot_pcm)/sr:.1f}s with combined prompt")
        result    = await _post_chunk(boot_wav, model, _BOOTSTRAP_PROMPT, -1)
        boot_text = result.get("text", "")

        # Preserve speaker order of first appearance
        speakers: list[str] = list(dict.fromkeys(re.findall(r'\[Speaker (\d+)\]:', boot_text)))
        print(f"[plus-proxy] bootstrap: found {len(speakers)} speaker(s): {speakers}")

        if len(speakers) < 2:
            continue

        has_ts = bool(re.search(r'\[T:\d+\]', boot_text))
        if has_ts:
            turns = parse_speaker_turns(boot_text)
            ref_clips = extract_ref_clips(boot_pcm, sr, turns)
        else:
            ref_clips = extract_ref_clips_no_ts(boot_pcm, sr, speakers)

        if len(ref_clips) >= 2:
            print(f"[plus-proxy] bootstrap: extracted {len(ref_clips)} reference clips "
                  f"({'timestamps' if has_ts else 'even-split'})")
            return ref_clips

    print("[plus-proxy] bootstrap: could not find >=2 speakers — preamble disabled")
    return {}


# ---------------------------------------------------------------------------
# HTTP forwarding to internal serve_plus
# ---------------------------------------------------------------------------

async def _post_chunk(wav_bytes: bytes, model: str, prompt: str, chunk_idx: int) -> dict:
    """POST one chunk to internal serve_plus with exponential back-off retry."""
    headers = {"Authorization": f"Bearer {_API_KEY}"} if _API_KEY else {}
    for attempt in range(MAX_RETRIES + 1):
        files = {"file": ("chunk.wav", wav_bytes, "audio/wav")}
        try:
            resp = await _http_client.post(
                PLUS_INTERNAL_URL,
                headers=headers,
                files=files,
                data={"model": model, "prompt": prompt},
                timeout=300.0,
            )
        except httpx.TimeoutException:
            if attempt >= MAX_RETRIES:
                raise HTTPException(503, f"chunk {chunk_idx}: plus server timed out")
            delay = random.uniform(BACKOFF_MIN, BACKOFF_MAX)
            print(f"[plus-proxy] chunk {chunk_idx} timeout — retry {attempt+1} in {delay:.1f}s")
            await asyncio.sleep(delay)
            continue

        if resp.status_code in (429, 503):
            if attempt >= MAX_RETRIES:
                raise HTTPException(503, f"chunk {chunk_idx}: busy after {MAX_RETRIES} retries")
            delay = random.uniform(BACKOFF_MIN, BACKOFF_MAX)
            print(f"[plus-proxy] chunk {chunk_idx} HTTP {resp.status_code} — retry {attempt+1} in {delay:.1f}s")
            await asyncio.sleep(delay)
            continue

        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, resp.text)
        return resp.json()

    raise HTTPException(503, f"chunk {chunk_idx}: max retries exceeded")  # unreachable


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        r = await _http_client.get(PLUS_HEALTH_URL, timeout=5.0)
        return r.json()
    except Exception:
        raise HTTPException(503, "plus server unreachable")


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_key)])
async def transcribe(
    file:   UploadFile = File(...),
    model:  str        = Form("plus"),
    prompt: str        = Form(DEFAULT_PROMPT),
):
    raw = await file.read()

    try:
        r = await _http_client.get(PLUS_HEALTH_URL, timeout=3.0)
        if r.status_code != 200:
            raise HTTPException(503, "plus backend unavailable or busy")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "plus backend unreachable")

    pcm = load_audio(raw)
    sr    = 16_000
    dur_s = len(pcm) / sr
    mode  = detect_mode(prompt)

    # Phase 1: speaker/combined — skip chunking for short audio so each full
    # recording has enough context for the model to identify multiple speakers.
    if mode in ("speakers", "combined") and dur_s <= PLUS_SPEAKER_MAX_UNCHUNKED_S:
        chunks = [(pcm, 0)]
        preamble_mode = False
        print(f"[plus-proxy] {dur_s:.1f}s audio → 1 chunk(s), mode={mode} (full audio - speaker mode)")

    # Phase 2: speaker/combined — long audio uses smaller chunks with a
    # speaker-reference preamble prepended to each chunk.
    elif mode in ("speakers", "combined"):
        chunks = chunk_audio(pcm, sr, max_s=PLUS_SPEAKER_CHUNK_MAX_S)
        preamble_mode = True
        print(f"[plus-proxy] {dur_s:.1f}s audio → {len(chunks)} chunk(s), mode={mode} (preamble mode)")

    # Plain/timestamps: existing chunking logic unchanged.
    else:
        chunks = chunk_audio(pcm, sr, max_s=MAX_CHUNK_S) if dur_s > MAX_CHUNK_S \
                 else [(pcm, 0)]
        preamble_mode = False
        print(f"[plus-proxy] {dur_s:.1f}s audio → {len(chunks)} chunk(s), mode={mode}")

    # Phase 2a: bootstrap — extract per-speaker reference clips.
    ref_clips: dict[str, np.ndarray] = {}
    if preamble_mode:
        ref_clips = await bootstrap_speaker_refs(pcm, sr, model)
        if not ref_clips:
            print("[plus-proxy] bootstrap failed — continuing without preamble")

    chunk_results: list[tuple[str, int]] = []
    for i, (chunk_pcm, start_sample) in enumerate(chunks):
        start_cs = (start_sample // sr) * 100

        if preamble_mode and ref_clips:
            # Phase 2b: prepend speaker reference audio to chunk.
            composed_pcm, preamble_dur_s = compose_with_preamble(chunk_pcm, ref_clips, sr)
            wav_bytes = pcm_to_wav_bytes(composed_pcm, sr)
            result    = await _post_chunk(wav_bytes, model, prompt, i)
            raw_text  = result.get("text", "")

            # Phase 2c/2d: strip preamble output and verify speaker presence.
            n_ref = len(ref_clips)
            matches = list(re.finditer(r"\[Speaker \d+\]:", raw_text))
            if len(matches) > n_ref:
                cut_pos = matches[n_ref].start()
                preamble_text = raw_text[:cut_pos]
                stripped = raw_text[cut_pos:]
                # Phase 2d: verify preamble effectiveness.
                for spk_id in ref_clips:
                    if f"[Speaker {spk_id}]:" not in preamble_text:
                        print(f"[plus-proxy] WARNING: Speaker {spk_id} not detected in preamble for chunk {i}")
            else:
                stripped = raw_text
                print(f"[plus-proxy] WARNING: chunk {i} had only {len(matches)} speaker tags, "
                      f"expected >{n_ref} — preamble strip skipped")

            # Phase 2e: subtract preamble duration from timestamps.
            if mode == "combined":
                preamble_cs = int(preamble_dur_s * 100)
                stripped = adjust_preamble_timestamps(stripped, preamble_cs)

            text = stripped
            print(f"[plus-proxy] chunk {i}: start={start_cs}cs  "
                  f"len={len(chunk_pcm)/sr:.1f}s  preamble={preamble_dur_s:.1f}s  words≈{len(text.split())}")
        else:
            wav_bytes = pcm_to_wav_bytes(chunk_pcm, sr)
            result    = await _post_chunk(wav_bytes, model, prompt, i)
            text      = result.get("text", "")
            print(f"[plus-proxy] chunk {i}: start={start_cs}cs  "
                  f"len={len(chunk_pcm)/sr:.1f}s  words≈{len(text.split())}")

        chunk_results.append((text, start_cs))

    # Phase 2f: stitching — existing logic works correctly with preamble because
    # speaker labels are anchored to the reference clips across all chunks.
    if mode == "combined":
        full_text = stitch_combined(chunk_results)
    elif mode == "timestamps":
        full_text = stitch_with_timestamps(chunk_results)
    elif mode == "speakers":
        full_text = stitch_with_speakers([t for t, _ in chunk_results])
    else:
        full_text = stitch_plain([t for t, _ in chunk_results])

    return JSONResponse({
        "text": full_text,
        "usage": {"chunks": len(chunk_results)},
    })
