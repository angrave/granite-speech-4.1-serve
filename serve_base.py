"""
serve_base.py — Chunking proxy for granite-speech-4.1-2b (llama-server backend).
Listens on :8700. Forwards to llama-server on :18700.
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
LLAMA_INTERNAL_URL = os.environ.get("LLAMA_INTERNAL_URL", "http://127.0.0.1:18700/v1/audio/transcriptions")
LLAMA_HEALTH_URL   = os.environ.get("LLAMA_HEALTH_URL",   "http://127.0.0.1:18700/health")
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
        split = find_split_sample(remaining, sr, min_s=MIN_SPLIT_S, max_s=max_s)
        chunks.append(remaining[:split])
        pos += split
    return chunks


def stitch(texts: list[str]) -> str:
    return " ".join(t.strip() for t in texts if t.strip())


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
        data = r.json()
    except Exception:
        raise HTTPException(503, "llama-server unreachable")
    data.setdefault("model", "ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0")
    data.setdefault("auth", "enabled" if _API_KEY else "disabled")
    return data


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_key)])
async def transcribe(
    file:   UploadFile = File(...),
    model:  str        = Form("ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0"),
    prompt: str        = Form("transcribe with punctuation and capitalization."),
):
    raw = await file.read()

    try:
        r = await _http_client.get(LLAMA_HEALTH_URL, timeout=3.0)
        if r.status_code != 200:
            raise HTTPException(503, "llama-server unavailable or busy")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, "llama-server unreachable")

    pcm = load_audio(raw)
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
