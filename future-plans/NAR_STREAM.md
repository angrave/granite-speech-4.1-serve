# NAR_STREAM — Streaming ASR for granite-speech-4.1-2b-nar

## Goal

Add a streaming WebSocket endpoint to `serve_nar.py` that is compatible with the
**OpenAI Realtime API transcription interface**.  Clients send audio incrementally;
the server returns transcript events as each chunk is processed.

Target endpoint:

```
GET ws://localhost:8702/v1/realtime?model=granite-nar
```

---

## OpenAI Realtime API compatibility target

The OpenAI Realtime API (docs: platform.openai.com/docs/api-reference/realtime)
uses a WebSocket that exchanges JSON event objects.  We implement the transcription
subset only (no LLM turns, no TTS).

### Client → Server events we must handle

| Event type | Purpose |
|---|---|
| `session.update` | Configure VAD sensitivity, chunk duration, input format |
| `input_audio_buffer.append` | Deliver a base64-encoded PCM audio chunk |
| `input_audio_buffer.commit` | Force-flush the current buffer as one utterance |
| `input_audio_buffer.clear` | Discard buffered audio without transcribing |

### Server → Client events we must emit

| Event type | When |
|---|---|
| `session.created` | Immediately after WebSocket handshake |
| `session.updated` | After processing `session.update` |
| `input_audio_buffer.speech_started` | VAD detects onset of speech |
| `input_audio_buffer.speech_stopped` | VAD detects end of utterance |
| `input_audio_buffer.committed` | Buffer flushed (by VAD or explicit commit) |
| `conversation.item.input_audio_transcription.delta` | Partial transcript (word-by-word, post-hoc) |
| `conversation.item.input_audio_transcription.completed` | Full transcript for one utterance |
| `error` | Any processing error |

### Audio wire format

OpenAI Realtime uses **24 kHz, 16-bit PCM, little-endian, mono**, base64-encoded
per `input_audio_buffer.append` event.  The NAR model expects 16 kHz mono float32.
We resample 24 kHz → 16 kHz on ingestion (torchaudio `AF.resample` already present
in the codebase).

Session configuration should also accept 16 kHz input directly (set
`input_audio_format: "pcm16"` with `sample_rate: 16000`) to avoid client-side
resampling overhead when the caller already has 16 kHz audio.

---

## Architecture

```
client (WebSocket)
  │  base64 PCM frames   (input_audio_buffer.append)
  ▼
[ WebSocket handler ]
  │  raw PCM appended to RingBuffer
  ▼
[ VAD thread ]  ──── silence detected ────►  [ Utterance queue ]
  │                                               │
  │  (or explicit commit)                         │
  ▼                                               ▼
[ RingBuffer drained ]                   [ Inference worker ]
                                               │  _nar_model.transcribe()
                                               │  (single MPS inference at a time)
                                               ▼
                                         [ Word emitter ]
                                               │  one delta event per word
                                               ▼
                                         [ WebSocket send ]
                                           completed event
```

Key constraints inherited from the existing server:

- `_INFER_SEM` (Semaphore(1)) — MPS handles one inference at a time.  Multiple
  concurrent WebSocket sessions share the same semaphore and queue behind it.
- Inference runs in `_EXECUTOR` (ThreadPoolExecutor) so the asyncio event loop
  stays responsive during inference.

---

## Phase 0 — Prerequisites & interface skeleton

**File:** `serve_nar_stream.py` (new file, imports `serve_nar` model/processor)

### 0a. Add WebSocket dependency

```
# requirements.txt — already satisfied by uvicorn[standard] which pulls websockets.
# No new package needed.
```

### 0b. Add FastAPI WebSocket route to serve_nar.py

```python
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/v1/realtime")
async def realtime(websocket: WebSocket, model: str = "granite-nar"):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    await websocket.send_json({
        "type": "session.created",
        "session": {
            "id": session_id,
            "model": model,
            "input_audio_format": "pcm16",
            "sample_rate": 24000,        # OpenAI default; overridable via session.update
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5,
                "silence_duration_ms": 600,
                "prefix_padding_ms": 300,
            },
        },
    })
    session = NarSession(session_id, websocket)
    try:
        await session.run()
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
```

### 0c. Session configuration via session.update

Fields to honour:

```python
{
  "type": "session.update",
  "session": {
    "input_audio_format": "pcm16",   # only supported format
    "sample_rate": 16000,            # 16000 or 24000
    "turn_detection": {
      "type": "server_vad",          # "server_vad" | "none"
      "threshold": 0.5,              # RMS energy threshold (0–1)
      "silence_duration_ms": 600,    # silence needed to close utterance
      "prefix_padding_ms": 300,      # audio prepended before speech onset
    },
  }
}
```

---

## Phase 1 — Audio ingestion & ring buffer

### 1a. Decode incoming frames

```python
import base64, numpy as np

def decode_audio_append(event: dict, session_sr: int) -> np.ndarray:
    """Decode base64 PCM16 payload → float32 mono at 16 kHz."""
    raw = base64.b64decode(event["audio"])
    pcm16 = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if session_sr != 16_000:
        import torch, torchaudio.functional as AF
        t = torch.from_numpy(pcm16).unsqueeze(0)
        t = AF.resample(t, session_sr, 16_000)
        pcm16 = t.squeeze(0).numpy()
    return pcm16
```

### 1b. Ring buffer

Maintain a `bytearray` (or `np.ndarray` with a write pointer) per session.
Append decoded audio on every `input_audio_buffer.append` event.

Maximum buffer size: **60 s × 16000 samples × 4 bytes = ~3.8 MB**.  Reject
(emit `error`) if the buffer exceeds this without a commit.

### 1c. Prefix padding

Keep a rolling window of the last `prefix_padding_ms` of audio.  When VAD fires
`speech_started`, prepend this window to the utterance so the first syllable is
not clipped.

---

## Phase 2 — Voice Activity Detection (VAD)

The existing `find_split_sample` in `serve_base.py` / `serve_plus_proxy.py` is a
batch silence finder; here we need an **online, streaming VAD** that fires as
audio arrives.

### 2a. Energy-based VAD (Phase 2, fast to implement)

Run in a background task that polls the ring buffer every 20 ms:

```python
FRAME_MS   = 20
FRAME_SAMP = int(16_000 * FRAME_MS / 1_000)   # 320 samples

async def vad_loop(session: "NarSession"):
    while not session.closed:
        if len(session.buffer) >= FRAME_SAMP:
            frame = session.buffer.drain(FRAME_SAMP)          # consume oldest frame
            rms   = np.sqrt(np.mean(frame ** 2))
            is_speech = rms > session.config.vad_threshold * 0.01   # normalised
            session.vad_state_machine(is_speech, frame)
        else:
            await asyncio.sleep(FRAME_MS / 1_000)
```

VAD state machine:

```
IDLE ──(speech onset)──► SPEECH ──(silence >= silence_duration_ms)──► FLUSH
 ▲                                                                        │
 └────────────────────────────────────────────────────────────────────────┘
```

- **IDLE → SPEECH**: emit `input_audio_buffer.speech_started`; start accumulating
  utterance (including prefix_padding).
- **SPEECH → FLUSH**: emit `input_audio_buffer.speech_stopped`; push utterance
  audio to the inference queue.
- **FLUSH → IDLE**: reset after inference completes.

### 2b. Silero VAD (Phase 2b, optional upgrade)

[Silero VAD](https://github.com/snakers4/silero-vad) is a lightweight PyTorch model
(~1 MB) that provides more accurate speech/silence detection than energy thresholds.
It runs in real time on CPU alongside MPS inference.

```python
# pip install silero-vad (add to requirements.txt)
from silero_vad import load_silero_vad, get_speech_timestamps
model, utils = load_silero_vad()
```

Use as a drop-in replacement for the energy loop in Phase 2a.  The session config
`turn_detection.type` can select `"server_vad_energy"` vs `"server_vad_silero"`.

---

## Phase 3 — Inference pipeline

### 3a. Utterance queue

```python
import asyncio
utterance_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
```

VAD pushes complete utterance audio; the inference worker consumes it.

### 3b. Inference worker

```python
async def inference_worker(session: "NarSession"):
    while not session.closed:
        audio = await session.utterance_queue.get()
        waveform = torch.from_numpy(audio)
        loop = asyncio.get_event_loop()
        async with _INFER_SEM:
            text = await loop.run_in_executor(
                _EXECUTOR, _infer, waveform
            )
        await session.emit_transcript(text)
```

### 3c. Chunking very long utterances before inference

If a VAD segment exceeds **14 s** (NAR encoder limit, to be confirmed — see
Investigation below), split using the same `find_split_sample` function from
`serve_base.py` / `serve_plus_proxy.py` before pushing to the inference queue.
Stitch results with a space (NAR produces no timestamps or speaker tags).

---

## Phase 4 — Streaming output

The NAR model produces all text in one shot — there are no incremental tokens.
To match the OpenAI `delta` event contract we emit word-level deltas
**post-hoc** after inference completes, with a small inter-word delay to simulate
streaming:

```python
async def emit_transcript(self, text: str, item_id: str):
    words = text.split()
    accumulated = ""
    for word in words:
        accumulated += ("" if not accumulated else " ") + word
        await self.ws.send_json({
            "type": "conversation.item.input_audio_transcription.delta",
            "item_id": item_id,
            "delta": word + " ",
        })
        await asyncio.sleep(0)    # yield to event loop; remove for max throughput

    await self.ws.send_json({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "transcript": text,
    })
```

The `await asyncio.sleep(0)` yields control so other event-loop tasks (e.g.,
accepting new audio frames) are not starved.  Remove it if the client prefers
all deltas in a single burst.

**Note:** True sub-word streaming (partial words) is not possible with NAR.
Clients that require it should use the plus model (port 8701) in timestamps mode,
which is autoregressive and can emit tokens as they are generated.

---

## Phase 5 — Explicit commit & turn detection off

When `turn_detection.type = "none"`, the server never auto-flushes.  The client
controls segmentation entirely via `input_audio_buffer.commit`:

```python
elif event["type"] == "input_audio_buffer.commit":
    audio = session.drain_buffer()
    if len(audio):
        await session.utterance_queue.put(audio)
        await session.ws.send_json({"type": "input_audio_buffer.committed"})
```

This mode is useful for batch-style clients that pre-segment audio externally and
want precise control over utterance boundaries.

---

## Phase 6 — NAR encoder limit investigation

**Before implementing chunking in Phase 3c**, measure the NAR model's actual
maximum input duration.

Add a diagnostic endpoint:

```python
@app.post("/v1/audio/probe")
async def probe(file: UploadFile = File(...)):
    """Return encoder shape info for the uploaded audio."""
    audio_bytes = await file.read()
    waveform = load_audio_bytes(audio_bytes)
    inputs = processor([waveform], device=DEVICE)
    # Inspect input tensor shape before inference
    shape_info = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}
    return JSONResponse({"duration_s": len(waveform) / 16000, "input_shapes": shape_info})
```

Run against 10 s, 30 s, 60 s, 120 s audio.  If any tensor dimension plateaus
or the model crashes, that determines the hard encoder ceiling and the required
chunk size for Phase 3c.

---

## Phase 7 — Concurrency & backpressure

Multiple simultaneous WebSocket sessions share `_INFER_SEM(1)`.  Sessions queue
behind the semaphore.  To avoid unbounded queueing:

- Cap `utterance_queue` at **4 items**.  If full, emit an `error` event and
  instruct the client to slow down.
- Expose queue depth in the health endpoint:

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "device": DEVICE,
        "active_sessions": len(_active_sessions),
        "inference_queue_depth": sum(s.utterance_queue.qsize() for s in _active_sessions),
    }
```

---

## Phase 8 — Authentication

The existing `verify_key` dependency works for HTTP routes only.  For WebSocket:

```python
@app.websocket("/v1/realtime")
async def realtime(websocket: WebSocket, ...):
    # OpenAI Realtime passes the key as a query param or Authorization header
    token = websocket.query_params.get("api_key") \
            or websocket.headers.get("authorization", "").removeprefix("Bearer ")
    if _API_KEY and not hmac.compare_digest(token, _API_KEY):
        await websocket.close(code=4401, reason="Unauthorized")
        return
    await websocket.accept()
    ...
```

---

## Implementation order

| Phase | Task | Effort |
|-------|------|--------|
| 0 | Skeleton WebSocket route + `session.created` handshake | Small |
| 1 | Audio ingestion, decode, ring buffer | Small |
| 6 | Encoder limit investigation (probe endpoint + tests) | Small |
| 2a | Energy-based VAD state machine | Medium |
| 3 | Utterance queue + inference worker | Small |
| 4 | Post-hoc word-delta streaming + `completed` event | Small |
| 5 | Explicit commit / `turn_detection: none` mode | Small |
| 7 | Backpressure, session tracking, health endpoint | Small |
| 8 | WebSocket auth | Small |
| 2b | Silero VAD upgrade (optional) | Medium |

---

## Test plan

### Unit tests (no server required)

```python
# test_nar_stream.py
def test_decode_audio_append_24k():
    """24 kHz input is resampled to 16 kHz."""

def test_decode_audio_append_16k():
    """16 kHz input passes through unchanged."""

def test_vad_state_machine_fires_on_silence():
    """VAD transitions SPEECH→FLUSH after silence_duration_ms of quiet frames."""

def test_emit_transcript_word_deltas():
    """Each word in the transcript produces one delta event before completed."""
```

### Integration tests (live server)

```bash
# 1. Short utterance — single VAD segment, fast path
python3 test_nar_stream_client.py --audio test.wav --mode vad
# Expect: speech_started, speech_stopped, N delta events, completed

# 2. Explicit commit — no VAD
python3 test_nar_stream_client.py --audio test.wav --mode commit
# Expect: committed, N delta events, completed

# 3. Long audio — verify chunking fires for audio > encoder limit
ffmpeg -stream_loop -1 -i test.wav -t 120 /tmp/long.wav
python3 test_nar_stream_client.py --audio /tmp/long.wav --mode vad
# Expect: multiple speech_started/completed cycles, word count >= 60 wpm

# 4. Concurrent sessions — two clients simultaneously
# Expect: both complete without error; second client queues behind semaphore

# 5. Auth rejection
wscat -c "ws://localhost:8702/v1/realtime" --no-auth
# Expect: close code 4401

# 6. Regression — existing POST /v1/audio/transcriptions still works
bash test_endpoints.sh
```

### Reference client

```python
# test_nar_stream_client.py — minimal WebSocket test client
import asyncio, base64, json, sys, numpy as np, soundfile as sf
import websockets

async def stream(audio_path: str, mode: str = "vad"):
    uri = "ws://127.0.0.1:8702/v1/realtime?model=granite-nar"
    audio, sr = sf.read(audio_path, dtype="int16", always_2d=True)
    pcm = audio.mean(axis=1).astype("<i2")

    async with websockets.connect(uri) as ws:
        msg = json.loads(await ws.recv())
        assert msg["type"] == "session.created"
        print("Session:", msg["session"]["id"])

        if mode == "commit":
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"turn_detection": {"type": "none"}, "sample_rate": sr}
            }))

        # Send audio in 100 ms chunks
        chunk = int(sr * 0.1)
        for i in range(0, len(pcm), chunk):
            frame = pcm[i:i+chunk].tobytes()
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode(),
            }))
            await asyncio.sleep(0.1)    # simulate real-time sending

        if mode == "commit":
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async for raw in ws:
            ev = json.loads(raw)
            print(f"  [{ev['type']}]", ev.get("delta", ev.get("transcript", "")))
            if ev["type"] == "conversation.item.input_audio_transcription.completed":
                break

asyncio.run(stream(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "vad"))
```

---

## Non-goals

- **Token-by-token streaming from the model** — architecturally impossible with NAR.
  Use the plus model (port 8701) if sub-sentence latency from the model itself is
  required.
- **Speaker diarization in streaming mode** — NAR does not support speaker tags.
- **Timestamps in streaming mode** — NAR does not emit `[T:N]` tags.
- **Multi-channel audio** — input is always downmixed to mono.
- **Language detection or translation** — out of scope for this model.
