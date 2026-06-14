# Bug Report: granite_speech — silent empty transcription for audio > ~17 s

**Repo:** https://github.com/ggml-org/llama.cpp  
**Commit tested:** `5ce013cd7` (build 8590, source-built, AppleClang 17, Darwin arm64)  
**Model:** `ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0`  
**Endpoint:** `POST /v1/audio/transcriptions`  
**Platform:** Apple M3 Ultra, macOS 15, Metal backend  

---

## Summary

When audio longer than roughly 17 seconds is submitted to `llama-server` with the
`granite_speech` multimodal projector, the API returns an **empty transcription**
(`"text": ""`) with **`output_tokens: 1`** and no HTTP error.  The server logs no
warning and the response looks structurally valid.  Short audio (≤ 15–17 s) works
correctly.

---

## Observed behaviour

```json
// 35-second audio → silent failure
{ "type":"transcript.text.done", "text":"",
  "usage":{ "input_tokens":366, "output_tokens":1, "total_tokens":367,
            "input_tokens_details":{ "cached_tokens":0 } } }

// 15-second clip of the same audio → correct output
{ "type":"transcript.text.done",
  "text":"i don't think i'll hang up mhm me either so tell me about your plans …",
  "usage":{ "input_tokens":165, "output_tokens":29, "total_tokens":194,
            "input_tokens_details":{ "cached_tokens":0 } } }
```

From `base.log` — note the `0.00 ms / 1 token` eval against 366 prompt tokens:

```
I slot print_timing: prompt eval time = 292.11 ms / 366 tokens
I slot print_timing:        eval time =   0.00 ms /   1 tokens   ← single EOS-like token
I slot      release: n_tokens = 366, truncated = 0
```

Subsequent requests with the **same** audio file correctly reuse the KV cache
(`cached_tokens = 365`) and return the same empty result — consistent with the
underlying failure being cached, not compounded.

---

## Audio token counts by clip duration

| Duration | Input tokens | Audio tokens¹ | Output tokens | Result |
|----------|-------------|---------------|---------------|--------|
| 15 s     | 165         | 155           | 27–29         | ✅ OK  |
| 16 s     | 177         | 167           | 1             | ❌ empty |
| 17 s     | 186         | 176           | 31            | ✅ OK  |
| 18 s     | 195         | 185           | 1             | ❌ empty |
| 20 s     | 216         | 206           | 1             | ❌ empty |
| 25 s     | 267         | 257           | 1             | ❌ empty |
| 30 s     | 315         | 305           | 1             | ❌ empty |
| 35 s     | 366         | 356           | 1             | ❌ empty |

¹ Audio tokens = input tokens − 10 text-prefix tokens.

The failure is **not monotonic** (17 s succeeds while 16 s and 18 s do not), which
suggests the failure threshold may depend on audio-encoder block boundaries rather
than a simple token count cutoff.

---

## Relevant GGUF audio parameters

Read from `mmproj-model-f16.gguf`:

```
clip.audio.chunk_size                = 200   # encoder context window (frames)
clip.audio.max_pos_emb               = 512   # max position embeddings in projector
clip.audio.projector.window_size     = 15    # projector cross-attn window
clip.audio.projector.downsample_rate = 5     # LLM token compression factor
clip.audio.projector.head_count      = 16
clip.audio.num_mel_bins              = 160
clip.audio.conv_kernel_size          = 15
```

Audio-token derivation for N seconds of 16 kHz audio:

```
n_padded   = N × 16000 + 512          (reflect padding, n_fft=512)
n_len_mel  = N × 16000 / 160 + 1     (hop=160)  → always odd → drop last frame
n_stacked  = N × 50                   (pair-stack every 2 mel frames)
nblocks_pr = ceil(n_stacked / 15)     (projector window_size = 15)
lm_tokens  = nblocks_pr × 3 + 5      (num_queries = 15/5 = 3, + ~5 framing tokens)
```

For 35 s: n_stacked = 1750, nblocks_pr = 117, lm_tokens ≈ 356 — well below
`max_pos_emb = 512`, so a position-embedding overflow alone does not explain the
failure.

---

## Preprocessor path (for reference)

`granite_speech` uses its own preprocessor
(`mtmd_audio_preprocessor_granite_speech::preprocess`, `mtmd-audio.cpp:774`).
It applies only reflective padding (n_fft/2 = 256 samples each side), pair-stacks
consecutive mel frames, and returns a **single** mel chunk regardless of audio
length.  It does **not** use the 3000-frame chunking that the Whisper preprocessor
applies.

The encoder then divides the stacked frames into `ceil(n_stacked / chunk_size)`
= `ceil(1750 / 200)` = 9 encoder blocks.  The projector sees 9 × 200 = 1800 padded
frames and produces 117 × 3 = 351 output tokens.

No assertion failure or error is logged by the server; the Metal graph completes but
the LLM generates only one token (empty string or invisible special token) and halts.

---

## Steps to reproduce

**Prerequisites:**  
- macOS, Apple Silicon  
- `llama-server` built from commit `5ce013cd7` (or nearby `main`)  
- `ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0` + `mmproj-model-f16.gguf`  
- A 35-second (or longer) 16 kHz mono WAV file

**1. Start the server:**

```bash
llama-server \
  -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 \
  --port 9797 --host 127.0.0.1 \
  --api-key test-key
```

Wait for: `server is listening on http://127.0.0.1:9797`

**2. Transcribe a short clip (should succeed):**

```bash
# Create a 15-second clip from any 16 kHz WAV
python3 - <<'EOF'
import wave
with wave.open("long.wav") as src:
    r,c,w = src.getframerate(), src.getnchannels(), src.getsampwidth()
    data = src.readframes(r * 15)
with wave.open("short.wav", "w") as dst:
    dst.setnchannels(c); dst.setsampwidth(w); dst.setframerate(r)
    dst.writeframes(data)
EOF

curl -s http://127.0.0.1:9797/v1/audio/transcriptions \
  -H "Authorization: Bearer test-key" \
  -F "model=ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  -F "file=@short.wav" \
  -F "prompt=transcribe with punctuation and capitalization."
# Expected: non-empty "text" field
```

**3. Transcribe the full file (reproduces the bug):**

```bash
curl -s http://127.0.0.1:9797/v1/audio/transcriptions \
  -H "Authorization: Bearer test-key" \
  -F "model=ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0" \
  -F "file=@long.wav" \
  -F "prompt=transcribe with punctuation and capitalization."
# Bug: {"text":""} with output_tokens=1, no HTTP error
```

**4. Confirm it is NOT a KV-cache artefact:**

Restart the server between step 2 and step 3 so slot 0 is cold. The failure
reproduces on a completely fresh KV cache (server log shows `sim = 0.000` for the
slot selection).

---

## Distinguishing from KV-cache artefacts

`llama-server`'s `get_common_prefix` (`server-common.cpp:470`) correctly uses the
FNV hash of the raw audio bytes as the chunk ID when comparing multimodal tokens.
Sending *different* audio files will therefore **not** reuse a stale KV entry.

What does happen: after a long-audio request fails and its 366-token prompt state is
cached, every subsequent request with **the same** audio file matches that cache
(correctly, since the content is identical) and returns the same empty result until
the server restarts or the slot is evicted.  This is expected behaviour, not a
separate bug.

---

## Suspected area

`tools/mtmd/models/granite-speech.cpp` — encoder graph or projector graph
construction for inputs with a high `num_blocks` value (≥ 5–9 encoder blocks,
i.e., n_stacked ≥ 850).  The Metal graph executes without error but the LLM sees
degenerate audio embeddings and immediately halts generation.

Possible sub-causes to investigate:
- Positional embedding tensor shape mismatch when `num_blocks_encoder > 4`
- `ggml_pad` producing NaN/zero embeddings for the tail block under Metal
- The projector's cross-attention (`K`, `V` reshaped with `nblocks_proj` in the
  batch dimension) triggering a Metal kernel path that silently returns zeros

---

## Environment

```
llama-server build : 8590 (5ce013cd7), AppleClang 17.0.0.17000604, Darwin arm64
GPU               : Apple M3 Ultra (MTLGPUFamilyApple9)
Metal             : libggml-metal, residency sets enabled, flash-attn auto → enabled
n_parallel        : 4 (auto), kv_unified = true
n_ctx             : 4096
Model             : granite-speech-4.1-2b Q8_0 (1.84 B params)
mmproj            : mmproj-model-f16.gguf (granite_speech projector)
```
