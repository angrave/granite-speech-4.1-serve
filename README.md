# granite-speech-4.1-serve

OpenAI-compatible speech-to-text API server for [IBM Granite Speech 4.1-2B](https://huggingface.co/ibm-granite), exposing three backends behind a single `POST /v1/audio/transcriptions` interface.

| Port | Service | Model | Notes |
|------|---------|-------|-------|
| 9797 | `granite-base` | `granite-speech-4.1-2b` (Q8_0 GGUF) | Chunking proxy → llama-server on :19797; splits audio > 14 s at word boundaries |
| 19797 | _(internal)_ | — | llama-server; loopback only |
| 8001 | `granite-plus-proxy` | `granite-speech-4.1-2b-plus` | Chunking proxy → model on :18001; timestamps + speaker stitching across chunks |
| 18001 | _(internal)_ | — | Plus model backend (PyTorch); loopback only |
| 8002 | `granite-nar` | `granite-speech-4.1-2b-nar` | Non-autoregressive, fastest |

### Long-audio support

Both public ports (9797 and 8001) are **chunking proxies** that handle arbitrarily long audio:

- Audio is split at word-boundary silences into chunks ≤ 14 s, forwarded sequentially to the backend, and stitched back together.
- **Base (9797):** text chunks are concatenated with a space.
- **Plus (8001):** three stitching modes depending on the prompt:
  - *Plain ASR* — text concatenation.
  - *Timestamps* — `[T:N]` values (centiseconds mod 1000 per model design) are unwrapped into a globally-monotone timeline across all chunks.
  - *Speaker attribution* — `[Speaker N]:` labels are remapped at chunk boundaries to prevent phantom speaker flips.
  - *Combined* — timestamp unwrapping applied first, then speaker remapping.

---

## Performance

Measured on **Apple M3 Ultra** (MPS) with `start_apple_dockerless.sh`, 33-minute
looped speech audio (226 chunks of ≤ 14 s each).

| Backend | Mode | Speed | Words | Chunks |
|---------|------|-------|-------|--------|
| Base :9797 | Plain ASR (punctuated) | **31.6× realtime** (62.7 s) | 4 437 (134 wpm) | 226 |
| Plus :8001 | Plain ASR | **15.0× realtime** (131.9 s) | 4 574 (139 wpm) | 226 |
| Plus :8001 | Word timestamps | **3.4× realtime** (582.7 s) | 5 424 tags, monotone [118..197951] cs | 226 |

"Speed" = audio duration ÷ wall-clock processing time (higher is faster).
Timestamps mode is slower because the model emits ~3 tokens per word instead of ~1.

---

## Quick start

```bash
cp .env.example .env          # set HF_TOKEN, LLAMA_API_KEY and GRANITE_API_KEY
```

**Option A — use pre-built images from ghcr.io (recommended):**
```bash
./start_ghcr.sh
```

**Option B — build from local source:**
```bash
./start_local_docker.sh
```

Both scripts auto-detect an NVIDIA GPU: on Linux with a CUDA-capable GPU they pull/build the `:cuda` image and enable GPU passthrough; otherwise they use the CPU image. On Mac Silicon, Docker pulls the `arm64` layer of `:latest` automatically — no separate script needed. Note that MPS acceleration is unavailable inside Docker on Mac (Linux VM); for native MPS performance, run the servers directly (see [Apple Silicon note](#apple-silicon-note) below).

Models are downloaded from HuggingFace on first start (several GB) and cached in a named volume.

---

## API usage

All three endpoints accept `multipart/form-data` with a `file` field (WAV, MP3, FLAC, …).

```bash
# Basic transcription (any backend)
curl http://localhost:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer $GRANITE_API_KEY" \
  -F file=@audio.wav

# Health check (no auth required)
curl http://localhost:8001/health
```

### Plus model prompt modes

The `granite-plus` backend (port 8001) accepts an optional `prompt` field to control output style.

| Mode | Prompt |
|------|--------|
| Plain ASR (default) | `<\|audio\|> can you transcribe the speech into a written format?` |
| Word timestamps | `<\|audio\|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]` |
| Speaker attribution | `<\|audio\|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.` |
| Timestamps + speakers | `<\|audio\|> Timestamps and Speaker attribution: Transcribe the speech with proper punctuation and capitalization. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]. Denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.` |
| Keyword biasing | `<\|audio\|> can you transcribe the speech into a written format? Keywords: word1, word2` |

When curling prompts that start with `<|audio|>`, use `--form-string` instead of `-F` (otherwise curl treats `<` as a file redirect and silently drops the value):

```bash
curl http://localhost:8001/v1/audio/transcriptions \
  -H "Authorization: Bearer $GRANITE_API_KEY" \
  -F file=@audio.wav \
  --form-string "prompt=<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
```

Timestamp values in raw model output wrap at 1000 centiseconds (model design); the proxy unwraps them into globally-monotone values across all chunks. The plus model does not reliably produce punctuation or capitalization regardless of prompt wording; use the base model (port 9797) for punctuated output.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRANITE_API_KEY` | _(unset = no auth)_ | Bearer token for plus and NAR servers |
| `LLAMA_API_KEY` | _(unset = no auth)_ | Bearer token for the llama.cpp base server |
| `GRANITE_SYSTEM_PROMPT` | IBM system prompt | Set to `""` to disable the system prompt |
| `HF_HOME` | `/cache/huggingface` | HuggingFace model cache directory |
| `PLUS_MAX_NEW_TOKENS` | `4096` | Max output tokens per chunk for the plus model (~3700 words) |
| `PLUS_INTERNAL_URL` | `http://127.0.0.1:18001/v1/audio/transcriptions` | Plus proxy → model URL (set automatically in Docker) |
| `PLUS_CHUNK_MAX_S` | `14` | Max chunk length in seconds for the plus proxy |

---

## Docker images

Pre-built images are published to `ghcr.io/angrave/granite-speech-4.1-serve` on every push to `main`.

| Tag | Platforms | When to use |
|-----|-----------|-------------|
| `latest` | `linux/amd64`, `linux/arm64` | CPU inference — plain x86_64 servers and Apple Silicon |
| `cuda` | `linux/amd64`, `linux/arm64` | NVIDIA GPU — x86_64 servers (`amd64`) and Spark G10/GB10/GH200 (`arm64`) |

Docker pulls the correct architecture automatically.

### Enabling NVIDIA GPU passthrough

Add to each service in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Then use the `cuda` image tag and ensure `nvidia-container-toolkit` is installed on the host.

### Apple Silicon note

MPS acceleration is not available inside Docker (Linux VM). For native MPS performance, use the provided script:

```bash
cp .env.example .env      # fill in GRANITE_API_KEY and LLAMA_API_KEY
./start_apple_dockerless.sh
```

`start_apple_dockerless.sh` lazy-installs all dependencies on first run (Python 3.11, a venv, PyTorch arm64 + MPS, and the Python requirements), then starts all three servers. The only prerequisite it does **not** auto-install:

- **Homebrew** — install from <https://brew.sh> if missing

Python 3.10+ is required (the NAR model's remote code uses Python 3.10+ union-type syntax). The script auto-installs `python@3.11` via Homebrew if no suitable interpreter is found.

**llama.cpp (`granite-base`):** The script checks for a suitable `llama-server` binary in this order:
1. A previously cached binary in `.llama_build/` (instant)
2. The Homebrew-installed `llama-server`, if it supports `granite_speech` (instant)
3. A pre-built binary downloaded from the [latest GitHub Release](https://github.com/angrave/granite-speech-4.1-serve/releases/latest) (~seconds)
4. A full source build from the llama.cpp `main` branch — only if all of the above fail (~10 min, cached for subsequent runs)

Server output is written to `base.log`, `plus.log`, and `nar.log` in the repo root. Run `tail -f *.log` in a second terminal to monitor startup. Models are downloaded from HuggingFace on first run (several GB each); subsequent starts load from cache.

The script starts five processes: `llama-server` (:19797), `serve_base` proxy (:9797), `serve_plus` model (:18001), `serve_plus_proxy` (:8001), and `serve_nar` (:8002). The proxy on :8001 waits for the model on :18001 to be healthy before starting.

Press `Ctrl-C` to stop all servers.

---

## Building locally

```bash
# CPU (default)
docker build -t granite-speech .

# NVIDIA CUDA 12.4
docker build --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t granite-speech:cuda .
```

# References
The test.wav was created from TalkBank multiconversastion (4404.mp3) at https://talkbank.org/ca/access/CallHome/eng.html

Linguistic Data Consortium (2008). CABank English CallHome Corpus. TalkBank. doi:10.21415/T5KP54
Canavan, A., Graff, D., & Zipperlen, G. (1997). CALLHOME American English Speech LDC97S42. Philadelphia: Linguistic Data Consortium.


