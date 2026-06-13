# granite-speech-4.1-serve

OpenAI-compatible speech-to-text API server for [IBM Granite Speech 4.1-2B](https://huggingface.co/ibm-granite), exposing three backends behind a single `POST /v1/audio/transcriptions` interface.

| Port | Service | Model | Notes |
|------|---------|-------|-------|
| 9797 | `granite-base` | `granite-speech-4.1-2b` (Q8_0 GGUF) | llama.cpp, fast, punctuated output |
| 8001 | `granite-plus` | `granite-speech-4.1-2b-plus` | FastAPI, timestamps + speaker diarization |
| 8002 | `granite-nar` | `granite-speech-4.1-2b-nar` | FastAPI, non-autoregressive, fastest |

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

Timestamp values wrap at 1000 centiseconds (model design). The plus model does not reliably produce punctuation or capitalization regardless of prompt wording; use the base model (port 9797) for punctuated output.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GRANITE_API_KEY` | _(unset = no auth)_ | Bearer token for plus and NAR servers |
| `LLAMA_API_KEY` | _(unset = no auth)_ | Bearer token for the llama.cpp base server |
| `GRANITE_SYSTEM_PROMPT` | IBM system prompt | Set to `""` to disable the system prompt |
| `HF_HOME` | `/cache/huggingface` | HuggingFace model cache directory |

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

`start_apple_dockerless.sh` lazy-installs all dependencies on first run (llama.cpp, Python 3.11, a venv, PyTorch arm64 + MPS, and the Python requirements), then starts all three servers. The only prerequisite it does **not** auto-install:

- **Homebrew** — install from <https://brew.sh> if missing

Python 3.10+ is required (the NAR model's remote code uses Python 3.10+ union-type syntax). The script auto-installs `python@3.11` via Homebrew if no suitable interpreter is found.

Server output is written to `base.log`, `plus.log`, and `nar.log` in the repo root. Run `tail -f *.log` in a second terminal to monitor startup. Models are downloaded from HuggingFace on first run (several GB each); subsequent starts load from cache.

Press `Ctrl-C` to stop all three servers.

> **llama.cpp version note:** The `granite-base` server (port 9797) requires a llama.cpp build that supports the `granite_speech` multimodal projector. If the Homebrew-installed version is too old you will see `unknown projector type: granite_speech` in `base.log` — the script will warn and keep the plus and NAR servers running. Build llama.cpp from source or wait for the Homebrew formula to update.

---

## Building locally

```bash
# CPU (default)
docker build -t granite-speech .

# NVIDIA CUDA 12.4
docker build --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t granite-speech:cuda .
```
