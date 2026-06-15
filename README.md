# granite-speech-4.1-serve

OpenAI-compatible speech-to-text API server for [IBM Granite Speech 4.1-2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b), plus and NAR variants, exposing three backends `POST /v1/audio/transcriptions` interface. This project also provides two wrapper endpoints that automatically chunck and restitch results of the base and plus models to overcome context length limitations.

| Port (env var) | Service | Model | Notes |
|----------------|---------|-------|-------|
| `$GRANITE_BASE_DIRECT_PORT` (default 8700) | `granite-base` | `granite-speech-4.1-2b` (Q8_0 GGUF) | Chunking proxy → llama-server on `$GRANITE_BASE_PROXY_PORT`; splits audio > 14 s at word boundaries |
| `$GRANITE_BASE_PROXY_PORT` (default 18700) | _(internal)_ | — | llama-server; loopback only |
| `$GRANITE_PLUS_DIRECT_PORT` (default 8701) | `granite-plus-proxy` | `granite-speech-4.1-2b-plus` | Chunking proxy → model on `$GRANITE_PLUS_PROXY_PORT`; timestamps + speaker stitching across chunks |
| `$GRANITE_PLUS_PROXY_PORT` (default 18701) | _(internal)_ | — | Plus model backend (PyTorch); loopback only |
| `$GRANITE_NAR_DIRECT_PORT` (default 8702) | `granite-nar` | `granite-speech-4.1-2b-nar` | Non-autoregressive, fastest |

### Long-audio support

Both public ports (`$GRANITE_BASE_DIRECT_PORT` and `$GRANITE_PLUS_DIRECT_PORT`) are **chunking proxies** that handle arbitrarily long audio:

- Audio is split at word-boundary silences into chunks ≤ 14 s, forwarded sequentially to the backend, and stitched back together.
- **Base (`$GRANITE_BASE_DIRECT_PORT`):** text chunks are concatenated with a space.
- **Plus (`$GRANITE_PLUS_DIRECT_PORT`):** four stitching modes depending on the prompt:
  - *Plain ASR* — text concatenation.
  - *Timestamps* — `[T:N]` values (centiseconds mod 1000 per model design) are unwrapped into a globally-monotone timeline across all chunks.
  - *Speaker attribution* — speaker-aware chunking: audio ≤ 120 s is sent as a single request so the model has full context to distinguish speakers. Audio > 120 s is split into 60 s chunks, each prefixed with short (~3 s) reference clips of each detected speaker so labels remain consistent across chunks.
  - *Combined* — timestamp unwrapping applied first, then speaker remapping.

**Model limitations (plus backend):**
- Our testing found the model assigned at most 2 speaker labels (`[Speaker 1]:` / `[Speaker 2]:`) regardless of how many distinct voices are present.
- The combined (speaker + timestamps) prompt can cause the model to collapse similar-sounding voices (e.g. female pairs) to a single speaker label. This is a model limitation; the plain speaker-attribution prompt is more reliable for similar voices.

---

## Performance

Measured on **Apple M3 Ultra** (MPS) with `scripts/start_apple_dockerless.sh`, 33-minute
looped speech audio (226 chunks of ≤ 14 s each).

| Backend | Mode | Speed | Words | Chunks |
|---------|------|-------|-------|--------|
| Base :8700 (`$GRANITE_BASE_DIRECT_PORT`) | Plain ASR (punctuated) | **31.6× realtime** (62.7 s) | 4 437 (134 wpm) | 226 |
| Plus :8701 (`$GRANITE_PLUS_DIRECT_PORT`) | Plain ASR | **15.0× realtime** (131.9 s) | 4 574 (139 wpm) | 226 |
| Plus :8701 (`$GRANITE_PLUS_DIRECT_PORT`) | Word timestamps | **3.4× realtime** (582.7 s) | 5 424 tags, monotone [118..197951] cs | 226 |

"Speed" = audio duration ÷ wall-clock processing time (higher is faster).
Timestamps mode is slower because the model emits ~3 tokens per word instead of ~1.

---

## Quick start

```bash
cp .env.example .env          # set HF_TOKEN, LLAMA_API_KEY and GRANITE_API_KEY
```

**Option A — use pre-built images from ghcr.io (recommended):**
```bash
./scripts/start_ghcr.sh
```

**Option B — build from local source:**
```bash
./scripts/start_local_docker.sh
```

Both scripts auto-detect an NVIDIA GPU and its maximum supported CUDA version, then pick the highest compatible image/wheel set automatically:

| Detected CUDA | Image tag / wheel set | Typical GPUs |
|---------------|----------------------|--------------|
| None | `:latest` / `cpu` | CPU-only |
| < 12.8 | `:cuda` / `cu124` | Pascal → Ada Lovelace (RTX 4000 and earlier) |
| 12.8 – 12.x | `:cuda128` / `cu128` | Blackwell (RTX 5000 series, GB200) |
| 13.0+ | `:cuda130` / `cu130` | Next-gen beyond Blackwell |

Both also load `docker-compose.local.yml` if it exists — see [Local overrides](#local-overrides) below. On Mac Silicon, Docker pulls the `arm64` layer of `:latest` automatically — no separate script needed. Note that MPS acceleration is unavailable inside Docker on Mac (Linux VM); for native MPS performance, run the servers directly (see [Apple Silicon note](#apple-silicon-note) below).

Models are downloaded from HuggingFace on first start (several GB) and cached in a named volume.

---

## Running as a background service

Scripts for installing granite-speech as an auto-starting system service are
provided for both macOS and Linux. They handle starting the stack at boot,
stopping it cleanly on shutdown, and restarting it if it crashes.

| Platform | Mechanism | Directory |
|----------|-----------|-----------|
| macOS | launchd LaunchAgent | [`service/osx/`](service/osx/README.md) |
| Linux (Ubuntu / systemd) | systemd system service | [`service/linux-systemd/`](service/linux-systemd/README.md) |
| Windows | multiple options (WSL2, NSSM, Task Scheduler) | [`service/windows/`](service/windows/README.txt) |

**macOS** — registers `scripts/start_apple_dockerless.sh` as a LaunchAgent that runs at
login and restarts automatically on crash:

```bash
bash service/osx/install.sh
```

**Linux** — registers the docker compose stack as a systemd system service that
starts at boot. Choose `ghcr` (recommended) or `local` for the image source:

```bash
sudo bash service/linux-systemd/install.sh --mode ghcr
```

See the platform README for the full command reference, log options, and
uninstall instructions.

---

## API usage

All three endpoints accept `multipart/form-data` with a `file` field. Supported
formats: WAV, FLAC, OGG, MP3, MP4/AAC, and any other format handled by ffmpeg
(installed in all deployments). Audio is decoded and resampled to 16 kHz mono
before transcription.

```bash
# Basic transcription (any backend)
curl http://localhost:${GRANITE_PLUS_DIRECT_PORT:-8701}/v1/audio/transcriptions \
  -H "Authorization: Bearer $GRANITE_API_KEY" \
  -F file=@audio.wav

# Health check (no auth required)
curl http://localhost:${GRANITE_PLUS_DIRECT_PORT:-8701}/health
```

### Plus model prompt modes

The `granite-plus` backend (port `$GRANITE_PLUS_DIRECT_PORT`, default 8701) accepts an optional `prompt` field to control output style.

| Mode | Prompt |
|------|--------|
| Plain ASR (default) | `<\|audio\|> can you transcribe the speech into a written format?` |
| Word timestamps | `<\|audio\|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]` |
| Speaker attribution | `<\|audio\|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.` |
| Timestamps + speakers | `<\|audio\|> Timestamps and Speaker attribution: Transcribe the speech with proper punctuation and capitalization. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]. Denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns.` |
| Keyword biasing | `<\|audio\|> can you transcribe the speech into a written format? Keywords: word1, word2` |

When curling prompts that start with `<|audio|>`, use `--form-string` instead of `-F` (otherwise curl treats `<` as a file redirect and silently drops the value):

```bash
curl http://localhost:${GRANITE_PLUS_DIRECT_PORT:-8701}/v1/audio/transcriptions \
  -H "Authorization: Bearer $GRANITE_API_KEY" \
  -F file=@audio.wav \
  --form-string "prompt=<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
```

Timestamp values in raw model output wrap at 1000 centiseconds (model design); the proxy unwraps them into globally-monotone values across all chunks. The plus model does not reliably produce punctuation or capitalization regardless of prompt wording; use the base model (`$GRANITE_BASE_DIRECT_PORT`, default 8700) for punctuated output.

---

## Selecting which services to run

Set `COMPOSE_PROFILES` in `.env` to control which service groups start. The three profile names map directly to the three backends:

| Profile | Services started | Public port (env var) | Memory (approx.) |
|---------|-----------------|----------------------|-----------------|
| `base` | `llama-base` + `granite-base` proxy | 8700 (`$GRANITE_BASE_DIRECT_PORT`) | ~2 GB |
| `plus` | `granite-plus` + `granite-plus-proxy` | 8701 (`$GRANITE_PLUS_DIRECT_PORT`) | ~8 GB |
| `nar` | `granite-nar` | 8702 (`$GRANITE_NAR_DIRECT_PORT`) | ~4 GB |

**Default (all three):**
```
COMPOSE_PROFILES=base,plus,nar
```

**Base + NAR only (skip the plus model to save GPU memory):**
```
COMPOSE_PROFILES=base,nar
```

**NAR only:**
```
COMPOSE_PROFILES=nar
```

`docker compose up -d` picks up `COMPOSE_PROFILES` from `.env` automatically. `scripts/test_endpoints.sh` and `scripts/start_apple_dockerless.sh` read the same variable and skip sections for inactive profiles.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPOSE_PROFILES` | `base,plus,nar` | Which service groups to start — see [Selecting which services to run](#selecting-which-services-to-run) |
| `GRANITE_API_KEY` | _(unset = no auth)_ | Bearer token for plus and NAR servers |
| `LLAMA_API_KEY` | _(unset = no auth)_ | Bearer token for the llama.cpp base server |
| `GRANITE_SYSTEM_PROMPT` | IBM system prompt | Set to `""` to disable the system prompt |
| `HF_HOME` | `/cache/huggingface` | HuggingFace model cache directory |
| `PLUS_MAX_NEW_TOKENS` | `4096` | Max output tokens per chunk for the plus model (~3700 words) |
| `PLUS_INTERNAL_URL` | `http://127.0.0.1:$GRANITE_PLUS_PROXY_PORT/v1/audio/transcriptions` | Plus proxy → model URL (set automatically in Docker) |
| `PLUS_CHUNK_MAX_S` | `14` | Max chunk length in seconds for plain/timestamps modes |
| `PLUS_SPEAKER_MAX_UNCHUNKED_S` | `120` | Audio at or below this duration is sent as a single request in speaker/combined modes (avoids per-chunk speaker label drift) |
| `PLUS_SPEAKER_CHUNK_MAX_S` | `60` | Chunk size for speaker/combined modes when audio exceeds `PLUS_SPEAKER_MAX_UNCHUNKED_S` (preamble mode) |
| `GRANITE_BASE_DIRECT_PORT` | `8700` | Client-facing port for the base chunking proxy |
| `GRANITE_BASE_PROXY_PORT` | `18700` | Internal port for llama-server (base model backend) |
| `GRANITE_PLUS_DIRECT_PORT` | `8701` | Client-facing port for the plus chunking proxy |
| `GRANITE_PLUS_PROXY_PORT` | `18701` | Internal port for the plus model server |
| `GRANITE_NAR_DIRECT_PORT` | `8702` | Client-facing port for the NAR model server |

---

## Local overrides

Create a `docker-compose.local.yml` file in the project root to customise your deployment without touching the git-tracked compose files. Both start scripts pick it up automatically if it exists; it is listed in `.gitignore` so it will never be committed. Common use cases for this is to expose service the direct ports, or change resource allocation settings. 

---

## Docker images

Pre-built images are published to `ghcr.io/angrave/granite-speech-4.1-serve` on every push to `main`.

| Tag | Platforms | PyTorch | When to use |
|-----|-----------|---------|-------------|
| `latest` | `linux/amd64`, `linux/arm64` | 2.6.0 | CPU inference — plain x86_64 servers and Apple Silicon |
| `cuda` | `linux/amd64`, `linux/arm64` | 2.6.0 | NVIDIA CUDA 12.4 — Pascal → Ada Lovelace (RTX 4000 and earlier) |
| `cuda128` | `linux/amd64`, `linux/arm64` | 2.11.0 | NVIDIA CUDA 12.8 — Blackwell (RTX 5000 series, GB200) |
| `cuda130` | `linux/amd64` | 2.11.0 | NVIDIA CUDA 13.0 — next-gen beyond Blackwell (amd64 only) |

Docker pulls the correct architecture automatically. The start scripts detect your GPU's CUDA version and select the right tag — no manual choice needed.

### Enabling NVIDIA GPU passthrough

docker-compose.gpu.yml has examples of mapping GPU resources to Docker containers.

Then pick the image tag that matches your GPU's CUDA version and ensure `nvidia-container-toolkit` is installed on the host. The `start_ghcr.sh` script does this selection automatically.

### Apple Silicon note

MPS acceleration is not available inside Docker (Linux VM). For native MPS performance, use the provided script:

```bash
cp .env.example .env      # fill in GRANITE_API_KEY and LLAMA_API_KEY
./scripts/start_apple_dockerless.sh
```

`scripts/start_apple_dockerless.sh` lazy-installs all dependencies on first run (Python 3.11, a venv, PyTorch arm64 + MPS, and the Python requirements), then starts all three servers. The only prerequisite it does **not** auto-install:

- **Homebrew** — install from <https://brew.sh> if missing

Python 3.10+ is required (the NAR model's remote code uses Python 3.10+ union-type syntax). The script auto-installs `python@3.11` via Homebrew if no suitable interpreter is found.

**llama.cpp (`granite-base`):** The script checks for a suitable `llama-server` binary in this order:
1. A previously cached binary in `.llama_build/` (instant)
2. The Homebrew-installed `llama-server`, if it supports `granite_speech` (instant)
3. A pre-built binary downloaded from the [latest GitHub Release](https://github.com/angrave/granite-speech-4.1-serve/releases/latest) (~seconds)
4. A full source build from the llama.cpp `main` branch — only if all of the above fail (~10 min, cached for subsequent runs)

Server output is written to `runtime/logs/base.log`, `runtime/logs/plus.log`, and `runtime/logs/nar.log`. Run `tail -f runtime/logs/*.log` in a second terminal to monitor startup. Models are downloaded from HuggingFace on first run (several GB each); subsequent starts load from cache.

The script starts one process per active profile: `base` → `llama-server` (`:$GRANITE_BASE_PROXY_PORT`) + `serve_base` proxy (`:$GRANITE_BASE_DIRECT_PORT`); `plus` → `serve_plus` model (`:$GRANITE_PLUS_PROXY_PORT`) + `serve_plus_proxy` (`:$GRANITE_PLUS_DIRECT_PORT`); `nar` → `serve_nar` (`:$GRANITE_NAR_DIRECT_PORT`). Which profiles are active is read from `COMPOSE_PROFILES` in `.env` (default: all three). The plus proxy waits for the plus model to be healthy before starting. Port defaults can be overridden via the `GRANITE_*_PORT` variables in `.env`.

Press `Ctrl-C` to stop all servers.

---

## Building locally

```bash
# CPU (default)
docker build -t granite-speech .

# NVIDIA CUDA 12.4 — Pascal → Ada Lovelace (RTX 4000 and earlier)
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  --build-arg PYTORCH_VERSION=2.6.0 \
  -t granite-speech:cuda .

# NVIDIA CUDA 12.8 — Blackwell (RTX 5000 series, GB200)
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  --build-arg PYTORCH_VERSION=2.11.0 \
  -t granite-speech:cuda128 .

# NVIDIA CUDA 13.0 — next-gen beyond Blackwell
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 \
  --build-arg PYTORCH_VERSION=2.11.0 \
  -t granite-speech:cuda130 .
```

# References
The test.wav and other testing files used TalkBank multiconversastion (4074.mp3 4404.mp3 4941.mp3) at https://talkbank.org/ca/access/CallHome/eng.html

Linguistic Data Consortium (2008). CABank English CallHome Corpus. TalkBank. doi:10.21415/T5KP54
Canavan, A., Graff, D., & Zipperlen, G. (1997). CALLHOME American English Speech LDC97S42. Philadelphia: Linguistic Data Consortium.


