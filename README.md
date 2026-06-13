# granite-speech-4.1-serve

OpenAI-compatible speech-to-text API server for [IBM Granite Speech 4.1-2B](https://huggingface.co/ibm-granite), exposing three backends behind a single `POST /v1/audio/transcriptions` interface.

| Port | Service | Model | Notes |
|------|---------|-------|-------|
| 9797 | `granite-base` | `granite-speech-4.1-2b` (Q8_0 GGUF) | llama.cpp, fast, punctuated output |
| 8001 | `granite-plus` | `granite-speech-4.1-2b-plus` | FastAPI, timestamps + speaker diarization |
| 8002 | `granite-nar` | `granite-speech-4.1-2b-nar` | FastAPI, non-autoregressive, fastest |

---

## Quick start (pre-built CUDA image — no build required)

Pull and run the published CUDA images with a single file — no source checkout needed:

```bash
# 1. Download the deploy compose file
curl -O https://raw.githubusercontent.com/angrave/granite-4.2/main/docker-compose-deploy.yml

# 2. Set credentials (all optional — omit to disable auth)
export HF_TOKEN=your-hf-token        # required if models are gated on HuggingFace
export GRANITE_API_KEY=your-secret   # bearer token for plus/NAR endpoints
export LLAMA_API_KEY=your-secret     # bearer token for base endpoint

# 3. Start all three services
docker compose -f docker-compose-deploy.yml up
```

Requires `nvidia-container-toolkit` on the host for GPU passthrough. Models download from HuggingFace on first start (several GB) and are cached in a named Docker volume.

---

## Quick start (build from source)

```bash
cp .env.example .env          # set HF_TOKEN, LLAMA_API_KEY and GRANITE_API_KEY
docker compose up --build
```

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

| Tag | Architecture | When to use |
|-----|-------------|-------------|
| `latest` | `linux/amd64` | CPU inference on x86_64 servers |
| `latest` | `linux/arm64` | CPU inference on Apple Silicon (Linux VM) |
| `cuda` | `linux/amd64` | NVIDIA GPU on x86_64 servers |
| `cuda` | `linux/arm64` | NVIDIA GPU on GB10 Grace-Blackwell / GH200 / Jetson Orin |

Each tag is a multi-arch manifest — `docker pull` selects the right architecture automatically:

```bash
docker pull ghcr.io/angrave/granite-speech-4.1-serve:latest   # CPU (amd64 or arm64)
docker pull ghcr.io/angrave/granite-speech-4.1-serve:cuda     # NVIDIA GPU (amd64 or arm64)
```

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

MPS acceleration is not available inside Docker (Linux VM). For native MPS performance, run the servers directly:

```bash
pip install torch torchaudio          # arm64 wheel, includes MPS
pip install -r requirements.txt
uvicorn serve_plus:app --port 8001
uvicorn serve_nar:app --port 8002
```

The llama.cpp base server has no arm64 Docker image; run it natively too:

```bash
llama-server -hf ibm-granite/granite-speech-4.1-2b-GGUF:Q8_0 \
  --port 9797 --host 127.0.0.1 --api-key "$LLAMA_API_KEY"
```

---

## Building locally

```bash
# CPU (default)
docker build -t granite-speech .

# NVIDIA CUDA 12.4
docker build --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  -t granite-speech:cuda .
```
