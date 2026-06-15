# Dockerfile for Granite Speech FastAPI servers (plus on port 18701, nar on port 8702).
# Runs on CPU by default; set NVIDIA_VISIBLE_DEVICES for GPU passthrough on Linux+CUDA.
# Note: Apple MPS is not available inside Docker (Linux VM).

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# PYTORCH_INDEX_URL controls which wheel set is used:
#   CPU (default): https://download.pytorch.org/whl/cpu
#   CUDA 12.4:     https://download.pytorch.org/whl/cu124
# PyTorch CUDA wheels are self-contained (bundled runtime); no CUDA host install needed.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir \
    torch torchaudio \
    --index-url ${PYTORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serve_plus.py serve_nar.py serve_plus_proxy.py serve_base.py ./

# HuggingFace model cache — mount a named volume here to avoid re-downloading on restart.
ENV HF_HOME=/cache/huggingface
VOLUME ["/cache/huggingface"]

# Default to plus server; override CMD in docker-compose or docker run.
EXPOSE 18701
CMD ["uvicorn", "serve_plus:app", "--host", "0.0.0.0", "--port", "18701"]
