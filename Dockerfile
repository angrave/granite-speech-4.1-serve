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
#   CUDA 12.6:     https://download.pytorch.org/whl/cu126   (Pascal -> Ada tier; see docker-compose.yml / docker-publish.yml)
# PyTorch CUDA wheels are self-contained (bundled runtime); no CUDA host install needed.
# PYTORCH_VERSION: arm64 runners top out at 2.5.1 on pytorch.org/whl; all others use 2.6.0.
#
# Whichever PYTORCH_VERSION/PYTORCH_INDEX_URL pair you pass here, pip resolves
# torch's *transitive* nvidia-cudnn-cu12 (etc.) pins against that same
# --index-url. Those pins are exact-version (torch==2.6.0 requires
# nvidia-cudnn-cu12==9.1.0.70) and pytorch.org's per-CUDA-tag indexes do not
# always retain every nvidia-* patch release torch has ever pinned — e.g.
# cu124's index is missing 9.1.0.70 entirely (still on PyPI, just not
# mirrored there), which breaks a `cu124 + torch==2.6.0` build with
# "No matching distribution found for nvidia-cudnn-cu12==9.1.0.70". This does
# NOT affect CPU builds (the +cpu wheels declare no nvidia-* dependencies at
# all) or non-x86_64/non-Linux hosts (the pin is marker-gated to
# platform_system=="Linux" and platform_machine=="x86_64"). If a CUDA build
# hits this again, check whether the pinned nvidia-cudnn-cu12 version for
# your PYTORCH_VERSION is still listed at
# https://download.pytorch.org/whl/<tag>/nvidia-cudnn-cu12/ and, if not, bump
# PYTORCH_VERSION (and PYTORCH_INDEX_URL if needed) to a release whose pin
# does resolve.
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG PYTORCH_VERSION=2.6.0
RUN pip install --no-cache-dir \
    torch==${PYTORCH_VERSION} torchaudio==${PYTORCH_VERSION} \
    --index-url ${PYTORCH_INDEX_URL}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/serve_plus.py src/serve_nar.py src/serve_plus_proxy.py src/serve_base.py ./

# HuggingFace model cache — mount a named volume here to avoid re-downloading on restart.
ENV HF_HOME=/cache/huggingface
VOLUME ["/cache/huggingface"]

# Default to plus server; override CMD in docker-compose or docker run.
EXPOSE 18701
CMD ["uvicorn", "serve_plus:app", "--host", "0.0.0.0", "--port", "18701"]
