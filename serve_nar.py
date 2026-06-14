"""FastAPI server for granite-speech-4.1-2b-nar on port 8002.

OpenAI-compatible endpoint: POST /v1/audio/transcriptions
Auth: set GRANITE_API_KEY env var to require Bearer token. Unset = no auth.
Local-only binding is enforced by the uvicorn --host argument at launch time.

Note on flash_attention_2: the model card recommends it but it is CUDA-only.
On Apple Silicon we use attn_implementation="sdpa" (PyTorch scaled-dot-product
attention), which is MPS-compatible and gives correct output.

Concurrency: see serve_plus.py for the full rationale. Same pattern applies here.
"""
import asyncio
import hmac
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

# Workaround: the NAR model's remote code imports PreTrainedConfig (capital T) but
# current transformers exposes PretrainedConfig (lowercase t) from configuration_utils.
# Patch the alias before the model's remote code runs.
import transformers.configuration_utils as _cu
if not hasattr(_cu, "PreTrainedConfig") and hasattr(_cu, "PretrainedConfig"):
    _cu.PreTrainedConfig = _cu.PretrainedConfig  # type: ignore[attr-defined]

import soundfile as sf
import torch
import torchaudio.functional as AF
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from transformers import AutoModel, AutoProcessor

MODEL_ID = "ibm-granite/granite-speech-4.1-2b-nar"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
_API_KEY = os.environ.get("GRANITE_API_KEY", "")
_bearer = HTTPBearer(auto_error=False)

_INFER_SEM = asyncio.Semaphore(1)
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

processor = None
_nar_model = None


def verify_key(creds: HTTPAuthorizationCredentials = Security(_bearer)):
    if not _API_KEY:
        return
    if creds is None or not hmac.compare_digest(creds.credentials, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, _nar_model
    if _API_KEY:
        print("Auth enabled: Bearer token required on /v1/audio/transcriptions")
    else:
        print("WARNING: GRANITE_API_KEY not set — server accepts unauthenticated requests")
    print(f"Loading {MODEL_ID} on {DEVICE} ...")
    attn_impl = "flash_attention_2" if DEVICE == "cuda" else "sdpa"
    _nar_model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        device_map=DEVICE,
        dtype=DTYPE,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("NAR model ready.")
    yield
    _EXECUTOR.shutdown(wait=False)


app = FastAPI(title="Granite Speech NAR", lifespan=lifespan)


def load_audio_bytes(data: bytes) -> torch.Tensor:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)  # (channels, samples)
    if sr != 16000:
        waveform = AF.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def _infer(waveform: torch.Tensor) -> str:
    """Synchronous inference — runs in the thread executor, not the event loop."""
    inputs = processor([waveform], device=DEVICE)
    with torch.inference_mode():
        output = _nar_model.transcribe(**inputs)
    return processor.batch_decode(output.preds)[0].strip()


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_key)])
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("nar"),
    prompt: str = Form(""),
):
    audio_bytes = await file.read()
    waveform = load_audio_bytes(audio_bytes)
    loop = asyncio.get_event_loop()
    async with _INFER_SEM:
        text = await loop.run_in_executor(_EXECUTOR, _infer, waveform)
    return JSONResponse({"text": text})


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE,
            "auth": "enabled" if _API_KEY else "disabled"}
