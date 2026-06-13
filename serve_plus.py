"""FastAPI server for granite-speech-4.1-2b-plus on port 8001.

OpenAI-compatible endpoint: POST /v1/audio/transcriptions
Auth: set GRANITE_API_KEY env var to require Bearer token. Unset = no auth.
Local-only binding is enforced by the uvicorn --host argument at launch time.

Plus-model prompt modes (pass via the `prompt` form field):
  Plain ASR:    "<|audio|> can you transcribe the speech into a written format?"
  Timestamps:   "<|audio|> Timestamps: Transcribe the speech. After each word, add a
                 timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
                 [T:N] timestamps wrap at 1000 (centiseconds mod 1000 per model design).
  Speakers:     "<|audio|> Speaker attribution: Transcribe and denote who is speaking
                 by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
  Keywords:     "<|audio|> can you transcribe the speech into a written format? Keywords: word1, word2"

The IBM system prompt (SYSTEM_PROMPT below) is required for timestamps and speaker
attribution to activate — omitting it causes those modes to fall back to plain transcription.

OpenAI API compatibility note:
  timestamp_granularities[], verbose_json response format, speaker diarization, and keyword
  biasing have no standard OpenAI fields. Use the `prompt` field with the formats above.
  When curling prompts that start with <|audio|>, use --form-string instead of -F:
  curl -F treats values starting with < as file-content references and silently drops them.

Concurrency:
  MPS (and CUDA) handle one inference at a time. _INFER_SEM serializes GPU access.
  Inference runs in a ThreadPoolExecutor so the asyncio event loop stays responsive
  (health checks, new connections) while a request is in flight.
  Set --limit-concurrency on uvicorn to cap the queue and return 503 for excess requests.
"""
import asyncio
import hmac
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import soundfile as sf
import torch
import torchaudio.functional as AF
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL_ID = "ibm-granite/granite-speech-4.1-2b-plus"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
_API_KEY = os.environ.get("GRANITE_API_KEY", "")
_bearer = HTTPBearer(auto_error=False)

# Required by the model for timestamps and speaker attribution to activate.
SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
DEFAULT_PROMPT = "<|audio|> can you transcribe the speech into a written format?"
SAA_PROMPT = "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."

# Serialize GPU access: MPS/CUDA handles one inference at a time.
_INFER_SEM = asyncio.Semaphore(1)
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

processor = None
_model = None


def verify_key(creds: HTTPAuthorizationCredentials = Security(_bearer)):
    if not _API_KEY:
        return
    if creds is None or not hmac.compare_digest(creds.credentials, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, _model
    if _API_KEY:
        print("Auth enabled: Bearer token required on /v1/audio/transcriptions")
    else:
        print("WARNING: GRANITE_API_KEY not set — server accepts unauthenticated requests")
    print(f"Loading {MODEL_ID} on {DEVICE} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        dtype=DTYPE,
        device_map=DEVICE,
    ).eval()
    print("Model ready.")
    yield
    _EXECUTOR.shutdown(wait=False)


app = FastAPI(title="Granite Speech Plus", lifespan=lifespan)


def load_audio_bytes(data: bytes) -> torch.Tensor:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T)  # (channels, samples)
    if sr != 16000:
        waveform = AF.resample(waveform, sr, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)


def _infer(waveform: torch.Tensor, user_content: str) -> str:
    """Synchronous inference — runs in the thread executor, not the event loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text_input, audio=waveform, device=DEVICE, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        generated = _model.generate(**inputs, max_new_tokens=800)
    input_len = inputs["input_ids"].shape[1]
    return processor.tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0].strip()


@app.post("/v1/audio/transcriptions", dependencies=[Depends(verify_key)])
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("plus"),
    prompt: str = Form(DEFAULT_PROMPT),
):
    audio_bytes = await file.read()
    waveform = load_audio_bytes(audio_bytes)
    user_content = prompt if prompt.startswith("<|audio|>") else f"<|audio|> {prompt}"
    loop = asyncio.get_event_loop()
    async with _INFER_SEM:
        text = await loop.run_in_executor(_EXECUTOR, _infer, waveform, user_content)
    return JSONResponse({"text": text})


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE,
            "auth": "enabled" if _API_KEY else "disabled"}
