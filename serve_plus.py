"""FastAPI server for granite-speech-4.1-2b-plus on port 18701 (internal; public proxy on 8701).

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
  Combined:     "<|audio|> Timestamps and Speaker attribution: Transcribe the speech with
                 proper punctuation and capitalization. After each word, add a timestamp tag
                 showing the end time in centiseconds, e.g. hello [T:45] world [T:82].
                 Denote who is speaking by adding [Speaker 1]: and [Speaker 2]: tags before
                 speaker turns."
                 Note: timestamps and speaker tags are produced reliably; punctuation/
                 capitalization is NOT — the plus model ignores that part of the prompt.
                 Use the base model (granite-speech-4.1-2b) for punctuated output (port 8700).
  Keywords:     "<|audio|> can you transcribe the speech into a written format? Keywords: word1, word2"

System prompt (SYSTEM_PROMPT / GRANITE_SYSTEM_PROMPT env var): optional in practice.
Tested with and without — timestamps and speaker attribution activate either way with
only ~1 cs timing jitter. Disable by setting GRANITE_SYSTEM_PROMPT="" if desired.

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

# IBM system prompt — used by default.  Set GRANITE_SYSTEM_PROMPT="" to disable.
# Required for timestamps and speaker attribution to activate reliably.
SYSTEM_PROMPT = os.environ.get(
    "GRANITE_SYSTEM_PROMPT",
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant",
)

MAX_NEW_TOKENS  = int(os.environ.get("PLUS_MAX_NEW_TOKENS", "4096"))

DEFAULT_PROMPT  = "<|audio|> can you transcribe the speech into a written format?"
PUNCT_PROMPT    = "<|audio|> transcribe the speech with proper punctuation and capitalization."
TS_PROMPT       = (
    "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag "
    "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
)
SAA_PROMPT      = (
    "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding "
    "[Speaker 1]: and [Speaker 2]: tags before speaker turns."
)
# Single prompt requesting punctuation, word-level timestamps, and speaker attribution.
# Tested: timestamps and speaker tags are produced reliably (with or without system prompt).
# Punctuation/capitalization is NOT produced by the plus model regardless of prompt wording.
COMBINED_PROMPT = (
    "<|audio|> Timestamps and Speaker attribution: Transcribe the speech with proper "
    "punctuation and capitalization. After each word, add a timestamp tag showing the "
    "end time in centiseconds, e.g. hello [T:45] world [T:82]. Denote who is speaking "
    "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
)

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
    if SYSTEM_PROMPT:
        print("System prompt: active (timestamps and speaker attribution enabled)")
    else:
        print("WARNING: GRANITE_SYSTEM_PROMPT is empty — timestamps/speaker attribution may fall back to plain ASR")
    print(f"Loading {MODEL_ID} on {DEVICE} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
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
    audio_key = next((k for k in ("audio_values", "input_features") if k in inputs), None)
    if audio_key:
        print(f"[plus] input audio tensor shape: {inputs[audio_key].shape}")
    with torch.inference_mode():
        generated = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
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
