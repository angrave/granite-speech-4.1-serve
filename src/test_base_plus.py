"""Direct-inference tests for granite-speech-4.1-2b (base) and granite-speech-4.1-2b-plus.

Tests both models with empty/missing system prompt and various user prompts,
including a combined prompt that requests punctuation, timestamps, and speaker
attribution in a single call.

Usage:
  python test_base_plus.py                        # test plus only (base cached?)
  python test_base_plus.py --base                 # include base model tests
  python test_base_plus.py --audio path/to/audio.wav
  python test_base_plus.py --base --audio my.wav
"""
import argparse
import io
import sys
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)

# ── Prompt constants ───────────────────────────────────────────────────────────

# Base model (granite-speech-4.1-2b) — no system prompt required
BASE_ASR_PROMPT   = "<|audio|>can you transcribe the speech into a written format?"
BASE_PUNCT_PROMPT = "<|audio|>transcribe the speech with proper punctuation and capitalization."

# Plus model (granite-speech-4.1-2b-plus)
PLUS_ASR_PROMPT   = "<|audio|> can you transcribe the speech into a written format?"
PLUS_PUNCT_PROMPT = "<|audio|> transcribe the speech with proper punctuation and capitalization."
PLUS_TS_PROMPT    = (
    "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag "
    "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
)
PLUS_SAA_PROMPT   = (
    "<|audio|> Speaker attribution: Transcribe and denote who is speaking by adding "
    "[Speaker 1]: and [Speaker 2]: tags before speaker turns."
)
# Single prompt combining punctuation, word-level timestamps, and speaker attribution.
PLUS_COMBINED_PROMPT = (
    "<|audio|> Timestamps and Speaker attribution: Transcribe the speech with proper "
    "punctuation and capitalization. After each word, add a timestamp tag showing the "
    "end time in centiseconds, e.g. hello [T:45] world [T:82]. Denote who is speaking "
    "by adding [Speaker 1]: and [Speaker 2]: tags before speaker turns."
)


def load_audio(path: str) -> torch.Tensor:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(audio.T)  # (channels, samples)
    if sr != 16000:
        wav = AF.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.squeeze(0)  # (samples,)


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


# ── Base model inference ───────────────────────────────────────────────────────

def run_base(processor, model, wav: torch.Tensor, user_prompt: str, label: str):
    """Base model uses (prompt, wav) positional args to processor, no system prompt."""
    chat = [{"role": "user", "content": user_prompt}]
    prompt_str = processor.tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(prompt_str, wav, device=DEVICE, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    n = inputs["input_ids"].shape[-1]
    text = processor.tokenizer.batch_decode(
        outputs[:, n:], skip_special_tokens=True, add_special_tokens=False
    )[0].strip()
    print(f"\n[BASE] {label}")
    print(f"  prompt : {user_prompt!r}")
    print(f"  output : {text}")


def test_base(audio_path: str):
    section("granite-speech-4.1-2b  (base, no system prompt)")
    model_id = "ibm-granite/granite-speech-4.1-2b"
    print(f"Loading {model_id} on {DEVICE} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, device_map=DEVICE, torch_dtype=DTYPE
    ).eval()

    wav = load_audio(audio_path)
    print(f"Audio: {audio_path}  shape={wav.shape}  sr=16000")

    # HuggingFace default example — no system prompt
    run_base(processor, model, wav, BASE_ASR_PROMPT,
             "default ASR  (no system prompt, HF example)")

    # New example: punctuation + capitalization — no system prompt
    run_base(processor, model, wav, BASE_PUNCT_PROMPT,
             "punctuated ASR  (no system prompt)")

    # Empty audio token only — fall-back behaviour
    run_base(processor, model, wav, "<|audio|>",
             "empty user prompt  (fall-back)")

    del model, processor
    _clear_cache()


# ── Plus model inference ───────────────────────────────────────────────────────

def run_plus(processor, model, wav: torch.Tensor, user_prompt: str,
             label: str, use_system_prompt: bool = True):
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_prompt})
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=text_input, audio=wav, device=DEVICE, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=800)
    n = inputs["input_ids"].shape[-1]
    text = processor.tokenizer.batch_decode(
        outputs[:, n:], skip_special_tokens=True
    )[0].strip()
    sys_tag = "with system prompt" if use_system_prompt else "NO system prompt"
    print(f"\n[PLUS] {label}  ({sys_tag})")
    print(f"  prompt : {user_prompt!r}")
    print(f"  output : {text}")


def test_plus(audio_path: str, multi_speaker_path: str):
    section("granite-speech-4.1-2b-plus")
    model_id = "ibm-granite/granite-speech-4.1-2b-plus"
    print(f"Loading {model_id} on {DEVICE} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=DTYPE, device_map=DEVICE
    ).eval()

    wav = load_audio(audio_path)
    wav_ms = load_audio(multi_speaker_path)
    print(f"Audio: {audio_path}  shape={wav.shape}  sr=16000")
    print(f"Multi-speaker audio: {multi_speaker_path}  shape={wav_ms.shape}  sr=16000")

    section("Plus — single-speaker audio, NO system prompt")
    run_plus(processor, model, wav, PLUS_ASR_PROMPT,
             "default ASR", use_system_prompt=False)
    run_plus(processor, model, wav, PLUS_PUNCT_PROMPT,
             "punctuated ASR", use_system_prompt=False)
    run_plus(processor, model, wav, PLUS_TS_PROMPT,
             "timestamps", use_system_prompt=False)
    run_plus(processor, model, wav, PLUS_SAA_PROMPT,
             "speaker attribution", use_system_prompt=False)
    run_plus(processor, model, wav, PLUS_COMBINED_PROMPT,
             "combined (punct + timestamps + speakers)", use_system_prompt=False)

    section("Plus — single-speaker audio, WITH system prompt")
    run_plus(processor, model, wav, PLUS_ASR_PROMPT,
             "default ASR", use_system_prompt=True)
    run_plus(processor, model, wav, PLUS_PUNCT_PROMPT,
             "punctuated ASR", use_system_prompt=True)
    run_plus(processor, model, wav, PLUS_TS_PROMPT,
             "timestamps", use_system_prompt=True)
    run_plus(processor, model, wav, PLUS_SAA_PROMPT,
             "speaker attribution", use_system_prompt=True)
    run_plus(processor, model, wav, PLUS_COMBINED_PROMPT,
             "combined (punct + timestamps + speakers)", use_system_prompt=True)

    section("Plus — multi-speaker audio, combined prompt (NO system prompt)")
    run_plus(processor, model, wav_ms, PLUS_COMBINED_PROMPT,
             "combined on multi-speaker", use_system_prompt=False)

    section("Plus — multi-speaker audio, combined prompt (WITH system prompt)")
    run_plus(processor, model, wav_ms, PLUS_COMBINED_PROMPT,
             "combined on multi-speaker", use_system_prompt=True)

    del model, processor
    _clear_cache()


def _clear_cache():
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Test Granite Speech base and plus models")
    parser.add_argument("--base", action="store_true",
                        help="Also run base model tests (downloads if not cached)")
    parser.add_argument("--audio", default="test.wav",
                        help="Single-speaker audio file (default: test.wav)")
    parser.add_argument("--multi-speaker-audio", default="test_multi_speaker.wav",
                        help="Multi-speaker audio file (default: test_multi_speaker.wav)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}  dtype: {DTYPE}")

    if args.base:
        test_base(args.audio)

    test_plus(args.audio, args.multi_speaker_audio)

    print("\nDone.")


if __name__ == "__main__":
    main()
