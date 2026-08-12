# Repetition-loop hallucination in `serve_plus.py` — analysis and fix

**Date:** 2026-08-12 · **Hardware:** NVIDIA RTX 5070 Ti (sm_120), driver 580.126.20
**Image:** `ghcr.io/angrave/granite-speech-4.1-serve:cuda130`
**Model:** `ibm-granite/granite-speech-4.1-2b-plus`, timestamps mode

## Summary

Granite Speech plus intermittently enters a repeating cycle mid-generation and emits
it hundreds of times — `herr kommissar` ×204, `thank you` ×340, `it` ×655, `but` ×668.
Over a 19-lecture / 16.3 h academic-lecture corpus this affected **12 of 19 lectures
and 4.65 % of all emitted tokens**, and cost:

| | WER | 1−WAR | insertion rate |
|---|---:|---:|---:|
| Granite plus, as decoded | 0.1141 | **0.0294** | 0.0846 |
| Granite plus, loops removed | 0.0668 | 0.0296 | 0.0372 |
| *(reference: Whisper large-v3, same audio)* | *0.0657* | *0.0384* | *0.0273* |

The loops are almost pure insertion: they roughly **double Granite's WER while
leaving 1−WAR untouched**. Removing them moves Granite from "clearly worse than
Whisper large-v3" to statistically indistinguishable (paired bootstrap over 19
lectures, p = 0.97), while preserving the finding that Granite *recalls* more of the
reference than Whisper does (1−WAR 0.0296 vs 0.0384, p = 0.023).

**Root cause: `serve_plus.py` called `generate()` with no repetition control.**

```python
generated = _model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
```

Greedy decoding with no `repetition_penalty` and no `no_repeat_ngram_size`. Once the
model enters a repeating cycle there is no mechanism to leave it, so it runs until
the token budget is exhausted. The signature is unmistakable: every looping response
lands at 682–684 tokens, i.e. `PLUS_MAX_NEW_TOKENS=4096` fully consumed.

## Reproduction

All four cases are public MIT OCW / Open Yale lectures. Audio as 16 kHz mono WAV:

```bash
yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
  "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

| lecture | video id | loop | onset |
|---|---|---|---|
| MIT 5.07 Biochemistry, *Carbohydrates / Membranes* | `KLb5CmPM7YY` | `herr kommissar` ×204 | 471.6 s, 1154.3 s, 2626.4 s |
| MIT 6.0001 *Intro to CS and Programming in Python*, Lec 1 | `xAcTmDO6NTI` | `it` ×655 | 2492.8 s |
| Yale PSYC 110 *Introduction to Psychology*, Lec 1 | `P3FKHH2RzjI` | `but` ×668 | 1274.2 s |
| MIT 24.900 *Introduction to 'The Society of Mind'* | `-pb3z2w9gDg` | `da` ×667, `ba` ×344 | 6260.3 s, 6201.3 s |

Cut a 14 s clip at the onset and post it to the plus backend directly:

```bash
ffmpeg -ss 2492.8 -t 14 -i 60001.wav -ar 16000 -ac 1 clip.wav
curl -X POST http://127.0.0.1:18701/v1/audio/transcriptions \
  -F "file=@clip.wav" \
  --form-string 'prompt=<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag showing the end time in centiseconds, e.g. hello [T:45] world [T:82]'
```

Expected on unpatched `serve_plus.py`: 683 tokens, `it` repeated 677 times.

> **Note — `herr kommissar` is not in the audio.** There is no German anywhere in
> this corpus, and the string appears in **zero** Whisper large-v3 transcripts of the
> same 19 lectures. It is training-data leakage surfacing over near-silence.

## Experiment 1 — is the chunking proxy the cause? No.

`serve_plus_proxy.py` feeds the model 14 s chunks, so a natural suspicion is that the
segmenter creates the pathological inputs. It does not. Each condition below is a
**direct call to the model endpoint (`:18701`), bypassing the proxy entirely**, over
the six largest loops.

| condition | looped |
|---|---|
| **A — no chunking at all (single 180 s request)** | **2 / 6** |
| B — the 14 s window the proxy would cut | 3 / 6 |
| C — same window shifted −7 s / −3 s | 2 / 6 · 1 / 6 |
| C — same window shifted +3 s / +7 s | 0 / 6 · 0 / 6 |
| D — control windows from loop-free regions | 0 / 12 |

Condition A: 24.900 loops `da` ×587 and 6.0001 loops `it` ×368 with no *external*
chunking. Note the model's own feature extractor still windows internally, so this
rules out the proxy, not all windowing. With n=6 the reproduction rate carries a wide
interval (Wilson ≈ [4 %, 71 %]) — this experiment alone would be thin evidence, and it
is superseded by the prompt-mode experiment below.

Two secondary findings. Loops are **audio-locked** — zero of twelve controls looped,
so it is not random. And the **window modulates whether a given loop fires** —
shifting by +3 s suppressed every loop tested — which makes the segmenter a *trigger*,
not a cause.

*Caveat:* condition B approximates the proxy's cut, since the proxy splits on silence
rather than a fixed grid, so the exact chunk boundary is not guaranteed. One known
loop (5.07 @1154 s) did not reproduce under any probe window. Condition A depends on
no boundary guess and is unaffected by this.

## Experiment 2 — can chunk size fix it? No.

Full re-decodes at `PLUS_CHUNK_MAX_S` ∈ {10, 14, 20}, loop tokens as % of output:

| lecture | 10 s | 14 s (default) | 20 s |
|---|---:|---:|---:|
| Yale PSYC 110 | **0.40 %** | 13.1 % | 13.1 % |
| MIT 5.07 | 23.67 % | 23.8 % | 17.2 % |

No chunk size is safe. 10 s rescues one lecture and does nothing for the other.
Consistent with Experiment 1: changing the window relocates triggers rather than
removing the failure mode.


## Experiment 5 — is it specific to the timestamps prompt? No.

Timestamps mode emits ~3 tokens per word, which is exactly why looping responses stop
at 682-684 of a 4096-token budget. That invites the obvious alternative explanation:
the `[T:N]` structure, not the missing repetition control, is what locks the model in.

Full-lecture decodes through the normal chunking proxy, unpatched server:

| lecture | timestamps | plain |
|---|---:|---:|
| MIT 5.07 | 23.8 % | **0.00 %** |
| MIT 6.0001 | 5.97 % | **28.93 %** |
| Yale PSYC 110 | 13.1 % | **0.00 %** |
| **aggregate loop tokens** | **3 768 / 26 394 = 14.3 %** | **4 063 / 26 043 = 15.6 %** |

**The prompt changes which lectures loop, not whether looping happens.** Plain mode is
marginally worse in aggregate. Both modes need the fix.

*A methodological warning, from getting this wrong first.* An earlier version probed
plain mode at the 15 onsets where *timestamps* mode looped and found 0/15 — a result
guaranteed by construction, since plain mode loops elsewhere. Conditioning test sites
on one arm's failures invalidates the comparison. The full-lecture decode above has no
such conditioning.

## Experiment 3 — is it an audio-quality artefact? Partly, for one class.

29 loop-onset windows vs 232 control windows drawn from loop-free regions **of the
same lectures**, so recording gain and room cancel. 25 ms frames, 10 ms hop.

| metric | loop (median [IQR]) | control | p | Bonferroni |
|---|---|---|---:|---:|
| noise floor (p10 frame energy) | −65.2 [−69.8, −59.0] dBFS | −63.5 [−68.9, −57.5] | 0.65 | 1.00 |
| speech level (p90) | −27.8 [−33.4, −22.8] | −26.1 [−29.0, −21.8] | 0.050 | 0.25 |
| within-window SNR | 34.5 [29.2, 38.8] dB | 37.5 [32.5, 41.0] | 0.045 | 0.23 |
| silence fraction | 0.37 | 0.35 | 0.27 | 1.00 |
| low-band (<300 Hz) ratio | 0.16 | 0.13 | 0.67 | 1.00 |

**The noise floor is not elevated** — if anything marginally lower. Nothing survives
correction. But this averages two distinct failure modes, which the intervention below
separates.

## Experiment 4 — level normalisation, and reproduction on a second backend

Each clip was re-levelled and re-submitted, to **both** the local container and the
hosted **NCSA Lumen** endpoint (`lumen.ncsa.illinois.edu`, `granite-speech-4.1-2b-plus`).

| clip | backend | original | rms→−20 dBFS | peak→−1 dBFS | loudnorm (R128) |
|---|---|---|---|---|---|
| 5.07 `herr kommissar` (−48.9 dBFS) | local | ×200 | **ok** (1 tok) | **ok** (4 tok) | **ok** (3 tok) |
| 5.07 `herr kommissar` (−48.9 dBFS) | lumen | ×98 | **ok** (1 tok) | **ok** (4 tok) | **ok** (3 tok) |
| 6.0001 `it` (−36.0 dBFS) | local | ×677 | ×677 | ×677 | ×678 |
| 6.0001 `it` (−36.0 dBFS) | lumen | ×336 | ×337 | ×336 | ×337 |
| Yale `but` (−25.3 dBFS) | local | ×3 | ×3 | ×3 | ×3 |
| Yale `but` (−25.3 dBFS) | lumen | ×4 | ×3 | ×3 | ×3 |

Two conclusions.

**It reproduces on Lumen.** Different infrastructure, different GPU, same loops —
independent confirmation that this is the model and not one deployment.

**There are two failure modes, not one.**

1. *Confabulation over near-silence.* The 5.07 clip is effectively silent
   (−48.9 dBFS RMS). Untreated, the model invents German. Levelled, it returns 1–4
   tokens — correctly reporting that there is nothing there. **Any** of the three
   normalisations fixes it, on both backends.
2. *Genuine decode-time repetition lock.* The 6.0001 `it` loop is completely
   unmoved by +16 dB — 677 repeats before, 677 after. Not level-related.

This is why Experiment 3 found nothing: the two modes were pooled. Level
normalisation is a real but *partial* mitigation, and it is complementary to the
decoding fix — it addresses the class the penalty only truncates.

## The fix, and what it costs

Full re-decodes of the three worst-affected lectures, scored against gold under the
Whisper `EnglishTextNormalizer` (Open ASR Leaderboard normalisation, which maps
spelled numbers to digits in both directions — necessary here because Granite spells
numbers out and Whisper emits digits).

| arm | WER | 1−WAR | insertion rate |
|---|---:|---:|---:|
| baseline (unpatched) | 0.2103 | 0.0271 | 0.1833 |
| post-hoc loop removal | **0.0470** | 0.0279 | 0.0192 |
| **`repetition_penalty=1.1`** | 0.0499 | 0.0288 | 0.0211 |

Loop tokens after the fix, per lecture: 5.07 **23.8 % → 0.00 %**, 6.0001
**5.97 % → 0.13 %**, Yale **13.1 % → 0.09 %**.

**Accuracy cost is +0.0017 in 1−WAR** — within noise — against a 0.16 WER reduction.
That is why the penalty is enabled by default. Set `PLUS_REPETITION_PENALTY=1.0` to
restore the previous behaviour exactly.

The penalty is also *better than post-hoc cleaning* where it matters most: while the
model is looping it is not transcribing, so that speech is lost and no downstream
filter can recover it. Re-decoding does. On MIT 5.07 the penalised decode yields
7 843 words against 7 773 non-loop words at baseline — it recovers content that
post-hoc removal cannot.

### Choosing the value

Max consecutive repeats of any 1–4 token cycle, on the two hardest clips:

| `PLUS_REPETITION_PENALTY` | 6.0001 `it` clip | 5.07 `herr kommissar` clip |
|---|---|---|
| 1.0 (previous behaviour) | 683 tok, **×677** | 616 tok, **×200** |
| 1.02 | 56 tok, ×8 | 13 tok, ×1 |
| 1.05 | 55 tok, ×7 | 13 tok, ×1 |
| **1.1 (new default)** | 49 tok, **×2** | 13 tok, ×1 |
| 1.15 | 50 tok, ×2 | 9 tok, ×1 |

Even 1.02 collapses the runaway — 683 tokens to 56 — which shows how shallow the
loop state is. But ×8 still trips a repetition detector. **1.1 is the lowest tested
value that fully clears it**, and 1.15 buys nothing while risking more suppression of
legitimate repeated speech.

Verified against the patched `src/serve_plus.py` in this PR, all three reproduction
clips:

```
ok  6.0001 'it'            49 tok  x2
ok  5.07 'herr kommissar'  13 tok  x1
ok  yale  'but'            44 tok  x2
```

### Recommended configuration

```bash
PLUS_REPETITION_PENALTY=1.1   # default; 1.0 restores previous behaviour
PLUS_NO_REPEAT_NGRAM=0        # off; blunter, blocks legitimate repeated phrases
```

Plus, for pipelines that control their own pre-processing: **normalise snippet level
before submission** (RMS to about −20 dBFS, or EBU R128). It removes the
near-silence confabulation class outright, which the penalty only truncates.


## Audio level normalisation (the second failure mode)

`repetition_penalty` fixes the decode-time repetition lock. It only *truncates* the
other mode — confabulation over near-silence — so the server also normalises level.

**Ablation, normalisation only, `PLUS_REPETITION_PENALTY=1.0`:**

| clip | class | result |
|---|---|---|
| 5.07 `herr kommissar` ×200 | near-silence | **ok**, 4 tok (−48.9 → −33.2 dBFS, +15.8 dB) |
| 6.0001 `it` ×677 | level-independent | **still ×677** (−36.0 → −24.2 dBFS, +11.8 dB) |
| 6.006 control | normal speech | ok, 48 tok (+7.2 dB) |
| Yale control | normal speech | ok, 57 tok (+2.1 dB) |

A clean dissociation: normalisation fixes that class and only that class, and leaves
normal speech unharmed. With both mitigations on (the shipped default) all four are
clean, including `it` ×677 → ×2.

The peak ceiling usually binds on quiet material — the 5.07 clip gets +15.8 dB rather
than the +28.9 dB the RMS target implies — which is exactly the `peak→−1` treatment
measured to work above.

### Why the gate is −50 dBFS and not the loss optimum

A level gate skips chunks below T. Two independent analyses say it cannot be the fix.

**Binary search from both directions.** Attenuate real speech until transcription
collapses; amplify a hallucinating silent clip until the loop stops.

| direction | result |
|---|---|
| speech floor (4 clips) | −87.4, −85.7, −91.0, −96.7 dBFS — survives 58–69 dB of attenuation |
| hallucination ceiling | −40.2 dBFS |

**The bounds are inverted.** Usable speech persists ~45 dB *below* where hallucination
still occurs, so there is no interval to split. Granite is essentially level-invariant
for speech, which is expected — the feature extractor normalises internally.

**Dose-response over 4 171 real 14 s windows** (all 19 lectures), labelled with RMS,
Whisper-attested word count, and Granite loop tokens. Gating at T costs the attested
words below T and saves the loop tokens of loops starting below T; both are word
errors, so they are directly comparable:

| gate | words lost | loop tokens saved | net |
|---:|---:|---:|---:|
| −50 | 3 | 0 | −3 |
| −45 | 137 | 408 | +271 |
| **−43** | **335** | **1 224** | **+889 (optimum)** |
| −40 | 1 141 | 1 624 | +483 |
| −35 | 11 786 | 4 434 | −7 352 |
| −30 | 58 811 | 5 963 | −52 848 |

The loss-optimal gate captures only **17.3 %** of loop tokens — 83 % of loops fire at
ordinary speech levels, where gating is catastrophic. And normalisation **dominates**
gating on the same audio: the windows a −43 dB gate discards hold 335 attested words,
whereas normalising those same clips stops the hallucination without discarding
anything.

So the shipped gate is a deep backstop at **−50 dBFS**, costing 3 transcribable words
across 16.3 h, rather than the loss optimum. Set `PLUS_NORMALIZE_AUDIO=0` to disable
normalisation and gating entirely.

## Known limitations

- `repetition_penalty=1.1` reduces but does not eliminate the 24.900 `da` loop
  (×585 → ×56). That lecture contains a *genuine* vocalised rhythm, so some
  repetition there is correct; the right threshold is unclear.
- Near-threshold cases are not bit-reproducible. The Yale clip measured ×4 and ×3 on
  identical input across runs — greedy decoding, so this is GPU float
  non-determinism flipping a borderline case.
- Loop incidence is measured on academic lecture audio only. Other domains may differ.
- Only `serve_plus.py` is affected. `serve_base.py` proxies to `llama-server`
  (repetition control is a llama-server flag, out of scope here) and the NAR model is
  non-autoregressive, so it is structurally immune to this failure mode.

## Artefacts

Detection and analysis code, plus a post-hoc cleaner for already-decoded output
(`PostGraniteDeHallucinator.py`), are in the companion research directory; the
per-loop coordinates in `loop_report.json` identify exactly which chunks to re-decode
for anyone repairing an existing corpus.
