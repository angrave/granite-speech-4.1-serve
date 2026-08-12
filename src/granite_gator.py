#!/usr/bin/env python3
"""granite-gator - a Granite Speech client that gates quiet audio and chomps hallucinations.

    GATE   refuse audio too quiet to transcribe, and level-normalise the rest
    CHOMP  collapse the repetition loops that survive anyway

Companion to the server-side fix in `serve_plus.py`. Full study, with every table
quoted below: `looping-analysis.md`.

WHY THIS EXISTS WHEN THE SERVER IS ALREADY FIXED
------------------------------------------------
You cannot always patch the server. Hosted Granite endpoints generally do not expose
`repetition_penalty`, and the loops occur there too - we reproduced them on NCSA's
hosted service (3 of 4 probe clips looped). This is the mitigation you can apply from
outside the server, and it needs no GPU.

===========================================================================
THE TWO FAILURE MODES THIS DEFENDS AGAINST, AND HOW TO RECREATE THEM
===========================================================================
Recreate before you tweak any threshold below. Every default here is measured, and
changing one without re-running the corresponding experiment will silently undo it.

Get the audio (all four are public lectures):

    yt-dlp -x --audio-format wav --postprocessor-args "ffmpeg:-ar 16000 -ac 1" \
        "https://www.youtube.com/watch?v=<VIDEO_ID>"

    VIDEO_ID       lecture                                   loop             onset
    xAcTmDO6NTI    MIT 6.0001 Intro to CS, Lecture 1         'it' x655        2492.8 s
    KLb5CmPM7YY    MIT 5.07 Carbohydrates/Membranes          'herr kommissar' 471.6 s,
                                                              x204            1154.3, 2626.4 s
    P3FKHH2RzjI    Yale PSYC 110 Intro to Psychology, Lec 1  'but' x668       1274.2 s
    -pb3z2w9gDg    MIT 24.900 'The Society of Mind'          'da' x667        6260.3 s

Cut a 14 s clip at an onset and post it to an UNPATCHED plus backend:

    ffmpeg -ss 2492.8 -t 14 -i 60001.wav -ar 16000 -ac 1 clip.wav
    curl -X POST localhost:18701/v1/audio/transcriptions -F "file=@clip.wav" \
         --form-string "prompt=$TS_PROMPT"

--- MODE 1: decode-time repetition lock -------------------------------------
Expected on the 6.0001 clip: 683 tokens, `it` repeated 677 times.

The model enters a repeating cycle and, with no repetition control in `generate()`,
cannot leave it - it emits the cycle until the token budget is gone. Diagnostic
signature: **every looping response lands at 682-684 tokens of a 4096 budget.**

LEVEL-INDEPENDENT. Amplifying the clip by +16 dB changes nothing: 677 repeats before,
677 after. Do not expect the gate or the normaliser to help here; only `chomp` (or the
server-side penalty) does.

--- MODE 2: confabulation over near-silence ---------------------------------
Expected on the 5.07 @471.6 s clip: 616 tokens of invented German,
`herr kommissarin` x200. There is no German anywhere in that corpus.

That clip is effectively silent (-48.9 dBFS RMS). Normalise it and the model returns
1-4 tokens instead - correctly reporting that nothing was said:

    ffmpeg -i clip.wav -af "volume=15.8dB" -ar 16000 -ac 1 norm.wav   # peak-limited

LEVEL-DEPENDENT, and this is the mode `normalize_level()` exists for. The content is
not a fixed phrase: `loudnorm` on the same clip yields Japanese instead. Treat it as a
multilingual degenerate-input attractor, not one memorised string.

===========================================================================
WHY EACH DEFAULT IS WHAT IT IS
===========================================================================
gate_dbfs = -50            NOT the loss-optimal -43 dBFS, deliberately.
    Dose-response over 4 171 real 14 s windows (16.3 h), trading words lost against
    loop tokens saved - both are word errors, so they are directly comparable:

        gate    words lost   loop tokens saved   net
        -50              3                   0     -3
        -45            137                 408   +271
        -43            335               1 224   +889   <- in-sample optimum
        -40          1 141               1 624   +483
        -35         11 786               4 434  -7 352

    The optimum captures only 17.3 % of loop tokens, because 83 % of loops fire at
    ordinary speech levels. And a binary search from both directions finds the bounds
    INVERTED: usable speech survives attenuation to -85..-97 dBFS (the model is
    essentially level-invariant for speech; its feature extractor normalises
    internally) while hallucination persists up to -40.2 dBFS - a 56.5 dB overlap.
    There is no threshold that separates the populations.

    So the gate is a backstop for material too quiet to normalise, not the fix.
    Below -50 dBFS the corpus held 3 transcribable words in 16.3 h.
    Recreate: level_gate_curve.py / bisect_level_gate.py in the study repository.

target_dbfs = -20          The level at which the near-silence clips stop
    confabulating, verified on two independent backends (local and hosted).

peak_ceiling_dbfs = -1     Never clip. On very quiet material THIS is what binds, not
    the RMS target: the 5.07 clip gets +15.8 dB rather than the +28.9 dB the target
    implies - which is exactly the `peak -> -1` treatment measured to fix it.

max_gain_db = 30           So digital silence is not amplified 60 dB into its own dither.

keep_repeats = 3           Collapse, never delete. Speakers genuinely repeat themselves,
    and MIT 24.900 contains a real vocalised rhythm ("da da da ...", ~6214 s) that
    Whisper large-v3 also transcribes. Deleting every repeated run destroys real
    content; keeping 3 preserves emphatic repetition while killing a 667-fold loop.

min_repeats = 4            Permissive on purpose: with collapse-to-3, a false positive
    costs almost nothing, so favour recall.

===========================================================================
WHAT THIS CANNOT DO - read before trusting it
===========================================================================
1. It cannot recover speech lost inside a loop. While the model loops it is not
   transcribing. Chomping removes insertions; the words underneath are gone.
   Re-decoding with the server-side penalty DOES recover them (MIT 5.07: 7 843 words
   penalised vs 7 773 non-loop words at baseline). Prefer the server fix where you
   control the server; use both where you can.

2. It cannot catch SHORT non-repetitive confabulations. ~120 words of invented
   European-Parliament boilerplate across the corpus trip no repetition, no
   compression-ratio and no confidence signal we could find. Per-token logprobs do not
   help: hallucinated clips show 0.922 of tokens at probability 1.0 versus 0.915 for
   controls - the model is maximally confident while inventing. Those are PREVENTED by
   normalisation, not detected. That is the argument for leaving `normalize` on.

3. A second recogniser is a poor hallucination oracle. Of 52 spans where Granite spoke
   and Whisper large-v3 was silent, only 16 lacked any gold speech - the other 36 were
   Granite correctly recovering speech Whisper missed. 69 % false-positive rate.

===========================================================================
A PARSER TRAP, IF YOU TOUCH `parse()`
===========================================================================
A response truncated at max_new_tokens - i.e. every looping one - can end mid-tag, and
the chunking proxy stitches that fragment into the middle of the stream. The
complete-tag regex does not match `[T:`, so without `_TAG_FRAG` the fragment survives
as a word (16 such tokens across 13 of 19 lectures in our corpus).

Do NOT widen that filter to catch bare `t` tokens. There are 38 of them and they are
LEGITIMATE SPEECH - each carries its own `[T:...]` timestamp and they read
`capital t for`, `i t undergraduate`. The lecturer is saying the letter T. Match on the
bracket characters, not on token content.

USAGE
-----
    # audio in, AsrWord stream out
    python granite_gator.py --audio lecture.wav --out lecture.asr.json \
        --endpoint http://localhost:8701/v1/audio/transcriptions

    # hosted endpoint that cannot set repetition_penalty
    python granite_gator.py --audio lecture.wav --out out.json \
        --endpoint https://<host>/v1/chat/completions --lumen --api-key "$KEY"

    # post-process an already-decoded stream (no audio, no endpoint, no GPU)
    python granite_gator.py --in raw.asr.json --out clean.asr.json --report r.json

    # library
    from granite_gator import GraniteGator, GatorConfig
    words, report = GraniteGator(GatorConfig(endpoint=...)).transcribe("lecture.wav")

VALIDATION (against a deliberately unpatched server)
----------------------------------------------------
    clip                            raw endpoint      via granite-gator
    5.07 near-silence               616 tok, x200     2 words, +15.8 dB, no loop
    6.0001 level-independent 'it'   683 tok, x677     7 words, 674 chomped
    6.006 normal speech control     39 words          39 words unchanged, +7.2 dB

    Post-processing 19 lectures: 152 227 -> 145 278 words (4.56 % chomped),
    29 loops found, 0 backward-timestamp events remaining.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["GraniteGator", "GatorConfig"]

TS_PROMPT = (
    "<|audio|> Timestamps: Transcribe the speech. After each word, add a timestamp tag "
    "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
)
PLAIN_PROMPT = "<|audio|> can you transcribe the speech into a written format?"
SYSTEM_PROMPT = ("Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
                 "You are Granite, developed by IBM. You are a helpful AI assistant")

_TS = re.compile(r"\[T:(\d+)\]")
_SPK = re.compile(r"\[Speaker (\d+)\]:?")
_SILENCE = "_"
# A response truncated at max_new_tokens -- i.e. every looping one -- can end
# mid-tag, and the proxy stitches that fragment into the middle of the stream.
# The complete-tag regexes above do not match `[T:` or a stranded `t`, so without
# this the fragments survive as words. Measured: 54 such tokens across 13 of 19
# lectures in the original corpus, concentrated exactly where the loops are.
_TAG_FRAG = re.compile(r"^\[+[Tt]?:?\d*\]?$|^\d*\]$")


@dataclass
class GatorConfig:
    endpoint: str = "http://127.0.0.1:8701/v1/audio/transcriptions"
    lumen: bool = False
    api_key: str | None = None
    model: str = "granite-speech-4.1-2b-plus"
    mode: str = "timestamps"            # timestamps | plain
    timeout_s: int = 7200

    # -- gate / normalise ------------------------------------------------ #
    gate_dbfs: float = -50.0
    normalize: bool = True
    target_dbfs: float = -20.0
    max_gain_db: float = 30.0
    peak_ceiling_dbfs: float = -1.0

    # -- chomp ----------------------------------------------------------- #
    chomp: bool = True
    min_repeats: int = 4
    max_cycle: int = 4
    keep_repeats: int = 3

    # -- chunking (client side; only used when we do our own splitting) --- #
    chunk_s: float = 14.0


@dataclass
class GatorReport:
    n_words_raw: int = 0
    n_words_out: int = 0
    chunks_total: int = 0
    chunks_gated: int = 0
    gain_applied_db: list[float] = field(default_factory=list)
    loops: list[dict] = field(default_factory=list)
    backward_before: int = 0
    backward_after: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_words_raw": self.n_words_raw, "n_words_out": self.n_words_out,
            "n_removed": self.n_words_raw - self.n_words_out,
            "chunks_total": self.chunks_total, "chunks_gated": self.chunks_gated,
            "median_gain_db": (sorted(self.gain_applied_db)[len(self.gain_applied_db) // 2]
                               if self.gain_applied_db else 0.0),
            "n_loops": len(self.loops),
            "loop_tokens": sum(l["tokens"] for l in self.loops),
            "backward_timestamp_events_before": self.backward_before,
            "backward_timestamp_events_after": self.backward_after,
            "loops": sorted(self.loops, key=lambda l: -l["tokens"])[:20],
        }


class GraniteGator:
    def __init__(self, config: GatorConfig | None = None, **kw) -> None:
        self.cfg = config or GatorConfig(**kw)

    # ---------------- audio ---------------- #
    @staticmethod
    def _read_wav(path: Path) -> tuple[list[float], int]:
        with wave.open(str(path)) as w:
            sr = w.getframerate()
            import array
            a = array.array("h")
            a.frombytes(w.readframes(w.getnframes()))
        return [x / 32768.0 for x in a], sr

    @staticmethod
    def _rms_dbfs(x: list[float]) -> float:
        if not x:
            return -999.0
        ms = sum(v * v for v in x) / len(x)
        return 10 * math.log10(ms) if ms > 0 else -999.0

    @staticmethod
    def _peak_dbfs(x: list[float]) -> float:
        pk = max((abs(v) for v in x), default=0.0)
        return 20 * math.log10(pk) if pk > 0 else -999.0

    def level_decision(self, samples: list[float]) -> tuple[bool, float, float]:
        """(gated, gain_db, rms_in). The GATE half of the name."""
        rms = self._rms_dbfs(samples)
        if rms < self.cfg.gate_dbfs:
            return True, 0.0, rms
        if not self.cfg.normalize:
            return False, 0.0, rms
        gain = max(min(self.cfg.target_dbfs - rms, self.cfg.max_gain_db), 0.0)
        headroom = self.cfg.peak_ceiling_dbfs - self._peak_dbfs(samples)
        return False, min(gain, max(headroom, 0.0)), rms

    def _prepare(self, src: Path, tmp: Path) -> tuple[Path | None, float, float]:
        """16 kHz mono, gated and level-normalised. None means gated."""
        conv = tmp / "in16k.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src),
                        "-ar", "16000", "-ac", "1", str(conv)], check=True)
        samples, _ = self._read_wav(conv)
        gated, gain, rms = self.level_decision(samples)
        if gated:
            return None, 0.0, rms
        if gain <= 0.05:
            return conv, 0.0, rms
        out = tmp / "norm.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(conv),
                        "-af", f"volume={gain:.2f}dB", "-ar", "16000", "-ac", "1", str(out)],
                       check=True)
        return out, gain, rms

    # ---------------- transport ---------------- #
    def _post(self, wav: Path) -> str:
        cfg = self.cfg
        prompt = TS_PROMPT if cfg.mode == "timestamps" else PLAIN_PROMPT
        if cfg.lumen:
            payload = {"model": cfg.model, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_audio",
                     "input_audio": {"data": base64.b64encode(wav.read_bytes()).decode(),
                                     "format": "wav"}}]}],
                "temperature": 0.0, "max_tokens": 2048}
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump(payload, fh)
                pf = fh.name
            cmd = ["curl", "-sS", "--max-time", str(cfg.timeout_s), "-X", "POST", cfg.endpoint,
                   "-H", "Content-Type: application/json", "--data-binary", f"@{pf}"]
            if cfg.api_key:
                cmd += ["-H", f"Authorization: Bearer {cfg.api_key}"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True)
                d = json.loads(r.stdout)
                return d["choices"][0]["message"]["content"] if "choices" in d else ""
            finally:
                Path(pf).unlink(missing_ok=True)
        cmd = ["curl", "-sS", "--max-time", str(cfg.timeout_s), "-X", "POST", cfg.endpoint,
               "-F", f"file=@{wav}", "--form-string", f"prompt={prompt}"]
        if cfg.api_key:
            cmd += ["-H", f"Authorization: Bearer {cfg.api_key}"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return json.loads(r.stdout).get("text", "")
        except json.JSONDecodeError:
            return ""

    # ---------------- parsing ---------------- #
    @staticmethod
    def parse(text: str, mode: str = "timestamps") -> list[dict]:
        """`[T:N]` stream -> AsrWord. Ported from granite_runner._parse_timestamps."""
        words: list[dict] = []
        if mode != "timestamps":
            for t in text.split():
                if t != _SILENCE:
                    words.append({"text": t.lower(), "start": 0.0, "end": 0.0,
                                  "speaker_id": None, "speaker_changed": False,
                                  "confidence": None, "is_last": False})
            if words:
                words[-1]["is_last"] = True
            return words
        prev, toks, i = 0.0, text.split(), 0
        prev_was_fragment = False
        while i < len(toks):
            t = toks[i]
            if _TS.match(t) or _SPK.match(t):
                i += 1
                prev_was_fragment = False
                continue
            if _TAG_FRAG.match(t) or "[" in t or "]" in t:
                i += 1
                prev_was_fragment = True
                continue
            if prev_was_fragment and t.lower() in ("t", "t:"):
                # `[T:` can arrive split across the fragment boundary
                i += 1
                continue
            prev_was_fragment = False
            if t == _SILENCE:
                i += 1
                if i < len(toks) and _TS.match(toks[i]):
                    i += 1
                continue
            end = prev
            if i + 1 < len(toks) and (m := _TS.match(toks[i + 1])):
                end = int(m.group(1)) / 100.0
                i += 2
            else:
                i += 1
            if end < prev:
                end = prev + 0.1
            words.append({"text": t.lower(), "start": round(prev, 3), "end": round(end, 3),
                          "speaker_id": None, "speaker_changed": False,
                          "confidence": None, "is_last": False})
            prev = end
        if words:
            words[-1]["is_last"] = True
        return words

    # ---------------- chomp ---------------- #
    def find_loops(self, words: list[dict]) -> list[dict]:
        toks = [w["text"] for w in words]
        n, loops, i = len(toks), [], 0
        while i < n:
            best = None
            for c in range(1, self.cfg.max_cycle + 1):
                if i + c > n:
                    break
                reps = 1
                while (i + (reps + 1) * c <= n
                       and toks[i:i + c] == toks[i + reps * c:i + (reps + 1) * c]):
                    reps += 1
                if reps >= self.cfg.min_repeats and (best is None or reps * c > best[0] * best[1]):
                    best = (reps, c)
            if best is None:
                i += 1
                continue
            reps, c = best
            span = reps * c
            loops.append({"index": i, "cycle": " ".join(toks[i:i + c]), "repeats": reps,
                          "tokens": span, "start_s": words[i]["start"],
                          "end_s": words[i + span - 1]["end"]})
            i += span
        return loops

    @staticmethod
    def _backward(words: list[dict]) -> int:
        n, prev = 0, float("-inf")
        for w in words:
            if w["end"] < prev - 1e-9:
                n += 1
            prev = max(prev, w["end"])
        return n

    def chomp(self, words: list[dict], report: GatorReport | None = None) -> list[dict]:
        rep = report or GatorReport()
        rep.n_words_raw = rep.n_words_raw or len(words)
        rep.backward_before = self._backward(words)
        loops = self.find_loops(words)
        rep.loops.extend(loops)
        drop: set[int] = set()
        for l in loops:
            cyc_len = len(l["cycle"].split())
            drop.update(range(l["index"] + self.cfg.keep_repeats * cyc_len,
                              l["index"] + l["tokens"]))
        out = [dict(w) for i, w in enumerate(words) if i not in drop]
        prev_end = 0.0
        for w in out:                      # monotone, without inventing a scale
            if w["start"] < prev_end:
                w["start"] = round(prev_end, 3)
            if w["end"] < w["start"]:
                w["end"] = round(w["start"], 3)
            prev_end = w["end"]
            w["is_last"] = False
        if out:
            out[-1]["is_last"] = True
        rep.n_words_out = len(out)
        rep.backward_after = self._backward(out)
        return out

    # ---------------- top level ---------------- #
    def transcribe(self, audio: str | Path) -> tuple[list[dict], dict]:
        rep = GatorReport()
        with tempfile.TemporaryDirectory() as td:
            wav, gain, rms = self._prepare(Path(audio), Path(td))
            rep.chunks_total = 1
            if wav is None:
                rep.chunks_gated = 1
                return [], rep.to_dict()
            rep.gain_applied_db.append(round(gain, 1))
            text = self._post(wav)
        words = self.parse(text, self.cfg.mode)
        rep.n_words_raw = len(words)
        out = self.chomp(words, rep) if self.cfg.chomp else words
        rep.n_words_out = len(out)
        return out, rep.to_dict()


def main() -> None:
    ap = argparse.ArgumentParser(description="granite-gator: gate quiet audio, chomp hallucinations")
    ap.add_argument("--audio", help="audio file to transcribe")
    ap.add_argument("--in", dest="src", help="existing .asr.json to post-process only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report")
    ap.add_argument("--endpoint", default=GatorConfig.endpoint)
    ap.add_argument("--lumen", action="store_true")
    ap.add_argument("--api-key")
    ap.add_argument("--mode", default="timestamps", choices=["timestamps", "plain"])
    ap.add_argument("--gate-dbfs", type=float, default=GatorConfig.gate_dbfs)
    ap.add_argument("--target-dbfs", type=float, default=GatorConfig.target_dbfs)
    ap.add_argument("--no-normalize", action="store_true")
    ap.add_argument("--no-chomp", action="store_true")
    ap.add_argument("--keep-repeats", type=int, default=GatorConfig.keep_repeats)
    args = ap.parse_args()

    gator = GraniteGator(GatorConfig(
        endpoint=args.endpoint, lumen=args.lumen, api_key=args.api_key, mode=args.mode,
        gate_dbfs=args.gate_dbfs, target_dbfs=args.target_dbfs,
        normalize=not args.no_normalize, chomp=not args.no_chomp,
        keep_repeats=args.keep_repeats))

    if args.src:
        words = json.loads(Path(args.src).read_text(encoding="utf-8"))
        rep = GatorReport(n_words_raw=len(words))
        out = gator.chomp(words, rep)
        report = rep.to_dict()
    elif args.audio:
        out, report = gator.transcribe(args.audio)
    else:
        ap.error("need --audio or --in")

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "loops"}, indent=2))
    if report["n_loops"]:
        worst = report["loops"][0]
        print(f"worst loop: {worst['cycle']!r} x{worst['repeats']} @{worst['start_s']:.1f}s")


if __name__ == "__main__":
    main()
