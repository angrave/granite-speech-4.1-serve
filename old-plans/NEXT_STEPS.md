# Next Steps / Handoff Notes

This document captures the current state of the repo, known issues, and what still
needs doing. Written after the native Mac (`start_apple_dockerless.sh`) work session.

---

## Current state

### What works

| Service | Path | Status |
|---------|------|--------|
| `granite-plus` (port 8001) | `start_apple_dockerless.sh` → PyTorch MPS | ✅ Tested, healthy |
| `granite-nar` (port 8002) | `start_apple_dockerless.sh` → PyTorch MPS | ✅ Tested, healthy |
| `granite-plus` + `granite-nar` | Docker (CPU / CUDA) | ✅ CI builds, images published to ghcr.io |
| `granite-base` (port 9797) | Docker via `ghcr.io/ggml-org/llama.cpp:server` | ⚠️ See below |
| `granite-base` (port 9797) | `start_apple_dockerless.sh` → llama.cpp source build | ✅ Tested, healthy |

### What's broken / unverified

**`granite-base` — llama.cpp `granite_speech` projector support missing**

The Granite Speech 4.1 base model uses a multimodal audio projector type called
`granite_speech`. Neither the Homebrew-packaged `llama-server` nor the official
`ghcr.io/ggml-org/llama.cpp:server` Docker image support it yet (as of the time
of writing). The error seen at runtime is:

```
clip_init: failed to load model '...mmproj-model-f16.gguf':
load_hparams: unknown projector type: granite_speech
```

The fix for the Mac native path is a source build, which `start_apple_dockerless.sh`
now handles automatically. The Docker path is still broken (see below).

---

## Immediate things to verify

### 1. Confirm the llama.cpp source build works — ✅ DONE (2026-06-13)

Source build completed. All three servers start and serve transcription correctly.

**Note on the `granite_speech` strings check:** The string is in `libmtmd.dylib`,
not the main `llama-server` binary. `start_apple_dockerless.sh` and the CI
workflow have been updated to check the dylib via the binary's embedded `@rpath`.

```bash
# Verified working:
strings .llama_build/build/bin/llama-server | grep granite_speech  # returns nothing
otool -l .llama_build/build/bin/llama-server | grep -A2 LC_RPATH   # → .llama_build/src/build/bin
strings .llama_build/src/build/bin/libmtmd.0.dylib | grep granite_speech  # ✓ found
```

### 2. Test end-to-end transcription for plus and NAR — ✅ DONE (2026-06-13)

All three models tested and working on macOS Apple Silicon. Results:

All return correct transcriptions. Multi-speaker diarisation and word timestamps also work on `granite-plus`.

### 3. Verify the GitHub Action runs cleanly

The workflow `.github/workflows/llama-cpp-mac.yml` has been updated (2026-06-13):

- **Fixed:** "Verify granite_speech support" step now also checks `libmtmd*.dylib`
  (the string is in the dylib, not the main binary — see note in §1 above)
- **Fixed:** Artifact now bundles `llama-server` + all `lib*.dylib` files into
  `llama-server-macos-arm64.tar.gz`. The binary is useless without the dylibs.

After pushing, confirm in the Actions tab:
- `macos-14` runner builds successfully
- "Verify granite_speech support" step passes
- Artifact `llama-server-macos-arm64` (a .tar.gz) is uploaded

Once confirmed, update `start_apple_dockerless.sh` to download the pre-built
tarball instead of compiling (saves ~10 min on first run). See "Future improvements".

### 4. Fix granite-base in Docker

The official `ghcr.io/ggml-org/llama.cpp:server` image almost certainly also lacks
`granite_speech` support. Options:

**Option A — wait for the official image to update** (easiest, no action needed)  
Monitor `https://github.com/ggml-org/llama.cpp` for a release that includes
`granite_speech` projector support, then the existing docker-compose.yml will
just work.

**Option B — build our own llama.cpp Docker image**  
Add a new Dockerfile (e.g. `Dockerfile.llama`) that builds `llama-server` from
source, and publish it via a new GitHub Action job. Update `docker-compose.yml`
to use `ghcr.io/angrave/granite-speech-4.1-serve:llama-server` instead of the
official image. The existing `llama-cpp-mac.yml` workflow logic is a good starting
point for the Linux/Docker build.

---

## Known workarounds / compatibility patches in the code

These were added in this session to work around bugs in upstream code. They should
be revisited or removed once the upstream issues are fixed.

### `serve_nar.py` — `PreTrainedConfig` alias

```python
import transformers.configuration_utils as _cu
if not hasattr(_cu, "PreTrainedConfig") and hasattr(_cu, "PretrainedConfig"):
    _cu.PreTrainedConfig = _cu.PretrainedConfig
```

**Why:** The NAR model's HuggingFace remote code (`configuration_granite_speech_nar.py`)
imports `PreTrainedConfig` (capital T) from `transformers.configuration_utils`, but
current versions of `transformers` only expose `PretrainedConfig` (lowercase t) there.
This is a bug in IBM's published model code.

**When to remove:** Once IBM updates the remote code on HuggingFace to use the
correct casing (or `from transformers import PretrainedConfig`), this patch can be
deleted.

### `serve_plus.py` — `trust_remote_code=True` on model load

```python
_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID, trust_remote_code=True, ...
)
```

**Why:** The `granite_speech_plus` model type is not registered in the transformers
AutoModel registry, so without `trust_remote_code=True` it throws `KeyError:
'granite_speech_plus'`. The processor does NOT need `trust_remote_code`.

**When to remove / change:** Once `granite_speech_plus` is added to the transformers
registry natively (i.e. `pip install transformers` includes it), `trust_remote_code`
can be dropped.

### Python 3.10+ requirement in `start_apple_dockerless.sh`

**Why:** The NAR model's remote code uses `int | None` union-type syntax introduced
in Python 3.10. System Python on macOS is 3.9.

**When to remove:** If/when IBM updates the remote code to be compatible with
Python 3.9 (using `Optional[int]` from `typing`), the version gate can drop back
to 3.9. For now, 3.11 or 3.14 are both confirmed working.

---

## README improvements to make

The README (`README.md`) is functional but could be improved:

1. **Add a troubleshooting section** covering:
   - `unknown projector type: granite_speech` → need newer llama.cpp
   - `KeyError: 'granite_speech_plus'` → need `trust_remote_code=True` (already fixed)
   - `int | None` TypeError → Python 3.9, need 3.10+
   - `preprocessor_config.json not found` → was caused by adding `trust_remote_code`
     to the processor call (don't do this)

2. **Document the `start_apple_dockerless.sh` source-build behaviour** — currently
   the README just says the script lazy-installs dependencies; it should mention
   that the base model may trigger a 10-minute source build of llama.cpp.

3. **Add a "Verified working environment" section** listing:
   - macOS Sequoia, Apple M3 Ultra, Python 3.14.6, transformers 4.57.6
   - What was tested and when

4. **Update the llama.cpp version note** — currently says "build from source or
   wait for Homebrew to update". Should also mention the GitHub Actions artifact
   as a download option once that's confirmed working.

5. **Note the Docker base model limitation** more prominently — currently buried
   in a comment in `docker-compose.yml`.

---

## Future improvements

### Download pre-built llama-server from GitHub Releases

Instead of always building from source (~10 min), `start_apple_dockerless.sh`
could try to download a pre-built binary from the latest GitHub Release first:

```bash
# Pseudocode
RELEASE_URL=$(gh release view --json assets -q '.assets[] | select(.name=="llama-server-macos-arm64") | .url')
curl -L -o "$LLAMA_LOCAL" "$RELEASE_URL"
chmod +x "$LLAMA_LOCAL"
```

This requires `gh` CLI to be installed, or using the GitHub API directly. Falls
back to source build if no release asset exists.

### Pin the NAR model's remote code revision

The NAR model uses `trust_remote_code=True` which downloads and executes code from
HuggingFace on every run. The `transformers` warning about this is real — pinning
to a known-good revision avoids surprise breakage:

```python
AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, revision="99a4df9")
```

(The revision `99a4df9` is the commit that was tested and works with Python 3.14.6.)

### Add `test_transcription.sh`

A simple script that starts the servers (or assumes they're running) and runs a
quick transcription test against all three, checking for non-empty output. Useful
for smoke-testing after any dependency change.

### Consider pinning `transformers` version

`requirements.txt` currently says `transformers>=4.52.1`, which got us 4.57.6.
That version drops `PreTrainedConfig` from `configuration_utils`, causing the NAR
monkeypatch. If stability matters more than latest features, pin to a specific
version (e.g. `transformers==4.52.4`) and document why.
