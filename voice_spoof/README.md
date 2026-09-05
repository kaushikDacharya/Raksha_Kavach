# Module 2: Voice Spoof / Deepfake Detection Pipeline

Part of Raksh Kavach. Takes a short audio chunk (as sent by the audio
capture module) and returns a risk score indicating how likely the voice
is a spoof/clone/synthetic voice rather than a genuine human.

This is a **self-contained black box**: give it audio in, get a risk score
out. It has no dependency on how any other module in the project is built.

**Are you integrating this into another module (fusion, Android backend)?**
Read **`INTEGRATION.md`** instead — it has the API contract and example
client code. This README is for setting up and testing the module itself.

Wraps the pretrained **AASIST-L** model (85k params, fast — used by
default) from the official [clovaai/aasist](https://github.com/clovaai/aasist)
repo (MIT licensed, see `AASIST_LICENSE.txt`). The full **AASIST** model
is also available (`variant="aasist"`) if you want to trade latency for
accuracy later.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Pretrained checkpoints are already included in `checkpoints/` — no
download step needed.

**Optional but recommended:** install `ffmpeg` so the module can accept
compressed audio formats phones commonly produce (M4A, 3GP, AMR, MP3) —
not just WAV:
```bash
brew install ffmpeg        # Mac
sudo apt install ffmpeg    # Ubuntu/Debian
```
Without ffmpeg, only WAV/FLAC/OGG will work; anything else returns a
clear error rather than crashing.

## 1. Test it yourself with your own audio file

This is the fastest way to see what the module actually does:

```bash
python try_it_yourself.py path/to/your_recording.wav
```

Or compare two files side by side (e.g. your real voice vs. an
AI-generated clip):
```bash
python try_it_yourself.py your_real_voice.wav your_ai_voice.m4a
```

You'll get a readable risk score, a low/medium/high label, and a visual bar —
no API, no server, just the model running directly on your file.

**Note on the bundled `tests/*.wav` files:** they're synthetic sine-wave/
noise signals, NOT real speech — they only prove the code runs without
crashing, not that detection is accurate. To actually validate the model:

- Record a few seconds of your own voice (phone voice memo, laptop mic).
- Generate an AI voice clip (Coqui TTS, ElevenLabs free tier, etc.).
- Run both through `try_it_yourself.py` and confirm your real voice scores
  noticeably lower than the AI clip.
- Ideally also test call-quality audio (compressed, background noise) since
  AASIST was trained on clean studio recordings — that's a real accuracy
  gap worth knowing about early.

## 2. Run the automated test suite

```bash
pytest tests/test_module.py -v
```

Covers three layers, all runnable without any teammate's code:

- **Unit tests** — model loads, output shape/range is sane, scoring is deterministic.
- **Edge cases** — silence, clips shorter than the model's ~4s window,
  non-16kHz sample rates, stereo input, corrupted/non-audio bytes.
- **API contract tests** — hit `/health` and `/voice-risk` via FastAPI's
  `TestClient` and assert the exact response shape, so a future change
  that breaks the contract fails a test immediately instead of silently
  breaking whoever integrates with this module.

All 14 tests should pass.

## 3. Run as a service (for integration testing with other modules)

```bash
uvicorn service:app --host 0.0.0.0 --port 8001
```

```bash
curl -X POST -F "file=@tests/synthetic_voice_like.wav" http://localhost:8001/voice-risk
```

Full request/response contract is documented in **`INTEGRATION.md`**.

## 4. Before pushing to GitHub

```bash
cd modules/voice_spoof        # or wherever this sits in your repo
git add .
git status                     # sanity check what's staged
git commit -m "Module 2: voice spoof/deepfake detection (AASIST-L)"
git push origin feature/voice-spoof-detection
```

- Run `pytest tests/test_module.py -v` one more time right before pushing.
- Open a **draft PR** early and link `INTEGRATION.md` in the description
  so whoever builds fusion can start integrating before your PR is "done."
- Don't commit `venv/`, `__pycache__/`, or `.pytest_cache/` — see `.gitignore`.

## Files

| File | Purpose |
|---|---|
| `aasist_arch.py` | Model architecture (copied verbatim from clovaai/aasist) |
| `config.py` | Model hyperparameters, checkpoint paths, risk thresholds |
| `inference.py` | `VoiceSpoofDetector` — the black box: preprocessing, format handling, scoring |
| `__init__.py` | Clean import surface: `from voice_spoof import VoiceSpoofDetector` |
| `service.py` | FastAPI app exposing `POST /voice-risk` and `GET /health` |
| `try_it_yourself.py` | Friendly script to test your own file(s) directly, no API needed |
| `test_inference.py` | Minimal standalone CLI test (single file, raw output) |
| `checkpoints/` | Pretrained `.pth` weights for both model variants |
| `tests/` | Test audio + `test_module.py` (pytest suite) |
| `INTEGRATION.md` | **API contract for other modules / Android team** |
| `README.md` | This file |

## Known limitations / next steps

1. **Latency**: ~0.7–1.3s per chunk on CPU (tested in a sandboxed
   environment — benchmark again on real deployment hardware).
2. **Accuracy on call-quality audio is unvalidated** — see section 1 above.
   ASVspoof training data is clean studio audio; real phone calls are
   compressed and noisier.
3. **Padding strategy**: chunks shorter than ~4.04s (64,600 samples at
   16kHz) are tiled (repeated) rather than zero-padded, matching the
   official repo's approach — silence padding hurts accuracy.
4. **`RISK_THRESHOLDS` in `config.py`** (used to compute `risk_label`) are
   reasonable starting defaults, not validated against real spoofed
   call data yet. Revisit once you've tested against real clips.
5. Swap `variant="aasist-l"` → `variant="aasist"` in `service.py` if you
   want to trade latency for the full model's higher accuracy later.
