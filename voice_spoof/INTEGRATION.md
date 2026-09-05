# Integration Guide: Voice Spoof Detection Module

This document is for **other people on the team** (fusion module owner,
Android developer, whoever wires the backend together) who need to call
this module without reading its internals. If that's not you, see
`README.md` instead.

---

## What this module does

Give it an audio chunk (a few seconds of a phone call). It tells you how
likely that chunk is a spoofed/synthetic/AI-cloned voice rather than a
genuine human voice.

**It does not** do speech-to-text, scam detection, or fusion — it only
scores the audio itself. That happens in modules 3, 4, and 5 respectively.

**It does not run on the Android app / inside the APK.** It's Python, and
the model (AASIST-L) is a PyTorch neural network — it runs as a backend
service (or backend library), and the Android app / other backend code
talks to it over HTTP or a direct Python import. See "Deployment pattern"
below.

---

## Deployment pattern: pick ONE

### Pattern A — Microservice (recommended, this is what's set up by default)

This module runs as its own small server. Whoever owns fusion (or the
Android networking layer, if there's no separate fusion service) calls
its HTTP endpoint.

```
Android app --(HTTP)--> Backend/Fusion --(HTTP)--> THIS MODULE (port 8001)
```

Start it:
```bash
cd modules/voice_spoof
uvicorn service:app --host 0.0.0.0 --port 8001
```

### Pattern B — Direct Python import (if the team builds one monolith backend)

If fusion and this module end up living in the same Python process
(one FastAPI app for the whole backend), skip the HTTP layer entirely:

```python
from voice_spoof import VoiceSpoofDetector

detector = VoiceSpoofDetector(variant="aasist-l")  # load ONCE, reuse across requests
result = detector.score_from_file("chunk.wav")
# or: result = detector.score_from_bytes(audio_bytes)
```

Either pattern returns the exact same fields — see "Response contract" below.

---

## API Contract (Pattern A - HTTP)

### `POST /voice-risk`

**Request:** `multipart/form-data`

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | file | yes | The audio chunk. WAV/FLAC/OGG read natively. M4A/MP3/3GP/AMR (common on Android) work too, IF `ffmpeg` is installed on the machine running this service — see "Audio format support" below. |
| `chunk_id` | string | no | Any identifier you want echoed back, useful for matching async responses to requests. |

**Response:** `200 OK`, `application/json`

```json
{
  "risk_score_voice": 0.87,
  "risk_label": "high",
  "variant": "aasist-l",
  "chunk_id": "abc-123",
  "latency_ms": 812.4
}
```

| Field | Type | Meaning |
|---|---|---|
| `risk_score_voice` | float, 0.0–1.0 | Higher = more likely spoofed/synthetic. **This is the field the fusion module should consume.** |
| `risk_label` | string | `"low"` / `"medium"` / `"high"` — a pre-thresholded convenience label, in case you don't want to pick your own cutoffs. Thresholds live in `config.py`'s `RISK_THRESHOLDS` and are starting defaults, not validated ground truth yet. |
| `variant` | string | Which model produced the score (`"aasist-l"` by default). |
| `chunk_id` | string or null | Echoes back whatever you sent, or `null` if you didn't send one. |
| `latency_ms` | float | How long inference took, for your own monitoring/debugging. |

**Error response:** `400 Bad Request`
```json
{ "detail": "human-readable reason the audio couldn't be processed" }
```
Treat any non-200 response as "could not score this chunk" — e.g. skip it
in fusion, or fall back to a neutral/default risk value, rather than
crashing the pipeline over one bad chunk.

### `GET /health`

Returns `{"status": "ok", "model_variant": "aasist-l"}`. Use this for
startup checks / liveness probes, not for scoring.

---

## Example client code

### curl (quick manual test)
```bash
curl -X POST \
  -F "file=@chunk.wav" \
  -F "chunk_id=call123-chunk007" \
  http://localhost:8001/voice-risk
```

### Python (e.g. from the fusion module, Pattern A)
```python
import requests

with open("chunk.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8001/voice-risk",
        files={"file": f},
        data={"chunk_id": "call123-chunk007"},
    )
resp.raise_for_status()
voice_risk = resp.json()["risk_score_voice"]
```

### Kotlin / Retrofit (if the Android app or its backend layer calls this directly)
```kotlin
interface VoiceRiskApi {
    @Multipart
    @POST("/voice-risk")
    suspend fun getVoiceRisk(
        @Part file: MultipartBody.Part,
        @Part("chunk_id") chunkId: RequestBody
    ): VoiceRiskResponse
}

data class VoiceRiskResponse(
    val risk_score_voice: Double,
    val risk_label: String,
    val variant: String,
    val chunk_id: String?,
    val latency_ms: Double
)
```

---

## Audio format support

| Format | Support |
|---|---|
| WAV, FLAC, OGG | Native, always works, fastest |
| M4A, MP3, 3GP, AMR, AAC | Works IF `ffmpeg` is installed on the machine running this service (`ffmpeg -version` to check). This covers most raw Android call-recording formats. |
| Anything else | Will return a `400` with a clear error message rather than crashing. |

Sample rate, mono/stereo, and clip length are all handled automatically
(resampled to 16kHz, downmixed to mono, padded/tiled to the model's
expected window) — **you do not need to preprocess audio before sending it.**

---

## Performance expectations

- ~0.7–1.3 seconds per chunk on CPU (tested in a sandboxed environment;
  benchmark again on your actual deployment hardware).
- If chunks arrive every 5 seconds (per the pipeline design) and this
  is comfortably under that, you're fine for real-time use as-is.
- If you need faster: batch multiple chunks into one call, or move to GPU
  inference — talk to whoever owns this module before changing the model
  itself (swapping to a smaller/quantized model changes accuracy).

## What NOT to assume

- Don't assume `risk_score_voice == 0` means "definitely human" — it means
  "this model found no spoof indicators," which isn't the same guarantee.
- Don't hardcode the `RISK_THRESHOLDS` cutoffs into your own code — read
  `risk_label` instead, so if thresholds get tuned later, you don't need
  a code change too.
- Don't call `VoiceSpoofDetector(...)` per-request in Pattern B — construct
  it once at startup and reuse it, or you'll reload the model from disk
  on every single chunk.
