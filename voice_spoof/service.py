"""
Raksh Kavach - Module 2: Voice Spoof / Deepfake Detection Pipeline
FastAPI service exposing /voice-risk

This is the microservice deployment of the black-box module. See
INTEGRATION.md for the full API contract and example client code
(Python, curl, and Android/Kotlin).

Run:
    uvicorn service:app --host 0.0.0.0 --port 8001

Test:
    curl -X POST -F "file=@tests/synthetic_voice_like.wav" http://localhost:8001/voice-risk
"""

import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from inference import VoiceSpoofDetector, UnsupportedAudioError

app = FastAPI(title="Raksh Kavach - Voice Spoof Detection", version="1.0.0")

# Loaded once at startup, reused across requests. Loading the model per-request
# would blow the latency budget for scoring 5-second real-time chunks.
detector = VoiceSpoofDetector(variant="aasist-l")


class VoiceRiskResponse(BaseModel):
    risk_score_voice: float   # 0.0-1.0, higher = more likely spoofed/synthetic
    risk_label: str           # "low" | "medium" | "high"
    variant: str              # which model variant produced this score
    chunk_id: Optional[str] = None  # echoed back if the caller sent one
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model_variant": detector.variant}


@app.post("/voice-risk", response_model=VoiceRiskResponse)
async def voice_risk(file: UploadFile = File(...), chunk_id: Optional[str] = Form(None)):
    """
    Accepts a multipart/form-data POST with:
      - file: the audio chunk (wav/flac/ogg natively; m4a/mp3/3gp/amr via
              ffmpeg fallback if ffmpeg is installed on this machine)
      - chunk_id: optional caller-supplied identifier, echoed back unchanged
                  so the caller can match responses to requests when calls
                  are made concurrently/out of order.
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty file upload.")

        start = time.time()
        result = detector.score_from_bytes(audio_bytes)
        result["latency_ms"] = round((time.time() - start) * 1000, 1)
        result["chunk_id"] = chunk_id
        return result

    except UnsupportedAudioError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}")
