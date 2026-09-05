"""
Test suite for Module 2 (Voice Spoof / Deepfake Detection).

Run:
    pytest -v

Covers three layers:
1. Unit tests on VoiceSpoofDetector directly (no server needed)
2. Edge cases: silence, short clips, different sample rates, stereo, bad input
3. API contract tests via FastAPI's TestClient (no need to actually run uvicorn)

The API contract tests are the important ones for teammates: they lock in
the exact response shape the fusion module (module 5) will depend on, so
if you accidentally change a field name/type later, these tests catch it
before it breaks someone else's integration.
"""

import pytest
from fastapi.testclient import TestClient

from inference import VoiceSpoofDetector
from service import app

TESTS_DIR = "tests"


# ---------- Fixtures ----------

@pytest.fixture(scope="module")
def detector():
    # Loaded once per test module run, not per test, since model loading is slow.
    return VoiceSpoofDetector(variant="aasist-l")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------- 1. Unit tests: does the model produce sane output shapes? ----------

def test_detector_loads():
    d = VoiceSpoofDetector(variant="aasist-l")
    assert d.variant == "aasist-l"


def test_score_returns_expected_keys(detector):
    result = detector.score_from_file(f"{TESTS_DIR}/synthetic_voice_like.wav")
    assert set(result.keys()) == {"risk_score_voice", "risk_label", "variant"}


def test_score_is_in_valid_range(detector):
    result = detector.score_from_file(f"{TESTS_DIR}/synthetic_voice_like.wav")
    assert 0.0 <= result["risk_score_voice"] <= 1.0


def test_score_is_deterministic(detector):
    # Same input should give the same output every time (model is in eval mode,
    # no dropout/randomness at inference). Important since flaky scores would
    # be very hard for the fusion module to reason about.
    r1 = detector.score_from_file(f"{TESTS_DIR}/synthetic_voice_like.wav")
    r2 = detector.score_from_file(f"{TESTS_DIR}/synthetic_voice_like.wav")
    assert r1["risk_score_voice"] == r2["risk_score_voice"]


# ---------- 2. Edge cases the real pipeline WILL hit eventually ----------

def test_silence_does_not_crash(detector):
    # Call audio often has silent gaps (pauses in speech). Must not throw.
    result = detector.score_from_file(f"{TESTS_DIR}/edge_silence.wav")
    assert 0.0 <= result["risk_score_voice"] <= 1.0


def test_short_clip_shorter_than_model_window(detector):
    # 0.5s clip vs model's ~4.04s expected window -> exercises the tiling/padding path
    result = detector.score_from_file(f"{TESTS_DIR}/edge_short_clip.wav")
    assert 0.0 <= result["risk_score_voice"] <= 1.0


def test_non_16khz_sample_rate_is_resampled(detector):
    # Android mic capture commonly delivers 44.1kHz, not the 16kHz the model expects.
    result = detector.score_from_file(f"{TESTS_DIR}/edge_44100hz.wav")
    assert 0.0 <= result["risk_score_voice"] <= 1.0


def test_stereo_input_is_downmixed(detector):
    result = detector.score_from_file(f"{TESTS_DIR}/edge_stereo.wav")
    assert 0.0 <= result["risk_score_voice"] <= 1.0


def test_corrupted_audio_raises_cleanly(detector):
    # Simulates a truncated/corrupted chunk from a flaky Wi-Fi upload.
    garbage_bytes = b"this is not a valid audio file at all"
    with pytest.raises(Exception):
        detector.score_from_bytes(garbage_bytes)


# ---------- 3. API contract tests (this is what module 5 / fusion depends on) ----------

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_voice_risk_endpoint_happy_path(client):
    with open(f"{TESTS_DIR}/synthetic_voice_like.wav", "rb") as f:
        resp = client.post("/voice-risk", files={"file": ("clip.wav", f, "audio/wav")})
    assert resp.status_code == 200
    body = resp.json()
    # Lock in the exact contract other modules will integrate against.
    assert set(body.keys()) == {"risk_score_voice", "risk_label", "variant", "chunk_id", "latency_ms"}
    assert isinstance(body["risk_score_voice"], float)
    assert 0.0 <= body["risk_score_voice"] <= 1.0
    assert body["risk_label"] in {"low", "medium", "high"}


def test_voice_risk_endpoint_echoes_chunk_id(client):
    with open(f"{TESTS_DIR}/synthetic_voice_like.wav", "rb") as f:
        resp = client.post(
            "/voice-risk",
            files={"file": ("clip.wav", f, "audio/wav")},
            data={"chunk_id": "abc-123"},
        )
    assert resp.status_code == 200
    assert resp.json()["chunk_id"] == "abc-123"


def test_voice_risk_endpoint_rejects_bad_file(client):
    resp = client.post(
        "/voice-risk",
        files={"file": ("bad.wav", b"not audio data", "audio/wav")},
    )
    assert resp.status_code == 400


def test_voice_risk_endpoint_latency_is_reasonable(client):
    # Soft check: on CPU this should stay well under a 5-second chunk budget.
    # Tune this threshold once you've benchmarked on real deployment hardware.
    with open(f"{TESTS_DIR}/synthetic_voice_like.wav", "rb") as f:
        resp = client.post("/voice-risk", files={"file": ("clip.wav", f, "audio/wav")})
    body = resp.json()
    assert body["latency_ms"] < 5000
