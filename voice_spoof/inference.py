"""
Voice Spoof / Deepfake Detection - Module 2 of Raksh Kavach.

Wraps the pretrained AASIST / AASIST-L checkpoint (from clovaai/aasist,
MIT licensed) for inference on short audio chunks.

This module is a self-contained black box:
    input  -> a path to an audio file, or raw audio bytes
    output -> a risk_score_voice float in [0, 1], plus a human-readable
              risk_label, with NO dependency on how the audio got there
              or what happens to the score afterwards.

See INTEGRATION.md for how other modules / the backend are meant to call this.
"""

import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from aasist_arch import Model as AasistModel
from config import (
    AASIST_L_CONFIG,
    AASIST_FULL_CONFIG,
    CHECKPOINTS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    RISK_THRESHOLDS,
)

BASE_DIR = Path(__file__).parent


class UnsupportedAudioError(Exception):
    """Raised when the input audio can't be decoded by soundfile OR ffmpeg."""


def _pad_or_tile(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Match the official repo's approach: tile (repeat) short clips rather
    than zero-padding, since zero-padding introduces silence the model
    wasn't trained on and hurts accuracy. Trims long clips to target_len.
    """
    cur_len = x.shape[0]
    if cur_len == 0:
        raise UnsupportedAudioError("Audio contains zero samples.")
    if cur_len >= target_len:
        return x[:target_len]
    num_repeats = int(target_len / cur_len) + 1
    tiled = np.tile(x, num_repeats)
    return tiled[:target_len]


def _ffmpeg_decode_to_wav_bytes(input_bytes: bytes) -> bytes:
    """
    Fallback decoder for formats soundfile can't read directly - notably
    the compressed formats Android commonly records/sends: .m4a, .3gp, .aac,
    .amr, .mp3. Requires ffmpeg to be installed on the machine running this
    module (it is NOT a pip dependency).
    """
    if shutil.which("ffmpeg") is None:
        raise UnsupportedAudioError(
            "Audio format not readable by soundfile, and ffmpeg is not installed "
            "to fall back on. Install ffmpeg (e.g. `brew install ffmpeg` / "
            "`apt install ffmpeg`) or convert the file to WAV before sending it."
        )
    with tempfile.NamedTemporaryFile(suffix=".input") as in_f:
        in_f.write(input_bytes)
        in_f.flush()
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", in_f.name,
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", "pipe:1",
            ],
            capture_output=True,
        )
    if result.returncode != 0:
        raise UnsupportedAudioError(
            f"ffmpeg could not decode this audio: {result.stderr.decode(errors='ignore')[-300:]}"
        )
    return result.stdout


def _load_waveform(audio_bytes: bytes) -> tuple:
    """
    Returns (waveform: torch.Tensor of shape (channels, samples), sr: int).
    Tries soundfile first (fast path for WAV/FLAC/OGG), falls back to ffmpeg
    for everything else (M4A/AAC/3GP/MP3/AMR - common phone recording formats).
    """
    try:
        wav_np, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    except Exception:
        wav_bytes = _ffmpeg_decode_to_wav_bytes(audio_bytes)
        wav_np, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)

    waveform = torch.from_numpy(wav_np.T)  # -> (channels, samples)
    return waveform, sr


def _risk_label(score: float) -> str:
    if score >= RISK_THRESHOLDS["high"]:
        return "high"
    if score >= RISK_THRESHOLDS["medium"]:
        return "medium"
    return "low"


class VoiceSpoofDetector:
    """
    The black-box detector. Construct once, reuse for every chunk -
    constructing it repeatedly reloads model weights from disk every time,
    which is slow and unnecessary.
    """

    def __init__(self, variant: str = "aasist-l", device: str = "cpu"):
        """
        variant: "aasist-l" (fast, ~85k params, recommended for real-time)
                 or "aasist" (full model, higher accuracy, heavier)
        """
        if variant not in CHECKPOINTS:
            raise ValueError(f"Unknown variant '{variant}', expected one of {list(CHECKPOINTS)}")

        config = AASIST_L_CONFIG if variant == "aasist-l" else AASIST_FULL_CONFIG
        checkpoint_path = BASE_DIR / CHECKPOINTS[variant]

        self.device = torch.device(device)
        self.model = AasistModel(config).to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.variant = variant

    def _preprocess(self, waveform: torch.Tensor, sr: int) -> torch.Tensor:
        """
        waveform: shape (channels, samples).
        Returns a 1D tensor of exactly WINDOW_SAMPLES at 16kHz, mono.
        """
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # downmix to mono
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        wav_np = waveform.numpy().astype(np.float32)
        wav_np = _pad_or_tile(wav_np, WINDOW_SAMPLES)
        return torch.from_numpy(wav_np)

    @torch.no_grad()
    def _score_waveform(self, waveform: torch.Tensor, sr: int) -> dict:
        x = self._preprocess(waveform, sr).unsqueeze(0).to(self.device)  # (1, WINDOW_SAMPLES)
        _last_hidden, logits = self.model(x)  # logits shape (1, 2): [spoof_logit, bonafide_logit]

        probs = torch.softmax(logits, dim=-1)[0]
        prob_spoof = round(probs[0].item(), 4)  # class 0 = spoof (see data_utils.py label convention)

        return {
            "risk_score_voice": prob_spoof,
            "risk_label": _risk_label(prob_spoof),
            "variant": self.variant,
        }

    # ---- Public black-box entry points ----

    def score_from_file(self, filepath: str) -> dict:
        """Accepts any audio format soundfile or ffmpeg can decode (wav, flac, ogg, m4a, mp3, 3gp, amr, ...)."""
        with open(filepath, "rb") as f:
            return self.score_from_bytes(f.read())

    def score_from_bytes(self, audio_bytes: bytes) -> dict:
        """The core entry point every other path (file, API upload, chunk stream) funnels through."""
        waveform, sr = _load_waveform(audio_bytes)
        return self._score_waveform(waveform, sr)
