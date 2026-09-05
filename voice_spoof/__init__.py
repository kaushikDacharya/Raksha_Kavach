"""
Public surface of the voice_spoof module. Other Python code (e.g. a
monolith backend, or the fusion module) should import from here rather
than reaching into inference.py's internals directly:

    from voice_spoof import VoiceSpoofDetector

    detector = VoiceSpoofDetector()          # loads model once
    result = detector.score_from_file("chunk.wav")
    # result = {"risk_score_voice": 0.12, "risk_label": "low", "variant": "aasist-l"}
"""

from .inference import VoiceSpoofDetector, UnsupportedAudioError

__all__ = ["VoiceSpoofDetector", "UnsupportedAudioError"]
