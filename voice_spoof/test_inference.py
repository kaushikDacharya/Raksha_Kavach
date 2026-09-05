"""
Quick standalone test: load the AASIST-L model and score a couple of
audio files. Run this before wiring anything into the API/pipeline.

    python test_inference.py tests/synthetic_voice_like.wav
    python test_inference.py tests/synthetic_tone_noisy.wav

NOTE: The bundled synthetic .wav files are placeholder sine/noise signals,
NOT real speech. They only prove the pipeline runs end-to-end without
shape/dtype errors. Before trusting any risk scores, re-run this against
real human speech clips and real TTS/voice-clone clips (see README.md).
"""

import sys
import time

from inference import VoiceSpoofDetector


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_inference.py <path/to/audio.wav> [variant]")
        sys.exit(1)

    filepath = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "aasist-l"

    print(f"Loading {variant} model...")
    detector = VoiceSpoofDetector(variant=variant)

    print(f"Scoring {filepath}...")
    start = time.time()
    result = detector.score_from_file(filepath)
    elapsed_ms = (time.time() - start) * 1000

    print("\n--- Result ---")
    for k, v in result.items():
        print(f"{k}: {v}")
    print(f"inference_latency_ms: {elapsed_ms:.1f}")


if __name__ == "__main__":
    main()
