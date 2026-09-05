"""
try_it_yourself.py - the simplest way to see what this module does with
your own audio file(s). No API, no server, just: give it a file, see the score.

Usage:
    # Test one file
    python try_it_yourself.py my_recording.wav

    # Compare a real voice clip against a suspected fake/TTS clip side by side
    python try_it_yourself.py my_real_voice.wav my_ai_voice.m4a

Accepts wav/flac/ogg natively, and m4a/mp3/3gp/amr if ffmpeg is installed
on your machine (it almost certainly is if you've ever used it before -
check with `ffmpeg -version`).
"""

import sys

from inference import VoiceSpoofDetector, UnsupportedAudioError


def describe(path: str, detector: VoiceSpoofDetector) -> dict:
    try:
        result = detector.score_from_file(path)
        result["file"] = path
        result["error"] = None
    except UnsupportedAudioError as e:
        result = {"file": path, "error": str(e)}
    except FileNotFoundError:
        result = {"file": path, "error": f"File not found: {path}"}
    return result


def print_result(result: dict):
    print(f"\n📁 {result['file']}")
    if result.get("error"):
        print(f"   ❌ {result['error']}")
        return
    bar_len = int(result["risk_score_voice"] * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}[result["risk_label"]]
    print(f"   {icon}  risk_score_voice = {result['risk_score_voice']}  ({result['risk_label'].upper()})")
    print(f"   [{bar}]")
    print(f"   model variant: {result['variant']}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    files = sys.argv[1:]
    print("Loading model (this happens once)...")
    detector = VoiceSpoofDetector(variant="aasist-l")

    results = [describe(f, detector) for f in files]

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    for r in results:
        print_result(r)

    if len(results) == 2 and not any(r.get("error") for r in results):
        a, b = results
        print("\n" + "-" * 50)
        if a["risk_score_voice"] < b["risk_score_voice"]:
            print(f"👉 '{a['file']}' scores LOWER risk than '{b['file']}'.")
            print("   If the first is your real voice and the second is AI-generated,")
            print("   this is the result you want to see.")
        elif b["risk_score_voice"] < a["risk_score_voice"]:
            print(f"👉 '{b['file']}' scores LOWER risk than '{a['file']}'.")
        else:
            print("👉 Both files scored identically - not very informative, try more distinct clips.")

    print("\nReminder: 'risk_label' thresholds (config.py -> RISK_THRESHOLDS) are")
    print("starting defaults, not validated ground truth. Tune them once you've")
    print("tested against enough real speech + real spoofed clips.\n")


if __name__ == "__main__":
    main()
