"""
Raksha Kavach - Scam Script Analysis Engine
Uses Google Gemini to analyse text for scam indicators.
Outputs clean JSON only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_configured_api_keys() -> list[str]:
    """Retrieve all available Gemini API keys from environment."""
    keys = []
    for var in ("GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3"):
        val = os.getenv(var)
        if val and val.strip() and val.strip() not in keys:
            keys.append(val.strip())

    # Also support comma-separated GEMINI_API_KEYS
    if extra := os.getenv("GEMINI_API_KEYS"):
        for k in extra.split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)

    return keys


_current_key_idx = 0


def _get_next_api_key(keys: list[str]) -> tuple[int, str]:
    """Get next key in round-robin order."""
    global _current_key_idx
    if not keys:
        raise RuntimeError(
            "No GEMINI_API_KEY found. Set GEMINI_API_KEY in .env or as environment variable."
        )
    idx = _current_key_idx % len(keys)
    _current_key_idx = (_current_key_idx + 1) % len(keys)
    return idx, keys[idx]


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
]


# Data Models
class Verdict(str, Enum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_SCAM = "LIKELY SCAM"
    SCAM = "SCAM"


@dataclass
class ScamAnalysisResult:
    """Structured result from the Gemini scam analysis."""

    scam_percentage: float
    verdict: str
    reasons: list[str]
    red_flag_phrases: list[str]
    safety_advice: list[str]
    category: str  # e.g. "phishing", "tech-support", "romance", etc.
    summary: str
    call_ended: bool = False
    raw_response: Optional[str] = field(default=None, repr=False)
    analysed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (excludes raw_response)."""
        return {
            "scam_percentage": self.scam_percentage,
            "verdict": self.verdict,
            "category": self.category,
            "summary": self.summary,
            "reasons": self.reasons,
            "red_flag_phrases": self.red_flag_phrases,
            "safety_advice": self.safety_advice,
            "call_ended": self.call_ended,
            "analysed_at": self.analysed_at,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# End of Call Text Detection
END_OF_CALL_PATTERNS = [
    re.compile(r"\[?(?:call\s*(?:ended|terminated|finished|disconnected|completed)|end\s*of\s*call|hung\s*up|hang\s*up)\]?", re.IGNORECASE),
    re.compile(r"(?:caller|agent|user)\s+(?:hung\s*up|disconnected|ended\s+the\s+call)", re.IGNORECASE),
    re.compile(r"\[(?:call_ended|end_of_call|call_over)\]", re.IGNORECASE),
]


def is_end_of_call_text(text: str) -> bool:
    """Check whether text contains an explicit end-of-call marker or message."""
    for pattern in END_OF_CALL_PATTERNS:
        if pattern.search(text):
            return True
    return False


# Gemini Integration
SYSTEM_PROMPT = textwrap.dedent("""\
    You are Raksha Kavach, an expert scam-detection analyst.

    Analyse the user-provided text and determine whether it is a scam, fraud, or
    social-engineering attempt.

    Return your analysis as valid JSON only (no markdown fences) matching this
    exact schema:

    {
      "scam_percentage": <float 0-100>,
      "verdict": "<SAFE | SUSPICIOUS | LIKELY SCAM | SCAM>",
      "category": "<phishing | tech-support | romance | lottery | investment | impersonation | advance-fee | job | other>",
      "summary": "<one-paragraph summary>",
      "reasons": ["<reason 1>", "<reason 2>", ...],
      "red_flag_phrases": ["<exact quote from text>", ...],
      "safety_advice": ["<actionable tip 1>", ...],
      "call_ended": <true | false>
    }

    Guidelines:
    - scam_percentage 0-29 -> verdict SAFE
    - scam_percentage 30-64 -> verdict SUSPICIOUS
    - scam_percentage 65-89 -> verdict LIKELY SCAM
    - scam_percentage 90-100 -> verdict SCAM
    - Provide at least 3 reasons.
    - Quote exact phrases from the text that are red flags.
    - Give at least 2 pieces of safety advice.
    - Set call_ended to true if the text indicates the conversation or call has concluded (farewells, hang up, disconnect, end-of-call message), otherwise false.
    - Be thorough but concise.
""")


def _call_gemini(text: str) -> str:
    """Send the text to Gemini and return the raw response text.
    Rotates through available API keys and models with automatic failover."""

    keys = get_configured_api_keys()
    if not keys:
        raise RuntimeError("No GEMINI_API_KEY found. Set GEMINI_API_KEY in .env.")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Analyse this text for scam indicators:\n\n{text}"}],
            }
        ],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.8,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
        },
    }

    # Start rotation from next key
    start_key_idx, _ = _get_next_api_key(keys)
    num_keys = len(keys)

    last_error = None

    # Try each available API key in round-robin order
    for offset in range(num_keys):
        current_key = keys[(start_key_idx + offset) % num_keys]

        # For each key, try available models
        for model in GEMINI_MODELS:
            url = f"{GEMINI_BASE_URL}/{model}:generateContent"
            try:
                resp = requests.post(
                    url,
                    params={"key": current_key},
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60,
                )
                resp.raise_for_status()

                data = resp.json()
                try:
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        f"Unexpected Gemini response structure: {json.dumps(data, indent=2)}"
                    ) from exc

            except requests.exceptions.HTTPError as exc:
                last_error = exc
                # If rate limited (429) or forbidden, immediately failover to next key
                if resp.status_code in (429, 403):
                    break  # Break inner model loop to try next API key
                continue  # Try next model on other errors

    raise RuntimeError(f"All Gemini models and API keys failed. Last error: {last_error}")


def _parse_response(raw: str) -> dict:
    """Extract JSON from Gemini's response (handles markdown fences)."""

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse Gemini JSON response:\n{raw}") from exc


# Public API
def analyse_script(text: str) -> ScamAnalysisResult:
    """
    Analyse a text/script for scam indicators using Gemini.

    Parameters
    ----------
    text : str
        The text or script to analyse.

    Returns
    -------
    ScamAnalysisResult
        Structured analysis with percentage, verdict, reasons, call_ended, etc.
    """

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")

    raw = _call_gemini(text)
    parsed = _parse_response(raw)

    call_ended_parsed = bool(parsed.get("call_ended", False))
    call_ended_detected = call_ended_parsed or is_end_of_call_text(text)

    return ScamAnalysisResult(
        scam_percentage=float(parsed.get("scam_percentage", 0)),
        verdict=parsed.get("verdict", "UNKNOWN"),
        category=parsed.get("category", "other"),
        summary=parsed.get("summary", ""),
        reasons=parsed.get("reasons", []),
        red_flag_phrases=parsed.get("red_flag_phrases", []),
        safety_advice=parsed.get("safety_advice", []),
        call_ended=call_ended_detected,
        raw_response=raw,
    )


def analyse_file(filepath: str | Path, save_report: bool = True) -> ScamAnalysisResult:
    """Read a text file once, analyse it, and optionally save a .scam_report.json."""

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    result = analyse_script(path.read_text(encoding="utf-8"))

    if save_report:
        report_path = path.with_suffix(".scam_report.json")
        report_data = {"source_file": str(filepath), **result.to_dict()}
        report_path.write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return result


def monitor_file(
    filepath: str | Path,
    interval: float = 5.0,
    scam_threshold: float = 80.0,
    save_report: bool = True,
) -> ScamAnalysisResult | None:
    """
    Continuously monitor a transcript file as it grows during a live call.
    Periodically checks the file every `interval` seconds.
    Stops when:
    1. scam_percentage reaches `scam_threshold` (default: 80.0%), OR
    2. An end-of-call message/marker is detected.
    """

    path = Path(filepath)
    last_text = ""
    last_result: ScamAnalysisResult | None = None

    while True:
        if not path.exists():
            time.sleep(interval)
            continue

        try:
            current_text = path.read_text(encoding="utf-8").strip()
        except Exception:
            time.sleep(interval)
            continue

        if not current_text:
            time.sleep(interval)
            continue

        # Skip duplicate analysis if transcript hasn't changed since last check
        if current_text == last_text:
            time.sleep(interval)
            continue

        last_text = current_text
        result = analyse_script(current_text)
        last_result = result

        if save_report:
            report_path = path.with_suffix(".scam_report.json")
            report_data = {"source_file": str(filepath), **result.to_dict()}
            report_path.write_text(
                json.dumps(report_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Output JSON result and immediately flush stdout for consumers
        print(result.to_json())
        sys.stdout.flush()

        # Stop condition 1: Scam percentage reached threshold
        if result.scam_percentage >= scam_threshold:
            break

        # Stop condition 2: Call ended detected
        if result.call_ended:
            break

        time.sleep(interval)

    return last_result


def batch_analyse(texts: list[str]) -> list[ScamAnalysisResult]:
    """Analyse multiple texts sequentially."""
    return [analyse_script(t) for t in texts]


# Entry Point
def main() -> None:
    """CLI entry point - outputs only clean JSON."""

    default_threshold = float(os.getenv("SCAM_THRESHOLD", "80.0"))
    default_interval = float(os.getenv("POLL_INTERVAL", "5.0"))

    parser = argparse.ArgumentParser(
        description="Raksha Kavach - Scam Script Analysis Engine (JSON only)"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Path to transcript file to analyse / monitor",
    )
    parser.add_argument(
        "--once",
        "-1",
        action="store_true",
        help="Analyse file once and exit immediately (disable live polling)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=default_interval,
        help=f"Polling interval in seconds for live monitoring (default: {default_interval})",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=default_threshold,
        help=f"Scam percentage threshold to stop monitoring (default: {default_threshold})",
    )

    args = parser.parse_args()

    try:
        if args.file:
            if args.once:
                result = analyse_file(args.file, save_report=True)
                print(result.to_json())
                sys.stdout.flush()
            else:
                monitor_file(
                    filepath=args.file,
                    interval=args.interval,
                    scam_threshold=args.threshold,
                    save_report=True,
                )
        elif not sys.stdin.isatty():
            text = sys.stdin.read()
            result = analyse_script(text)
            print(result.to_json())
            sys.stdout.flush()
        else:
            parser.print_help(file=sys.stderr)
            sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()