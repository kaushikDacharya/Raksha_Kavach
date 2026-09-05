from pathlib import Path
from pydub import AudioSegment


def split_audio(audio_path, output_dir, chunk_duration=5000):
    audio = AudioSegment.from_file(audio_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks = []

    for start in range(0, len(audio), chunk_duration):
        chunk = audio[start:start + chunk_duration]

        chunk_path = output_dir / f"chunk_{start // chunk_duration:04d}.wav"

        chunk.export(chunk_path, format="wav")
        chunks.append(str(chunk_path))

    return chunks