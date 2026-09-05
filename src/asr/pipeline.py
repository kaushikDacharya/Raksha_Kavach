from src.asr.audio_utils import split_audio
from src.asr.transcriber import ASRTranscriber

def process_audio(audio_path):
    chunks = split_audio(
        audio_path,
        "temp_chunks"
    )

    transcriber = ASRTranscriber()

    transcripts = []

    for chunk in chunks:
        result = transcriber.transcribe(chunk)

        transcripts.append(result)

    return transcripts