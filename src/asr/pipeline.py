from .audio_utils import split_audio
from .transcriber import ASRTranscriber
from .config import MODEL_NAME, CHUNK_DURATION_MS, CHUNK_OUTPUT_DIR


def process_audio(audio_path):
    chunks = split_audio(
        audio_path,
        CHUNK_OUTPUT_DIR,
        CHUNK_DURATION_MS
    )

    transcriber = ASRTranscriber(MODEL_NAME)

    transcripts = []

    for chunk in chunks:
        result = transcriber.transcribe(chunk)
        transcripts.append(result)

    return transcripts