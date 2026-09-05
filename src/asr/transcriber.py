import whisper


class ASRTranscriber:
    def __init__(self, model_name="small"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)

        return {
            "text": result["text"].strip(),
            "language": result.get("language")
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python transcriber.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    transcriber = ASRTranscriber()
    result = transcriber.transcribe(audio_file)

    print("Language:", result["language"])
    print("Transcript:", result["text"])
