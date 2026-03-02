"""
Transcribe audio file to text using local Whisper.
"""
import tempfile
from pathlib import Path

import whisper

from app.config import AUDIO_TEMP_DIR, WHISPER_MODEL

_model = None


def get_whisper_model():
    global _model
    if _model is None:
        _model = whisper.load_model(WHISPER_MODEL)
    return _model


def transcribe_audio(audio_bytes: bytes, suffix: str = ".wav") -> str:
    """Write audio bytes to a temp file and transcribe with Whisper. Returns transcript text."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=AUDIO_TEMP_DIR) as f:
        f.write(audio_bytes)
        path = Path(f.name)

    try:
        model = get_whisper_model()
        result = model.transcribe(str(path), fp16=False)
        return (result.get("text") or "").strip()
    finally:
        path.unlink(missing_ok=True)
