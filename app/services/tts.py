"""
Convert text to speech using ElevenLabs. Returns audio bytes (mp3).
Streaming API yields chunks for lower latency.
"""
from typing import Iterator

from elevenlabs import ElevenLabs

from app.config import ELEVENLABS_API_KEY, ELEVENLABS_VOICE

_client = None  # ElevenLabs


def get_elevenlabs_client() -> ElevenLabs:
    global _client
    if _client is None:
        if not ELEVENLABS_API_KEY:
            raise RuntimeError("ELEVENLABS_API_KEY is not set in environment")
        _client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _client


def text_to_speech_stream(text: str) -> Iterator[bytes]:
    """Generate speech audio from text as a stream of mp3 chunks."""
    if not text.strip():
        return
    client = get_elevenlabs_client()
    for chunk in client.text_to_speech.stream(
        voice_id=ELEVENLABS_VOICE,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    ):
        yield chunk


def text_to_speech(text: str) -> bytes:
    """Generate speech audio from text. Returns mp3 bytes."""
    if not text.strip():
        return b""
    client = get_elevenlabs_client()
    # convert() may return bytes or a stream; normalize to bytes
    result = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    if isinstance(result, bytes):
        return result
    if hasattr(result, "read"):
        return result.read()
    # Generator/iterator of chunks
    return b"".join(result)
