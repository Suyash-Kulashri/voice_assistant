"""
Speech-to-text using Deepgram.
- Streaming: Deepgram live WebSocket (transcription as user speaks).
- Batch fallback: Deepgram pre-recorded REST (single blob).
"""
import asyncio
import logging
from typing import Optional

from deepgram import AsyncDeepgramClient
from deepgram.listen.v1.types.listen_v1results import ListenV1Results

from app.config import DEEPGRAM_API_KEY, DEEPGRAM_MODEL

# Live streaming requires a model id that supports streaming (e.g. nova-2-general)
DEEPGRAM_LIVE_MODEL = "nova-2-general"

logger = logging.getLogger(__name__)
_client: Optional[AsyncDeepgramClient] = None


def get_deepgram_client() -> AsyncDeepgramClient:
    global _client
    if _client is None:
        if not DEEPGRAM_API_KEY:
            raise RuntimeError("DEEPGRAM_API_KEY is not set in environment")
        _client = AsyncDeepgramClient(api_key=DEEPGRAM_API_KEY)
    return _client


async def transcribe_audio_async(audio_bytes: bytes, encoding: str = "webm") -> str:
    """Transcribe a single audio blob using Deepgram pre-recorded API. Returns transcript text."""
    if not audio_bytes:
        return ""
    client = get_deepgram_client()
    try:
        response = await client.listen.v1.media.transcribe_file(
            request=audio_bytes,
            model=DEEPGRAM_MODEL,
            encoding=encoding,
            punctuate=True,
            smart_format=True,
        )
    except Exception as e:
        logger.warning("Deepgram transcribe_file failed: %s", e)
        return ""
    if not response.results or not response.results.channels:
        return ""
    alternatives = response.results.channels[0].alternatives
    if not alternatives:
        return ""
    return (alternatives[0].transcript or "").strip()


async def run_deepgram_live(
    audio_queue: asyncio.Queue,
    transcript_queue: asyncio.Queue,
    final_result: list,
    *,
    sample_rate: str = "48000",
) -> None:
    """
    Run Deepgram live WebSocket: read audio from audio_queue, send to Deepgram,
    put (transcript, is_final) into transcript_queue; append final transcript to final_result.
    Expects linear16 PCM at the given sample_rate (e.g. 48000 for browser AudioContext).
    """
    client = get_deepgram_client()
    try:
        async with client.listen.v1.connect(
            model=DEEPGRAM_LIVE_MODEL,
            encoding="linear16",
            sample_rate=sample_rate,
            channels="1",
            interim_results="true",
            punctuate="true",
            smart_format="true",
            endpointing="300",
        ) as dg_socket:
            recv_task = asyncio.create_task(_dg_recv_loop(dg_socket, transcript_queue, final_result))
            try:
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:  # sentinel = end of stream
                        await dg_socket.send_finalize()
                        await dg_socket.send_close_stream()
                        # Let recv_task get the final transcript(s) before connection closes
                        try:
                            await asyncio.wait_for(recv_task, timeout=5.0)
                        except asyncio.TimeoutError:
                            recv_task.cancel()
                            try:
                                await recv_task
                            except asyncio.CancelledError:
                                pass
                        break
                    await dg_socket.send_media(chunk)
            finally:
                if not recv_task.done():
                    recv_task.cancel()
                    try:
                        await recv_task
                    except asyncio.CancelledError:
                        pass
    except Exception as e:
        logger.warning("Deepgram live error: %s", e)
        if not final_result:
            final_result.append("")


async def _dg_recv_loop(dg_socket, transcript_queue: asyncio.Queue, final_result: list) -> None:
    """
    Receive loop: sends (text, is_final, speech_final) tuples to transcript_queue.
    is_final = segment complete (phrase done, but user may still be speaking).
    speech_final = user stopped speaking (endpointing detected silence, or finalize was sent).
    """
    try:
        while True:
            msg = await dg_socket.recv()
            if isinstance(msg, ListenV1Results) and msg.channel and msg.channel.alternatives:
                text = (msg.channel.alternatives[0].transcript or "").strip()
                is_final = bool(getattr(msg, "is_final", False))
                speech_final = (
                    bool(getattr(msg, "speech_final", False))
                    or bool(getattr(msg, "from_finalize", False))
                )
                await transcript_queue.put((text, is_final, speech_final))
                if is_final and text:
                    final_result.append(text)
                    logger.debug("Deepgram final (speech_final=%s): %r", speech_final, text)
                elif text:
                    logger.debug("Deepgram interim: %r", text)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning("Deepgram recv error: %s", e)
