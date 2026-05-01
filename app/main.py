"""
Phase 3: Voice assistant — WebSocket endpoint.
Flow: audio in -> Deepgram STT -> intent -> context + long-term memory -> filler -> Groq LLM -> TTS.
Supports barge-in (interrupt), domain prompts, context summarization, and filler phrases.
"""
import asyncio
import json
import logging
import queue
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.services.stt import run_deepgram_live, transcribe_audio_async
from app.services.llm import build_messages, get_llm_response, get_llm_response_stream
from app.services.memory import add_turn, clear_memory, get_messages_for_llm
from app.services.tts import text_to_speech_stream
from app.services.intent import classify_intent
from app.services.prompts import get_system_prompt
from app.services.context import get_messages_with_summary
from app.services.long_term_memory import (
    clear_facts,
    extract_and_store_facts_async,
    get_facts,
    format_facts_for_prompt,
)
from app.services.fillers import get_filler_phrase
from app.services.rag import (
    build_rag_context,
    build_sources,
    is_smalltalk_or_greeting,
    is_product_info_query,
    retrieve_context,
)
from app.services.product_matcher import match_product_name
from app.services.tavily_search import has_tavily, tavily_fallback_answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a helpful voice assistant. Keep replies concise and natural for spoken conversation."

SENTENCE_END_RE = re.compile(r"[.!?\n]")
MAX_BUFFER_LEN = 120
NO_CONTEXT_REPLY = "I don't have that specific information in the provided datasheets"
TAVILY_CONFIRM_REPLY = (
    "I do not have information regarding this in my datasheets, "
    "but I can search it on Google (Advanced Energy and DigiKey) if you allow. "
    "Please reply with yes or no."
)
_pending_tavily: dict[str, dict] = {}


def _build_product_suggestion_reply(suggestions):
    if not suggestions:
        return (
            "I could not confidently match that product name from my catalog. "
            "Please provide the exact product or series name."
        )
    top = suggestions[:5]
    lines = [
        "I could not confidently match that product name. Did you mean one of these?",
    ]
    for s in top:
        lines.append(f"- {s}")
    lines.append("Please reply with the exact product/series name from the list.")
    return "\n".join(lines)


def _is_no_context_reply(text: str) -> bool:
    t = (text or "").strip().lower()
    baseline = NO_CONTEXT_REPLY.lower()
    return baseline in t or "i don't have that specific information in the provided datasheets" in t


def _is_yes(text: str) -> bool:
    t = re.sub(r"[^a-z\s]", " ", (text or "").strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    yes_phrases = {
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "please do",
        "go ahead",
        "yes please",
        "search it",
        "search on google",
    }
    if t in yes_phrases:
        return True
    return ("yes" in t or "sure" in t or "okay" in t or "ok" in t) and ("no" not in t)


def _is_no(text: str) -> bool:
    t = re.sub(r"[^a-z\s]", " ", (text or "").strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    no_phrases = {"no", "n", "nope", "not now", "cancel", "do not", "don't"}
    if t in no_phrases:
        return True
    return "no" in t and "yes" not in t


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice assistant server starting.")
    yield
    logger.info("Voice assistant server shutting down.")


app = FastAPI(title="Voice Assistant", lifespan=lifespan)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Voice assistant WebSocket at /ws"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/app")
async def app_page():
    index = STATIC_DIR / "index.html"
    if index.is_file():
        return FileResponse(index)
    return {"message": "Frontend not found. Create static/index.html."}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected to /ws")
    session_id = str(uuid.uuid4())
    audio_encoding = "opus"
    streaming_stt = False
    audio_queue = None
    transcript_queue = None
    final_result = None
    dg_task = None
    transcript_forward_task = None
    reply_aborted = asyncio.Event()
    speculative_task = None
    speculative_transcript = None
    speculative_reply = None
    events = asyncio.Queue()
    ws_receive_task = None

    async def ws_receiver():
        try:
            while True:
                msg = await websocket.receive()
                await events.put(("ws", msg))
        except (asyncio.CancelledError, Exception):
            pass

    try:
        ws_receive_task = asyncio.create_task(ws_receiver())
        while True:
            kind, payload = await events.get()
            if kind == "final":
                transcript = payload
                reply_aborted.set()
                if speculative_task and not speculative_task.done():
                    speculative_task.cancel()
                speculative_task = None
                speculative_transcript = None
                speculative_reply = None

                async def run_reply():
                    try:
                        await handle_audio_streaming_final(websocket, session_id, transcript, reply_aborted)
                    except Exception as err:
                        logger.exception("Error processing final transcript: %s", err)
                        try:
                            await websocket.send_text(json.dumps({"type": "error", "message": str(err)}))
                            await websocket.send_text(json.dumps({"type": "done"}))
                        except Exception:
                            pass

                asyncio.create_task(run_reply())
                continue
            assert kind == "ws"
            message = payload
            if message.get("type") == "websocket.disconnect":
                break
            if "bytes" in message and message["bytes"]:
                audio_bytes = message["bytes"]
                if streaming_stt and audio_queue is not None:
                    await audio_queue.put(audio_bytes)
                else:
                    try:
                        await handle_audio(websocket, session_id, audio_bytes, encoding=audio_encoding)
                    except Exception as err:
                        logger.exception("Error processing audio: %s", err)
                        try:
                            await websocket.send_text(json.dumps({"type": "error", "message": str(err)}))
                            await websocket.send_text(json.dumps({"type": "done"}))
                        except Exception:
                            pass
                continue
            if "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                        continue
                    if data.get("type") == "interrupt":
                        reply_aborted.set()
                        logger.info("Barge-in: reply aborted")
                        continue
                    if data.get("type") == "session":
                        sid = data.get("session_id") or ""
                        if sid and isinstance(sid, str) and len(sid) <= 128:
                            session_id = sid
                            logger.info("Session set: %s", session_id[:16] + "...")
                        continue
                    if data.get("type") == "clear_memory":
                        if session_id:
                            clear_memory(session_id)
                            clear_facts(session_id)
                            await websocket.send_text(json.dumps({"type": "memory_cleared"}))
                        continue
                    if data.get("type") == "text_message":
                        text = (data.get("text") or "").strip()
                        if text:
                            reply_aborted.set()
                            async def run_text_reply():
                                try:
                                    await handle_text_message(websocket, session_id, text)
                                except Exception as err:
                                    logger.exception("Error processing text message: %s", err)
                                    try:
                                        await websocket.send_text(json.dumps({"type": "error", "message": str(err)}))
                                        await websocket.send_text(json.dumps({"type": "done"}))
                                    except Exception:
                                        pass
                            asyncio.create_task(run_text_reply())
                        continue
                    if data.get("type") == "stable_interim":
                        text = (data.get("text") or "").strip()
                        if text and len(text) > 2:
                            if speculative_task and not speculative_task.done():
                                speculative_task.cancel()
                            speculative_transcript = text
                            speculative_reply = None

                            async def run_speculative():
                                nonlocal speculative_reply
                                try:
                                    intent = await classify_intent(text, use_llm=False)
                                    system_prompt = get_system_prompt(intent)
                                    summary, history = await get_messages_with_summary(session_id)
                                    if summary:
                                        system_prompt = system_prompt + "\n\n" + summary
                                    facts = get_facts(session_id)
                                    if facts:
                                        system_prompt = system_prompt + "\n\n" + format_facts_for_prompt(facts)
                                    speculative_reply = await get_llm_response(text, system_prompt, history)
                                except (asyncio.CancelledError, Exception) as e:
                                    if not isinstance(e, asyncio.CancelledError):
                                        logger.debug("Speculative LLM failed: %s", e)

                            speculative_task = asyncio.create_task(run_speculative())
                        continue
                    if data.get("type") == "format":
                        audio_encoding = data.get("encoding", "opus")
                        continue
                    if data.get("type") == "start_speech":
                        if streaming_stt:
                            continue
                        streaming_stt = True
                        audio_queue = asyncio.Queue()
                        transcript_queue = asyncio.Queue()
                        final_result = []
                        ev = events

                        async def forward_transcripts():
                            accumulated = []
                            while True:
                                item = await transcript_queue.get()
                                if item is None:
                                    if accumulated:
                                        full_text = " ".join(accumulated).strip()
                                        if full_text:
                                            try:
                                                await websocket.send_text(json.dumps({
                                                    "type": "transcript",
                                                    "text": full_text,
                                                    "is_final": True,
                                                    "speech_final": True,
                                                }))
                                                await ev.put(("final", full_text))
                                            except Exception:
                                                pass
                                        accumulated = []
                                    break
                                text, is_final, speech_final = item
                                if is_final and text:
                                    accumulated.append(text)
                                display = " ".join(accumulated)
                                if not is_final and text:
                                    display = (display + " " + text).strip() if display else text
                                try:
                                    await websocket.send_text(json.dumps({
                                        "type": "transcript",
                                        "text": display or text,
                                        "is_final": is_final,
                                        "speech_final": speech_final,
                                    }))
                                except Exception:
                                    break
                                if speech_final and accumulated:
                                    full_text = " ".join(accumulated).strip()
                                    accumulated = []
                                    if full_text:
                                        await ev.put(("final", full_text))

                        transcript_forward_task = asyncio.create_task(forward_transcripts())
                        try:
                            sample_rate = str(int(float(data.get("sample_rate", 48000))))
                        except (TypeError, ValueError):
                            sample_rate = "48000"
                        logger.info("Streaming STT started, sample_rate=%s", sample_rate)
                        dg_task = asyncio.create_task(
                            run_deepgram_live(
                                audio_queue, transcript_queue, final_result,
                                sample_rate=sample_rate,
                            )
                        )
                        continue
                    if data.get("type") == "end_speech":
                        if not streaming_stt or audio_queue is None:
                            continue
                        await audio_queue.put(None)
                        try:
                            await asyncio.wait_for(dg_task, timeout=10.0)
                        except asyncio.TimeoutError:
                            logger.warning("Deepgram live task timed out")
                        except Exception as e:
                            logger.warning("Deepgram task error: %s", e)
                        transcript_queue.put_nowait(None)
                        if transcript_forward_task:
                            try:
                                await asyncio.wait_for(transcript_forward_task, timeout=2.0)
                            except (asyncio.TimeoutError, asyncio.CancelledError):
                                transcript_forward_task.cancel()
                        streaming_stt = False
                        audio_queue = None
                        transcript_queue = None
                        dg_task = None
                        transcript_forward_task = None
                        continue
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        if ws_receive_task and not ws_receive_task.done():
            ws_receive_task.cancel()
            try:
                await ws_receive_task
            except asyncio.CancelledError:
                pass


async def handle_text_message(websocket: WebSocket, session_id: str, text: str):
    """Reply to a text-only message: LLM stream as text, no TTS."""
    reply_id = str(uuid.uuid4())
    await websocket.send_text(json.dumps({"type": "reply_start", "reply_id": reply_id}))

    pending = _pending_tavily.get(session_id)
    if pending:
        if _is_yes(text):
            try:
                result = await tavily_fallback_answer(pending["product_name"], pending["original_query"])
                reply_text = result.get("answer") or NO_CONTEXT_REPLY
                reply_sources = result.get("sources", [])
            except Exception as e:
                logger.warning("Tavily fallback failed: %s", e)
                reply_text = "I couldn't complete the web search right now. Please try again."
                reply_sources = []
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "reply",
                        "text": reply_text,
                        "reply_id": reply_id,
                        "sources": reply_sources,
                    }
                )
            )
            add_turn(session_id, text, reply_text)
            await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
            _pending_tavily.pop(session_id, None)
            return
        if _is_no(text):
            reply_text = "Okay, I will only use the provided datasheets."
            await websocket.send_text(json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []}))
            add_turn(session_id, text, reply_text)
            await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
            _pending_tavily.pop(session_id, None)
            return

    intent = await classify_intent(text, use_llm=False)
    system_prompt = get_system_prompt(intent)
    rag_chunks = []
    matched_product = None
    if not is_smalltalk_or_greeting(text):
        # Run matching on all non-smalltalk queries to support spoken forms
        # like "LCM thousand" that do not always match strict product regexes.
        match = match_product_name(text)
        product_query = (
            is_product_info_query(text)
            or bool(match.matched_name)
            or bool(match.mentioned_product_like_term)
        )
        if product_query:
            if match.matched_name and not match.ambiguous:
                matched_product = match.matched_name
            elif match.mentioned_product_like_term or match.ambiguous:
                reply_text = _build_product_suggestion_reply(match.suggestions)
                await websocket.send_text(
                    json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
                )
                add_turn(session_id, text, reply_text)
                await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
                return
        try:
            rag_chunks = await retrieve_context(
                text, product_name=matched_product, top_k=5, min_similarity=0.25
            )
        except Exception as e:
            logger.warning("RAG retrieval failed for text message: %s", e)
        if rag_chunks:
            system_prompt = system_prompt + "\n\n" + build_rag_context(rag_chunks, text, matched_product)
        else:
            if matched_product and has_tavily():
                _pending_tavily[session_id] = {
                    "product_name": matched_product,
                    "original_query": text,
                }
                reply_text = TAVILY_CONFIRM_REPLY
                await websocket.send_text(
                    json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
                )
                add_turn(session_id, text, reply_text)
                await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
                return
            reply_text = NO_CONTEXT_REPLY
            await websocket.send_text(json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id}))
            add_turn(session_id, text, reply_text)
            await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
            return
    summary, history = await get_messages_with_summary(session_id)
    if summary:
        system_prompt = system_prompt + "\n\n" + summary
    facts = get_facts(session_id)
    if facts:
        system_prompt = system_prompt + "\n\n" + format_facts_for_prompt(facts)

    full_reply_parts = []
    async for token in get_llm_response_stream(text, system_prompt, history):
        full_reply_parts.append(token)
        await websocket.send_text(json.dumps({"type": "reply_delta", "text": token, "reply_id": reply_id}))

    reply_text = "".join(full_reply_parts).strip()
    if matched_product and has_tavily() and _is_no_context_reply(reply_text):
        _pending_tavily[session_id] = {
            "product_name": matched_product,
            "original_query": text,
        }
        reply_text = TAVILY_CONFIRM_REPLY
        await websocket.send_text(
            json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
        )
        add_turn(session_id, text, reply_text)
        await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
        return
    sources = []
    if rag_chunks and not _is_no_context_reply(reply_text):
        sources = build_sources(rag_chunks)
    await websocket.send_text(
        json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": sources})
    )
    add_turn(session_id, text, reply_text)
    asyncio.create_task(extract_and_store_facts_async(session_id, text, reply_text))
    await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))


async def handle_audio(websocket: WebSocket, session_id: str, audio_bytes: bytes, encoding: str = "opus"):
    """Pipeline: Deepgram pre-recorded STT -> then reply pipeline."""
    transcript = await transcribe_audio_async(audio_bytes, encoding=encoding)
    if not transcript:
        await websocket.send_text(json.dumps({
            "type": "transcript",
            "text": "",
            "reply": "I didn't catch that. Could you say it again?",
        }))
    else:
        await websocket.send_text(json.dumps({"type": "transcript", "text": transcript}))
    reply_aborted = asyncio.Event()
    await handle_audio_streaming_final(websocket, session_id, transcript or None, reply_aborted)


async def handle_audio_streaming_final(
    websocket: WebSocket,
    session_id: str,
    transcript: Optional[str],
    reply_aborted: asyncio.Event,
):
    """Run intent -> context -> filler -> LLM + TTS. Respects reply_aborted for barge-in."""
    reply_id = str(uuid.uuid4())
    await websocket.send_text(json.dumps({"type": "reply_start", "reply_id": reply_id}))

    if not transcript:
        reply_text = "I didn't catch that. Could you say it again?"
        await _stream_tts_segment(
            websocket, asyncio.get_event_loop(), reply_text, reply_aborted, reply_id=reply_id
        )
        await websocket.send_text(json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id}))
        await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
        return

    reply_aborted.clear()

    pending = _pending_tavily.get(session_id)
    if pending:
        if _is_yes(transcript):
            try:
                result = await tavily_fallback_answer(pending["product_name"], pending["original_query"])
                reply_text = result.get("answer") or NO_CONTEXT_REPLY
                reply_sources = result.get("sources", [])
            except Exception as e:
                logger.warning("Tavily fallback failed: %s", e)
                reply_text = "I couldn't complete the web search right now. Please try again."
                reply_sources = []
            await _stream_tts_segment(
                websocket, asyncio.get_event_loop(), reply_text, reply_aborted, reply_id=reply_id
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "reply",
                        "text": reply_text,
                        "reply_id": reply_id,
                        "sources": reply_sources,
                    }
                )
            )
            add_turn(session_id, transcript, reply_text)
            await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
            _pending_tavily.pop(session_id, None)
            return
        if _is_no(transcript):
            reply_text = "Okay, I will only use the provided datasheets."
            await _stream_tts_segment(
                websocket, asyncio.get_event_loop(), reply_text, reply_aborted, reply_id=reply_id
            )
            await websocket.send_text(
                json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
            )
            add_turn(session_id, transcript, reply_text)
            await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
            _pending_tavily.pop(session_id, None)
            return

    intent = await classify_intent(transcript, use_llm=False)
    system_prompt = get_system_prompt(intent)
    rag_chunks = []
    matched_product = None
    if not is_smalltalk_or_greeting(transcript):
        # Same behavior as text flow: keep product matching active even when
        # STT rewrites model names into spoken forms.
        match = match_product_name(transcript)
        product_query = (
            is_product_info_query(transcript)
            or bool(match.matched_name)
            or bool(match.mentioned_product_like_term)
        )
        if product_query:
            if match.matched_name and not match.ambiguous:
                matched_product = match.matched_name
            elif match.mentioned_product_like_term or match.ambiguous:
                reply_text = _build_product_suggestion_reply(match.suggestions)
                await _stream_tts_segment(
                    websocket, asyncio.get_event_loop(), reply_text, reply_aborted, reply_id=reply_id
                )
                await websocket.send_text(
                    json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
                )
                await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
                return
        try:
            rag_chunks = await retrieve_context(
                transcript, product_name=matched_product, top_k=5, min_similarity=0.25
            )
        except Exception as e:
            logger.warning("RAG retrieval failed for voice transcript: %s", e)
        if rag_chunks:
            system_prompt = system_prompt + "\n\n" + build_rag_context(
                rag_chunks, transcript, matched_product
            )
        else:
            if matched_product and has_tavily():
                _pending_tavily[session_id] = {
                    "product_name": matched_product,
                    "original_query": transcript,
                }
                reply_text = TAVILY_CONFIRM_REPLY
                await _stream_tts_segment(
                    websocket, asyncio.get_event_loop(), reply_text, reply_aborted, reply_id=reply_id
                )
                await websocket.send_text(
                    json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
                )
                await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
                return
            reply_text = NO_CONTEXT_REPLY
            await _stream_tts_segment(
                websocket, asyncio.get_event_loop(), reply_text, reply_aborted, reply_id=reply_id
            )
            await websocket.send_text(
                json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
            )
            await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
            return
    summary, history = await get_messages_with_summary(session_id)
    if summary:
        system_prompt = system_prompt + "\n\n" + summary
    facts = get_facts(session_id)
    if facts:
        system_prompt = system_prompt + "\n\n" + format_facts_for_prompt(facts)

    filler = get_filler_phrase()
    await _stream_tts_segment(
        websocket, asyncio.get_event_loop(), filler, reply_aborted, reply_id=reply_id
    )
    if reply_aborted.is_set():
        await websocket.send_text(json.dumps({"type": "done", "interrupted": True, "reply_id": reply_id}))
        return

    full_reply_parts = []
    buffer = ""
    loop = asyncio.get_event_loop()

    async for token in get_llm_response_stream(transcript, system_prompt, history):
        if reply_aborted.is_set():
            break
        full_reply_parts.append(token)
        await websocket.send_text(json.dumps({"type": "reply_delta", "text": token, "reply_id": reply_id}))
        buffer += token
        if SENTENCE_END_RE.search(buffer) or len(buffer) >= MAX_BUFFER_LEN:
            sentence = buffer.strip()
            buffer = ""
            if sentence:
                await _stream_tts_segment(
                    websocket, loop, sentence, reply_aborted, reply_id=reply_id
                )
                if reply_aborted.is_set():
                    break

    if buffer.strip() and not reply_aborted.is_set():
        await _stream_tts_segment(
            websocket, loop, buffer.strip(), reply_aborted, reply_id=reply_id
        )

    reply_text = "".join(full_reply_parts).strip()
    if matched_product and has_tavily() and _is_no_context_reply(reply_text):
        _pending_tavily[session_id] = {
            "product_name": matched_product,
            "original_query": transcript,
        }
        reply_text = TAVILY_CONFIRM_REPLY
        await _stream_tts_segment(
            websocket, asyncio.get_event_loop(), reply_text, reply_aborted, reply_id=reply_id
        )
        await websocket.send_text(
            json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": []})
        )
        await websocket.send_text(json.dumps({"type": "done", "reply_id": reply_id}))
        return
    sources = []
    if rag_chunks and not _is_no_context_reply(reply_text):
        sources = build_sources(rag_chunks)
    await websocket.send_text(
        json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id, "sources": sources})
    )
    if not reply_aborted.is_set():
        add_turn(session_id, transcript, reply_text)
        asyncio.create_task(extract_and_store_facts_async(session_id, transcript, reply_text))
    await websocket.send_text(json.dumps({"type": "done", "interrupted": reply_aborted.is_set(), "reply_id": reply_id}))


def _split_into_sentences(text: str):
    """Split text into sentence-sized chunks for TTS (same logic as streaming)."""
    buffer = ""
    for ch in text:
        buffer += ch
        if SENTENCE_END_RE.search(buffer) or len(buffer) >= MAX_BUFFER_LEN:
            s = buffer.strip()
            buffer = ""
            if s:
                yield s
    if buffer.strip():
        yield buffer.strip()


async def handle_audio_streaming_final_with_reply(
    websocket: WebSocket,
    session_id: str,
    transcript: str,
    reply_text: str,
    reply_aborted: asyncio.Event,
):
    """Use pre-computed reply (from speculative LLM): send reply, stream TTS only, add_turn."""
    reply_id = str(uuid.uuid4())
    await websocket.send_text(json.dumps({"type": "reply_start", "reply_id": reply_id}))
    reply_aborted.clear()
    await websocket.send_text(json.dumps({"type": "reply", "text": reply_text, "reply_id": reply_id}))
    loop = asyncio.get_event_loop()
    for sentence in _split_into_sentences(reply_text):
        if reply_aborted.is_set():
            break
        await _stream_tts_segment(
            websocket, loop, sentence, reply_aborted, reply_id=reply_id
        )
    if not reply_aborted.is_set():
        add_turn(session_id, transcript, reply_text)
        asyncio.create_task(extract_and_store_facts_async(session_id, transcript, reply_text))
    await websocket.send_text(json.dumps({"type": "done", "interrupted": reply_aborted.is_set(), "reply_id": reply_id}))


async def _stream_tts_segment(
    websocket: WebSocket,
    loop: asyncio.AbstractEventLoop,
    text: str,
    reply_aborted: Optional[asyncio.Event] = None,
    reply_id: Optional[str] = None,
) -> None:
    """Send one TTS segment. If reply_aborted is set, stop sending and return."""
    if not text.strip():
        return
    chunk_queue = queue.Queue()
    aborted = reply_aborted or asyncio.Event()

    def tts_producer():
        try:
            for chunk in text_to_speech_stream(text):
                chunk_queue.put(chunk)
        except Exception as e:
            chunk_queue.put(("error", e))
        finally:
            chunk_queue.put(None)

    t = threading.Thread(target=tts_producer)
    t.start()
    try:
        if reply_id:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "reply_spoken_delta",
                        "reply_id": reply_id,
                        "text": text,
                    }
                )
            )
        await websocket.send_text(json.dumps({"type": "audio_segment_start"}))
        while True:
            if aborted.is_set():
                break
            try:
                item = await loop.run_in_executor(
                    None,
                    lambda: chunk_queue.get(timeout=0.25),
                )
            except queue.Empty:
                continue
            if item is None:
                break
            if isinstance(item, tuple) and item[0] == "error":
                err_msg = str(item[1])
                if "401" in err_msg or "Unauthorized" in err_msg or "unusual_activity" in err_msg:
                    err_msg = "ElevenLabs API limit or account restriction."
                await websocket.send_text(json.dumps({"type": "tts_error", "message": err_msg}))
                logger.warning("TTS segment failed: %s", item[1])
                break
            payload = item if isinstance(item, bytes) else bytes(item)
            if payload:
                await websocket.send_bytes(payload)
        await websocket.send_text(json.dumps({"type": "audio_segment_end"}))
    finally:
        t.join()
