# Voice Assistant — Phase 3 (Intelligence Layer)

End-to-end pipeline: **audio in → Deepgram STT → intent → context + long-term memory → filler → Groq LLM → TTS** over WebSockets. Interruptions handled with barge-in; context and memory keep conversations natural.

## Features

- **Deepgram STT** — Streaming (hold-to-talk) and pre-recorded fallback.
- **VAD-based barge-in** — Hold the button while the assistant is speaking to interrupt: playback stops, server aborts the current reply, and your new utterance is processed.
- **Intent classification and routing** — Transcript is classified (keyword-based by default; optional LLM) into `general`, `weather`, `calendar`, `search`. Domain-specific system prompts improve replies.
- **Context window management** — When history exceeds ~14k chars, oldest turns are summarized via LLM and prepended to the system prompt so recent turns stay in full.
- **Long-term memory** — Facts about the user are stored in Redis (extracted from exchanges in the background) and injected into the prompt as “Remembered about the user: …”.
- **Filler phrases** — A short phrase (“Let me check on that.”, “One moment.”, etc.) is played before the main reply so the user hears immediate feedback.
- **Speculative processing** — When the interim transcript is stable for 400ms, the client sends `stable_interim`; the server runs the LLM in the background. When the user releases (final transcript), if it matches the interim, the cached reply is used so latency is lower.
- **Redis memory** — Chat history and long-term facts in Redis; session ID in the browser for persistence. **New conversation** clears both.

## What you need

1. **Groq API key** — [console.groq.com](https://console.groq.com/)
2. **ElevenLabs API key** — [elevenlabs.io](https://elevenlabs.io/) → Profile → API Key
3. **Deepgram API key** — [developers.deepgram.com](https://developers.deepgram.com/)
4. **Redis** (optional) — For conversation and long-term memory. Run locally: `docker run -d -p 6379:6379 redis` or use Redis Cloud.

## Setup

```bash
cd voice_assistant
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
# Edit .env: GROQ_API_KEY, ELEVENLABS_API_KEY, DEEPGRAM_API_KEY, REDIS_URL (optional)
```

## Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --reload-dir app --reload-dir static
```

## Test in the browser

Open **http://localhost:8000/app**. **Connect** → **Hold to talk** → speak → release. Reply streams as text and TTS. Hold again while the assistant is speaking to **interrupt** (barge-in). Use **New conversation** to clear history and long-term memory.

## Env vars (`.env`)

| Variable            | Description |
|---------------------|-------------|
| `GROQ_API_KEY`      | Required. Groq API key. |
| `ELEVENLABS_API_KEY`| Required. ElevenLabs API key. |
| `ELEVENLABS_VOICE`  | Optional. Voice ID (e.g. `21m00Tcm4TlvDq8ikWAM` for Rachel). |
| `DEEPGRAM_API_KEY`  | Required. Deepgram API key for STT. |
| `DEEPGRAM_MODEL`    | Optional. Default `nova-2`. |
| `REDIS_URL`         | Optional. Default `redis://localhost:6379/0`. Chat history + long-term facts. |
| `GROQ_MODEL`        | Optional. Default `llama-3.3-70b-versatile`. |

## Project layout

```
voice_assistant/
├── app/
│   ├── main.py           # FastAPI + WebSocket; barge-in, intent, context, filler, speculative
│   ├── config.py
│   └── services/
│       ├── stt.py        # Deepgram streaming + pre-recorded
│       ├── llm.py        # Groq streaming + history
│       ├── tts.py        # ElevenLabs streaming
│       ├── memory.py     # Redis chat history
│       ├── context.py    # Context window + summarization
│       ├── intent.py     # Intent classification (keyword + optional LLM)
│       ├── prompts.py    # Domain-specific system prompts
│       ├── fillers.py    # Filler phrases
│       ├── long_term_memory.py  # Facts storage, retrieval, extraction
│       └── transcribe.py # Legacy Whisper (unused)
├── static/index.html    # Hold-to-talk, barge-in, stable_interim
├── requirements.txt
└── .env.example
```

## Success metric

Conversations feel natural; interruptions are handled gracefully (barge-in stops TTS and accepts the new query).
