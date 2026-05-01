import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Temp dir for incoming audio (legacy/fallback)
AUDIO_TEMP_DIR = Path(os.getenv("AUDIO_TEMP_DIR", "/tmp/voice_assistant_audio"))
AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Deepgram live model (e.g. nova-2, flux for WebM/opus)
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2")

# Groq model
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ElevenLabs voice_id (from dashboard; "Rachel" -> 21m00Tcm4TlvDq8ikWAM)
ELEVENLABS_VOICE = os.getenv("ELEVENLABS_VOICE", "21m00Tcm4TlvDq8ikWAM")

# RAG settings
EMBEDDED_OUTPUT_PATH = os.getenv(
    "EMBEDDED_OUTPUT_PATH",
    str(Path(__file__).resolve().parent.parent / "embedded_output.jsonl"),
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.25"))
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
PRODUCT_NAMES_PATH = os.getenv(
    "PRODUCT_NAMES_PATH",
    str(Path(__file__).resolve().parent.parent / "names.txt"),
)
