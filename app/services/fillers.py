"""
Filler phrases played while the assistant is "thinking" (before LLM response).
"""
import random

FILLER_PHRASES = [
    "Let me check on that.",
    "One moment.",
    "Let me look that up.",
    "Give me a second.",
    "Thinking.",
]

def get_filler_phrase() -> str:
    """Return a random filler phrase."""
    return random.choice(FILLER_PHRASES)
