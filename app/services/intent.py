"""
Intent classification and routing. Uses LLM to classify user transcript into a domain.
"""
import logging
import re
from app.services.llm import get_groq_client

logger = logging.getLogger(__name__)

INTENT_CLASSIFY_PROMPT = """Classify the user's message into exactly one intent. Reply with only one word from this list: general, weather, calendar, search.

Examples:
- "What's the weather tomorrow?" -> weather
- "Set a reminder for 5pm" -> calendar
- "Who won the game?" -> search
- "Hello" -> general
- "Tell me a joke" -> general

User message: """

# Fast keyword fallback when we don't want to call LLM every time (optional)
KEYWORD_INTENT = [
    (re.compile(r"\b(weather|forecast|temperature|rain|snow|sunny|cold|hot)\b", re.I), "weather"),
    (re.compile(r"\b(calendar|schedule|remind|meeting|appointment|tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), "calendar"),
    (re.compile(r"\b(search|find|look up|who is|what is|when did|where is)\b", re.I), "search"),
]


def classify_intent_fast(transcript: str) -> str:
    """Keyword-based intent (no API call). Returns first match or 'general'."""
    t = (transcript or "").strip()
    if not t:
        return "general"
    for pattern, intent in KEYWORD_INTENT:
        if pattern.search(t):
            return intent
    return "general"


async def classify_intent(transcript: str, use_llm: bool = True) -> str:
    """
    Classify user transcript into intent. If use_llm=True, calls LLM for accuracy;
    otherwise uses keyword fallback.
    """
    transcript = (transcript or "").strip()
    if not transcript:
        return "general"
    if not use_llm:
        return classify_intent_fast(transcript)
    try:
        client = get_groq_client()
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",  # fast model for classification
            messages=[
                {"role": "user", "content": INTENT_CLASSIFY_PROMPT + transcript[:500]},
            ],
            temperature=0,
            max_tokens=10,
        )
        raw = (completion.choices[0].message.content or "").strip().lower()
        for intent in ("weather", "calendar", "search", "general"):
            if intent in raw:
                return intent
        return classify_intent_fast(transcript)
    except Exception as e:
        logger.warning("Intent classification failed, using keyword fallback: %s", e)
        return classify_intent_fast(transcript)
