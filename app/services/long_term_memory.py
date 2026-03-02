"""
Long-term memory: persistent facts about the user/session (beyond chat history).
Stored in Redis; retrieved and injected into context for the LLM.
"""
import json
import logging
from typing import List, Optional

from redis import Redis

from app.config import REDIS_URL

logger = logging.getLogger(__name__)
FACTS_TTL = 86400 * 7  # 7 days
KEY_PREFIX = "voice_assistant:facts:"

_client: Optional[Redis] = None


def _get_redis() -> Optional[Redis]:
    global _client
    if _client is None:
        if not REDIS_URL:
            return None
        try:
            _client = Redis.from_url(REDIS_URL, decode_responses=True)
            _client.ping()
        except Exception as e:
            logger.warning("Redis (long-term memory) connection failed: %s", e)
            return None
    return _client


def get_facts(session_id: str) -> List[str]:
    """Return list of stored facts for the session."""
    client = _get_redis()
    if not client:
        return []
    try:
        raw = client.get(KEY_PREFIX + session_id)
        if not raw:
            return []
        return json.loads(raw)
    except Exception as e:
        logger.warning("Long-term memory get_facts failed: %s", e)
        return []


def add_fact(session_id: str, fact: str) -> None:
    """Append a fact (deduplicated by simple string match)."""
    if not (fact and fact.strip()):
        return
    client = _get_redis()
    if not client:
        return
    try:
        key = KEY_PREFIX + session_id
        raw = client.get(key)
        facts = json.loads(raw) if raw else []
        f = fact.strip()
        if f and f not in facts:
            facts.append(f)
            client.setex(key, FACTS_TTL, json.dumps(facts))
    except Exception as e:
        logger.warning("Long-term memory add_fact failed: %s", e)


def clear_facts(session_id: str) -> None:
    """Clear long-term facts for the session (e.g. on new conversation)."""
    client = _get_redis()
    if not client:
        return
    try:
        client.delete(KEY_PREFIX + session_id)
    except Exception as e:
        logger.warning("Long-term memory clear_facts failed: %s", e)


def format_facts_for_prompt(facts: List[str]) -> str:
    """Turn facts list into a string for system prompt."""
    if not facts:
        return ""
    return "Remembered about the user: " + "; ".join(facts[:20]) + "."


EXTRACT_PROMPT = """From this exchange, extract 0-3 short facts about the user we could remember for later (e.g. name, preference, location). One fact per line. Only factual statements. If nothing to remember, output nothing.

User: """
EXTRACT_SUFFIX = """

Assistant: """


async def extract_and_store_facts_async(session_id: str, user_text: str, assistant_text: str) -> None:
    """Call LLM to extract facts from the exchange and store. Fire-and-forget from main."""
    try:
        from app.services.llm import get_groq_client
        client = get_groq_client()
        out = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": EXTRACT_PROMPT + (user_text[:300] or "") + EXTRACT_SUFFIX + (assistant_text[:300] or ""),
            }],
            temperature=0,
            max_tokens=150,
        )
        raw = (out.choices[0].message.content or "").strip()
        for line in raw.splitlines():
            line = line.strip().lstrip("-•* ").strip()
            if line and len(line) < 200:
                add_fact(session_id, line)
    except Exception as e:
        logger.debug("Fact extraction failed: %s", e)
