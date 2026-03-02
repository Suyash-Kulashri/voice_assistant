"""
Conversation memory using Redis (plain keys, no RediSearch).
Works on any Redis, including Redis Cloud without Redis Stack.
If Redis is unavailable, history is treated as empty and add_turn is a no-op.
"""
import json
import logging
from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from redis import Redis

from app.config import REDIS_URL

logger = logging.getLogger(__name__)
MEMORY_TTL = 3600
KEY_PREFIX = "voice_assistant:chat:"

_redis_client: Optional[Redis] = None


def _get_redis() -> Optional[Redis]:
    global _redis_client
    if _redis_client is None:
        if not REDIS_URL:
            return None
        try:
            _redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
            _redis_client.ping()
        except Exception as e:
            logger.warning("Redis connection failed: %s", e)
            return None
    return _redis_client


def _key(session_id: str) -> str:
    return f"{KEY_PREFIX}{session_id}"


def get_messages_for_llm(session_id: str) -> List[BaseMessage]:
    """Load chat history for the LLM. Returns [] if Redis is unavailable or empty."""
    client = _get_redis()
    if not client:
        return []
    try:
        raw = client.get(_key(session_id))
        if not raw:
            return []
        items = json.loads(raw)
        messages: List[BaseMessage] = []
        for item in items:
            role = item.get("role")
            content = item.get("content") or ""
            if role == "human":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))
        return messages
    except Exception as e:
        logger.warning("Redis memory get_messages_for_llm failed: %s", e)
        return []


def add_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    """Append one user and one assistant message to the session history."""
    client = _get_redis()
    if not client:
        return
    try:
        key = _key(session_id)
        raw = client.get(key)
        items = json.loads(raw) if raw else []
        items.append({"role": "human", "content": user_text})
        items.append({"role": "ai", "content": assistant_text})
        client.setex(key, MEMORY_TTL, json.dumps(items))
    except Exception as e:
        logger.warning("Redis memory add_turn failed: %s", e)


def clear_memory(session_id: str) -> None:
    """Clear conversation history for the session."""
    client = _get_redis()
    if not client:
        return
    try:
        client.delete(_key(session_id))
    except Exception as e:
        logger.warning("Redis memory clear failed: %s", e)
