"""
Context window management: trim history with summarization when too long.
"""
from __future__ import annotations

import logging
from typing import List, Tuple


from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.services.llm import get_groq_client
from app.services.memory import get_messages_for_llm

logger = logging.getLogger(__name__)

# Approximate: keep context under this many chars (≈ 4k tokens at ~4 chars/token)
MAX_CONTEXT_CHARS = 14000
# Keep at least this many recent turns in full form
MIN_RECENT_TURNS = 4
# When summarizing, summarize oldest turns into one system-style message
SUMMARY_PROMPT = """Summarize this conversation in 2-4 short sentences. Preserve key facts, decisions, and context. Output only the summary, no preamble.

Conversation:
"""


def _count_chars(messages: List[BaseMessage]) -> int:
    return sum(len(getattr(m, "content", "") or "") for m in messages)


async def _summarize_turns(messages: List[BaseMessage]) -> str:
    """Use LLM to summarize a list of messages."""
    if not messages:
        return ""
    blob = "\n".join(
        ("User: " if isinstance(m, HumanMessage) else "Assistant: ") + (getattr(m, "content", "") or "")
        for m in messages
    )
    try:
        client = get_groq_client()
        completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": SUMMARY_PROMPT + blob[:8000]}],
            temperature=0.3,
            max_tokens=300,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Context summarization failed: %s", e)
        return ""


async def get_messages_with_summary(session_id: str) -> Tuple[str, List[BaseMessage]]:
    """
    Load chat history; if over MAX_CONTEXT_CHARS, summarize oldest turns.
    Returns (summary_string, messages). Summary can be prepended to system prompt; messages are recent Human/AI only.
    """
    raw = get_messages_for_llm(session_id)
    if not raw:
        return "", []
    if _count_chars(raw) <= MAX_CONTEXT_CHARS:
        return "", raw
    n_keep_turns = MIN_RECENT_TURNS
    keep_count = n_keep_turns * 2
    if len(raw) <= keep_count:
        return "", raw
    to_summarize = raw[:-keep_count]
    rest = raw[-keep_count:]
    summary = await _summarize_turns(to_summarize)
    if summary:
        return f"Previous conversation summary: {summary}", rest
    return "", rest
