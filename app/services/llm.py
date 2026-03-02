"""
Get LLM response from Groq. Supports streaming and conversation history.
"""
from typing import AsyncIterator, List, Optional

from groq import AsyncGroq
from langchain_core.messages import AIMessage, HumanMessage

from app.config import GROQ_API_KEY, GROQ_MODEL

_client = None  # AsyncGroq


def get_groq_client() -> AsyncGroq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in environment")
        _client = AsyncGroq(api_key=GROQ_API_KEY)
    return _client


def _langchain_to_groq_messages(messages: List) -> List[dict]:
    """Convert LangChain BaseMessage list to Groq API format."""
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            out.append({"role": "assistant", "content": m.content})
        elif hasattr(m, "content"):
            out.append({"role": "assistant", "content": m.content})
        else:
            continue
    return out


def build_messages(
    history_messages: List,
    new_user_text: str,
    system_prompt: Optional[str] = None,
) -> List[dict]:
    """Build Groq messages from history + new user message + optional system prompt."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(_langchain_to_groq_messages(history_messages))
    messages.append({"role": "user", "content": new_user_text})
    return messages


async def get_llm_response(
    user_text: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List] = None,
) -> str:
    """Send user message to Groq and return full assistant reply (non-streaming)."""
    client = get_groq_client()
    messages = build_messages(history_messages or [], user_text, system_prompt)

    completion = await client.chat.completions.create(
        messages=messages,
        model=GROQ_MODEL,
        temperature=0.7,
    )
    return (completion.choices[0].message.content or "").strip()


async def get_llm_response_stream(
    user_text: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[List] = None,
) -> AsyncIterator[str]:
    """Stream Groq reply token-by-token. Yields content deltas."""
    client = get_groq_client()
    messages = build_messages(history_messages or [], user_text, system_prompt)

    stream = await client.chat.completions.create(
        messages=messages,
        model=GROQ_MODEL,
        temperature=0.7,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
