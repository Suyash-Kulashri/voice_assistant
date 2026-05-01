"""
Consent-gated Tavily fallback search for product-specific unanswered queries.
Search scope is the open web.
"""
import asyncio
import json
import re
from typing import Dict, List, Optional

from tavily import TavilyClient

from app.config import TAVILY_API_KEY
from app.services.llm import get_groq_client

_client: Optional[TavilyClient] = None


def has_tavily() -> bool:
    return bool(TAVILY_API_KEY)


def get_tavily_client() -> TavilyClient:
    global _client
    if _client is None:
        if not TAVILY_API_KEY:
            raise RuntimeError("TAVILY_API_KEY is not set")
        _client = TavilyClient(api_key=TAVILY_API_KEY)
    return _client


def _search_sync(product_name: str, user_query: str, max_results: int = 5) -> Dict:
    client = get_tavily_client()
    query = f"{product_name} {user_query} price cost"
    return client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )


async def search_product_info(product_name: str, user_query: str) -> Dict:
    return await asyncio.to_thread(_search_sync, product_name, user_query, 6)


def _build_sources(results: List[Dict], max_items: int = 4) -> List[Dict[str, str]]:
    sources: List[Dict[str, str]] = []
    seen = set()
    for r in results:
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = (r.get("title") or "").strip() or url
        sources.append({"label": title, "url": url})
        if len(sources) >= max_items:
            break
    return sources


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _product_regex(product_name: str) -> re.Pattern:
    raw = (product_name or "").strip().lower()
    alpha = "".join(ch for ch in raw if ch.isalpha())
    digits = "".join(ch for ch in raw if ch.isdigit())
    if alpha and digits:
        return re.compile(rf"\b{re.escape(alpha)}[\s\-_]*{re.escape(digits)}\b", re.I)
    return re.compile(rf"\b{re.escape(raw)}\b", re.I)


def _filter_results_for_product(results: List[Dict], product_name: str) -> List[Dict]:
    pnorm = _normalize(product_name)
    preg = _product_regex(product_name)
    out: List[Dict] = []
    for r in results:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        low = url.lower()
        if "/products/" in low and not preg.search(low):
            # Avoid neighboring product pages (e.g., LCM3000 page for LCM300 query).
            continue
        plain_blob = " ".join([r.get("title", ""), r.get("content", ""), url]).lower()
        blob = _normalize(plain_blob)
        # Keep only entries that mention the exact product token (e.g. LCM300, not LCM3000).
        if pnorm and pnorm not in blob:
            continue
        if not preg.search(plain_blob):
            continue
        out.append(r)
    return out


async def synthesize_search_answer(product_name: str, user_query: str, tavily_results: Dict) -> str:
    items = tavily_results.get("results", []) or []
    if not items:
        return "I couldn't find reliable web results for that specific request."

    snippets = []
    raw_payload = []
    for i, r in enumerate(items[:6], start=1):
        snippets.append(
            f"[{i}] title={r.get('title','')}\nurl={r.get('url','')}\ncontent={r.get('content','')}"
        )
        raw_payload.append(
            {
                "rank": i,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score"),
                "published_date": r.get("published_date"),
            }
        )
    prompt = (
        "You are a technical assistant. Answer using ONLY the web snippets below as context.\n"
        "These snippets are already fetched from live web search results.\n"
        "Do not claim you cannot browse the internet.\n"
        f"Product: {product_name}\n"
        f"User question: {user_query}\n"
        "If data is unavailable in snippets, clearly say so.\n"
        "Keep concise. No raw URLs in answer body."
    )
    user_context = (
        "SNIPPETS:\n"
        + "\n\n".join(snippets)
        + "\n\nRAW_TAVILY_RESULTS_JSON:\n"
        + json.dumps(raw_payload, ensure_ascii=True)
    )
    client = get_groq_client()
    out = await client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_context},
        ],
        temperature=0.2,
    )
    return (out.choices[0].message.content or "").strip()


async def tavily_fallback_answer(product_name: str, user_query: str) -> Dict:
    results = await search_product_info(product_name, user_query)
    filtered = _filter_results_for_product(results.get("results", []) or [], product_name)
    if not filtered:
        return {
            "answer": "I couldn't find reliable web results for that specific request.",
            "sources": [],
        }
    payload = {"results": filtered}
    answer = await synthesize_search_answer(product_name, user_query, payload)
    sources = _build_sources(filtered)
    return {"answer": answer, "sources": sources}
