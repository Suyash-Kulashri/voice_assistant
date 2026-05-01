"""
Simple RAG retriever (Option A): stream JSONL embeddings and do cosine top-k in memory.
"""
import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from app.config import (
    EMBEDDED_OUTPUT_PATH,
    OPENAI_API_KEY,
    RAG_EMBEDDING_MODEL,
    RAG_MIN_SIMILARITY,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)

_openai_client: Optional[AsyncOpenAI] = None
_load_lock = asyncio.Lock()
_loaded = False
_vectors: Optional[np.ndarray] = None
_norms: Optional[np.ndarray] = None
_metadata: List[Dict[str, Any]] = []

GREETING_SMALLTALK_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b", re.I),
    re.compile(r"\b(how are you|what's up|whats up|thank you|thanks)\b", re.I),
]
PRODUCT_CODE_RE = re.compile(r"\b[A-Z]{2,}\s?-?\d{2,}[A-Z0-9-]*\b", re.I)
PRODUCT_INFO_PATTERNS = [
    re.compile(r"\b(spec|specification|datasheet|technical|rating|power supply|adapter|ac[-\s]?dc)\b", re.I),
    re.compile(r"\b(what is|tell me about|details of|info about)\b", re.I),
]
TECH_KEY_HINTS = (
    "input",
    "output",
    "power",
    "efficiency",
    "voltage",
    "current",
    "frequency",
    "temperature",
    "range",
    "protection",
    "isolation",
    "safety",
    "regulation",
    "ripple",
    "noise",
    "hold_up",
)


@dataclass
class RetrievedChunk:
    score: float
    text: str
    metadata: Dict[str, Any]


def is_smalltalk_or_greeting(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t.split()) <= 3:
        low = t.lower()
        if low in {"hi", "hello", "hey", "yo", "thanks", "thank you"}:
            return True
    return any(p.search(t) for p in GREETING_SMALLTALK_PATTERNS)


def _extract_product_terms(text: str) -> List[str]:
    # Normalize product tokens for fuzzy matching, e.g. "LCM 300" -> "lcm300".
    terms: List[str] = []
    for m in PRODUCT_CODE_RE.findall(text or ""):
        norm = re.sub(r"[^a-z0-9]", "", m.lower())
        if norm and norm not in terms:
            terms.append(norm)
    return terms


def is_product_info_query(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _extract_product_terms(t):
        return True
    return any(p.search(t) for p in PRODUCT_INFO_PATTERNS)


def _get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in environment")
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def _load_index_sync(path: Path) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    vectors: List[List[float]] = []
    metadata: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            emb = row.get("embedding")
            sent = (row.get("sentence") or row.get("value_text") or "").strip()
            if not emb or not isinstance(emb, list) or not sent:
                continue
            if not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in emb):
                continue
            vectors.append(emb)
            metadata.append(
                {
                    "id": row.get("id"),
                    "series": row.get("series"),
                    "model": row.get("model"),
                    "key": row.get("key"),
                    "page": row.get("page"),
                    "datasheet_url": row.get("datasheet_url"),
                    "sentence": sent,
                }
            )
    if not vectors:
        raise RuntimeError(f"No valid embedding rows found in {path}")
    vec_arr = np.asarray(vectors, dtype=np.float32)
    vec_arr = np.nan_to_num(vec_arr, nan=0.0, posinf=0.0, neginf=0.0)
    vec_arr = np.clip(vec_arr, -10.0, 10.0)
    norms = np.linalg.norm(vec_arr, axis=1)
    norms = np.maximum(norms, 1e-12)
    return vec_arr, norms, metadata


async def _ensure_loaded() -> None:
    global _loaded, _vectors, _norms, _metadata
    if _loaded:
        return
    async with _load_lock:
        if _loaded:
            return
        path = Path(EMBEDDED_OUTPUT_PATH)
        if not path.is_file():
            raise RuntimeError(f"Embedding file not found: {path}")
        logger.info("Loading RAG index from %s", path)
        _vectors, _norms, _metadata = await asyncio.to_thread(_load_index_sync, path)
        _loaded = True
        logger.info("Loaded RAG index with %d chunks", len(_metadata))


async def _embed_query(text: str) -> np.ndarray:
    client = _get_openai_client()
    resp = await client.embeddings.create(
        model=RAG_EMBEDDING_MODEL,
        input=text,
    )
    q = np.asarray(resp.data[0].embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm <= 1e-12:
        return q
    return q / q_norm


def _select_top_k(
    qvec: np.ndarray,
    query: str,
    product_name: Optional[str],
    top_k: int,
    min_similarity: float,
) -> List[RetrievedChunk]:
    def metadata_matches_product(md: Dict[str, Any], terms: List[str]) -> bool:
        if not terms:
            return False
        series_norm = re.sub(r"[^a-z0-9]", "", str(md.get("series") or "").lower())
        model_norm = re.sub(r"[^a-z0-9]", "", str(md.get("model") or "").lower())
        return any(t == series_norm or (model_norm and t == model_norm) for t in terms)

    assert _vectors is not None and _norms is not None
    scores = (_vectors @ qvec) / _norms
    pool_k = min(max(top_k * 8, 40), len(scores))
    if pool_k >= len(scores):
        idxs = np.argsort(-scores)
    else:
        candidate = np.argpartition(scores, -pool_k)[-pool_k:]
        idxs = candidate[np.argsort(-scores[candidate])]

    product_terms = _extract_product_terms(query)
    if product_name:
        pnorm = re.sub(r"[^a-z0-9]", "", product_name.lower())
        if pnorm and pnorm not in product_terms:
            product_terms.append(pnorm)
    product_query = bool(product_terms) or is_product_info_query(query)
    out: List[RetrievedChunk] = []
    seen_sentences = set()
    per_key_count: Dict[str, int] = {}
    for idx in idxs.tolist():
        score = float(scores[idx])
        if score < min_similarity:
            continue
        md = _metadata[idx]
        text = md["sentence"]
        sent_norm = text.strip().lower()
        if sent_norm in seen_sentences:
            continue

        key = (md.get("key") or "").lower()
        series = str(md.get("series") or "")
        model = str(md.get("model") or "")
        combined_norm = re.sub(r"[^a-z0-9]", "", f"{series} {model} {text}".lower())

        boosted_score = score
        strict_product_match = metadata_matches_product(md, product_terms)
        matched_target = strict_product_match or (any(term in combined_norm for term in product_terms) if product_terms else False)
        if product_query:
            if matched_target:
                boosted_score += 0.25
            if any(h in key for h in TECH_KEY_HINTS):
                boosted_score += 0.08
            if "marketing" in key or "feature" in key:
                boosted_score -= 0.03
            if product_name and not strict_product_match:
                continue

        key_bucket = key or "unknown"
        if product_query and per_key_count.get(key_bucket, 0) >= 2:
            continue

        seen_sentences.add(sent_norm)
        per_key_count[key_bucket] = per_key_count.get(key_bucket, 0) + 1
        out.append(RetrievedChunk(score=boosted_score, text=text, metadata=md))
        if len(out) >= top_k * 3:
            break
    out.sort(key=lambda c: c.score, reverse=True)
    out = out[:top_k]
    return out


async def retrieve_context(
    query: str,
    *,
    product_name: Optional[str] = None,
    top_k: int = RAG_TOP_K,
    min_similarity: float = RAG_MIN_SIMILARITY,
) -> List[RetrievedChunk]:
    await _ensure_loaded()
    qvec = await _embed_query(query)
    if qvec.size == 0:
        return []
    return await asyncio.to_thread(_select_top_k, qvec, query, product_name, top_k, min_similarity)


def build_rag_context(chunks: List[RetrievedChunk], query: str = "", product_name: Optional[str] = None) -> str:
    if not chunks:
        return ""
    product_query = is_product_info_query(query)
    lines = [
        "You are a technical assistant for Advanced Energy Industries products.",
        "You answer questions based on the provided datasheet content extracted from PDFs.",
        "",
        "CRITICAL RULES:",
        "1. Answer based on the provided context from datasheets. Synthesize information from multiple chunks if needed.",
        "2. If the information is truly NOT in the context (after carefully reviewing all provided chunks), say exactly: I don't have that specific information in the provided datasheets",
        "3. IMPORTANT: If chunks contain information about the product (even if not directly answering the question), provide a helpful response based on what IS available. Do not say you don't have information when relevant product details exist.",
        "4. Always mention which product/series you're referencing.",
        "5. For specifications, be PRECISE with exact values and units from the datasheet.",
        "6. You can infer and synthesize information from context; exact phrase matches are not required.",
        "7. Do NOT include raw URLs, citation tags, or source lists in the answer body. Sources are rendered separately by the UI.",
        "8. Answer ONLY for the matched product/series target; do not mix with other products.",
        "",
        "User can ask two types of queries: PRODUCT OVERVIEW or SPECIFIC TECHNICAL INFORMATION.",
        "",
        "PRODUCT OVERVIEW format (e.g., 'What is X?', 'Tell me about X'):",
        "- Start with heading: PRODUCT SUMMARY",
        "- Include: Product Type (AC-DC/DC-DC/DC-AC/other), Purpose/Applications, and a brief 2-3 sentence overview",
        "- Then heading: TECHNICAL SPECIFICATIONS",
        "- List only 5-6 most important specifications (power rating, input/output voltage, key features, etc.)",
        "- Use bullet points and bold important values with **text**",
        "- Direct user to datasheet link for full specifications",
        "",
        "SPECIFIC TECHNICAL INFORMATION format (e.g., efficiency/temperature/output current questions):",
        "- Provide a detailed answer to the specific question with exact values and units when available",
        "- If exact value isn't found, provide related available information",
        "- Then provide a brief detailed description of the product and intended applications",
        "- List only 5-6 most important specifications",
        "- Use bullet points and bold important values with **text**",
        "- Direct user to datasheet link for complete specifications",
        "",
        "Formatting guidelines:",
        "- Keep response concise and focused",
        "- Use clear section headers and bullet points for specs",
        "- Present table-derived values clearly",
        "- Synthesize information from multiple chunks when useful",
        "- Do not include internal source tags, chunk IDs, or citation markers like [S1]",
        "",
        f"Query type hint: {'PRODUCT OVERVIEW/TECHNICAL PRODUCT QUESTION' if product_query else 'GENERAL PRODUCT QUESTION'}",
        f"Matched product/series target: {product_name or 'None'}",
    ]
    lines.extend([
        "",
        "Context:",
    ])
    for i, c in enumerate(chunks, start=1):
        md = c.metadata
        lines.append(
            f"[S{i}] {c.text} (series={md.get('series')}, key={md.get('key')}, page={md.get('page')})"
        )
    return "\n".join(lines)


def build_sources(chunks: List[RetrievedChunk], *, max_items: int = 3) -> List[Dict[str, Any]]:
    if not chunks:
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for c in chunks:
        md = c.metadata
        url = (md.get("datasheet_url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        series = md.get("series") or "Unknown"
        key = md.get("key") or "detail"
        page = md.get("page")
        label = f"{series} - {key}"
        if page is not None:
            label += f" (page {page})"
        out.append({"label": label, "url": url})
        if len(out) >= max_items:
            break
    return out
