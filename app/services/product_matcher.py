"""
Product/series name matcher using names.txt.
"""
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Optional

from app.config import PRODUCT_NAMES_PATH

_loaded_names: Optional[List[str]] = None
_loaded_norms: Optional[List[str]] = None

PRODUCT_MENTION_RE = re.compile(r"\b[a-z]{2,}[\s\-]?\d{2,}[a-z0-9\-]*\b", re.I)
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "hundred": "100",
    "thousand": "1000",
}


@dataclass
class ProductMatchResult:
    matched_name: Optional[str]
    suggestions: List[str]
    confidence: float
    mentioned_product_like_term: bool
    ambiguous: bool


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _split_alpha_num(s: str) -> tuple[str, str]:
    alpha = "".join(ch for ch in s if ch.isalpha())
    digits = "".join(ch for ch in s if ch.isdigit())
    return alpha, digits


def _extract_query_targets(query: str) -> List[str]:
    q = (query or "").strip()
    # Normalize numeric commas: "1,000" -> "1000"
    q = re.sub(r"(?<=\d),(?=\d)", "", q)
    targets: List[str] = []

    # Primary: product-like spans such as "SLP 65", "AIF06", "LCM-300".
    for m in PRODUCT_MENTION_RE.findall(q):
        n = _normalize(m)
        if n and n not in targets:
            targets.append(n)

    # Fallback: strong alnum tokens from the sentence.
    if not targets:
        for tok in re.findall(r"[a-z0-9\-]+", q.lower()):
            n = _normalize(tok)
            if len(n) >= 4 and any(ch.isdigit() for ch in n):
                if n not in targets:
                    targets.append(n)

    # Handle spaced spellings like "b a 10" or "d a ten".
    tokens = re.findall(r"[a-z0-9]+", q.lower())
    normalized_tokens = [NUMBER_WORDS.get(t, t) for t in tokens]
    for i in range(len(normalized_tokens) - 2):
        a, b, c = normalized_tokens[i], normalized_tokens[i + 1], normalized_tokens[i + 2]
        if len(a) == 1 and len(b) == 1 and c.isdigit():
            n = _normalize(a + b + c)
            if n and n not in targets:
                targets.append(n)

    # Handle spoken forms like "lcm thousand" / "lcm one thousand" / "lcm 1000".
    for i in range(len(normalized_tokens) - 1):
        base = normalized_tokens[i]
        nxt = normalized_tokens[i + 1]
        if base.isalpha() and len(base) >= 2 and nxt.isdigit():
            n = _normalize(base + nxt)
            if n and n not in targets:
                targets.append(n)
        if i + 2 < len(normalized_tokens):
            nxt2 = normalized_tokens[i + 2]
            if base.isalpha() and len(base) >= 2 and nxt == "1" and nxt2 == "1000":
                n = _normalize(base + "1000")
                if n and n not in targets:
                    targets.append(n)
    return targets


def _load_names() -> None:
    global _loaded_names, _loaded_norms
    if _loaded_names is not None and _loaded_norms is not None:
        return
    path = Path(PRODUCT_NAMES_PATH)
    if not path.is_file():
        _loaded_names = []
        _loaded_norms = []
        return
    names: List[str] = []
    norms: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if not name:
                continue
            norm = _normalize(name)
            if not norm:
                continue
            names.append(name)
            norms.append(norm)
    _loaded_names = names
    _loaded_norms = norms


def _score(targets: List[str], cand_norm: str) -> float:
    if not targets or not cand_norm:
        return 0.0

    cand_alpha, cand_digits = _split_alpha_num(cand_norm)
    best = 0.0
    for t in targets:
        if t == cand_norm:
            return 1.0
        if t in cand_norm or cand_norm in t:
            best = max(best, 0.95)

        t_alpha, t_digits = _split_alpha_num(t)
        full_ratio = SequenceMatcher(None, t, cand_norm).ratio()
        alpha_ratio = SequenceMatcher(None, t_alpha, cand_alpha).ratio() if t_alpha and cand_alpha else 0.0

        digit_score = 0.0
        if t_digits and cand_digits:
            if t_digits == cand_digits:
                digit_score = 1.0
            elif t_digits in cand_digits or cand_digits in t_digits:
                digit_score = 0.75
            else:
                digit_score = SequenceMatcher(None, t_digits, cand_digits).ratio()

        # Favor candidates with same numeric family + close letter prefix.
        combined = (0.45 * full_ratio) + (0.35 * alpha_ratio) + (0.20 * digit_score)
        if t_digits and cand_digits and t_digits == cand_digits and alpha_ratio >= 0.5:
            combined += 0.08
        best = max(best, combined)
    return min(best, 1.0)


def match_product_name(query: str, *, top_n: int = 5) -> ProductMatchResult:
    _load_names()
    names = _loaded_names or []
    norms = _loaded_norms or []

    q = (query or "").strip()
    targets = _extract_query_targets(q)
    mentioned = bool(PRODUCT_MENTION_RE.search(q))

    if not names:
        return ProductMatchResult(
            matched_name=None, suggestions=[], confidence=0.0, mentioned_product_like_term=mentioned, ambiguous=False
        )

    # If user didn't include any product-like token, skip matching logic.
    if not targets:
        return ProductMatchResult(
            matched_name=None, suggestions=[], confidence=0.0, mentioned_product_like_term=mentioned, ambiguous=False
        )

    scored = []
    for name, norm in zip(names, norms):
        s = _score(targets, norm)
        if s > 0:
            scored.append((s, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return ProductMatchResult(
            matched_name=None, suggestions=[], confidence=0.0, mentioned_product_like_term=mentioned, ambiguous=False
        )

    best_score, best_name = scored[0]
    suggestions = [n for _, n in scored[:top_n]]
    best_norm = _normalize(best_name)
    exact_target_hit = best_norm in set(targets)
    matched_name = best_name if (best_score >= 0.84 or exact_target_hit) else None
    second_score = scored[1][0] if len(scored) > 1 else 0.0
    ambiguous = bool(
        (matched_name is None and best_score >= 0.68)
        or (matched_name is not None and not exact_target_hit and (best_score - second_score) < 0.08)
    )
    return ProductMatchResult(
        matched_name=matched_name,
        suggestions=suggestions,
        confidence=float(best_score),
        mentioned_product_like_term=mentioned,
        ambiguous=ambiguous,
    )
