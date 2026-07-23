"""
Local Query Analyzer
====================
A pure-Python, zero-API-call replacement for the LLM-based QueryAnalyzer.
Used in API_SAVER_MODE to eliminate the first Gemini call per query.

Handles:
  - Greeting / chitchat detection  → requires_web_search=False
  - Temporal keyword resolution    → injects current year
  - Compound query decomposition   → vs / compare / difference splits
  - Time sensitivity detection
  - Basic spatial context          → "in India", "at Chennai"
"""

from __future__ import annotations

import re
import logging
from datetime import datetime
from typing import List, Optional

# Import the same QueryAnalysis schema so the return type is drop-in compatible
from hybrid_rag.query_analyzer import QueryAnalysis

logger = logging.getLogger("satgraffin.local_analyzer")

# ---------------------------------------------------------------------------
# Intent keyword sets
# ---------------------------------------------------------------------------

_GREETING_WORDS: set[str] = {
    "hi", "hello", "hey", "greetings", "howdy", "sup", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "bye", "goodbye", "see you", "cya",
}

_TEMPORAL_WORDS: set[str] = {
    "current", "currently", "latest", "recent", "recently", "now",
    "today", "this year", "last year", "newest", "new", "upcoming",
    "this week", "this month", "present", "modern", "updated",
}

_SPATIAL_PATTERNS: list[str] = [
    r"\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    r"\bat\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
    r"\b(?:from|near|around)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
]

# Patterns that signal a compound / comparative query → decomposition needed.
# Each tuple is (regex_pattern, kind_label).
_SPLIT_PATTERNS: list[tuple[str, str]] = [
    # "A vs B"  /  "A versus B"
    (r"(?i)(.+?)\s+(?:vs\.?|versus)\s+(.+)", "vs"),
    # "compare A and B"  /  "compare A with B"
    (r"(?i)compare\s+(.+?)\s+(?:and|with|to)\s+(.+)", "compare"),
    # "difference between A and B"
    (r"(?i)difference\s+between\s+(.+?)\s+and\s+(.+)", "diff"),
    # "pros and cons of X"
    (r"(?i)pros\s+and\s+cons\s+(?:of\s+)?(.+)", "pros_cons"),
    # "causes and effects of X"
    (r"(?i)causes?\s+and\s+effects?\s+(?:of\s+)?(.+)", "cause_effect"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_greeting(query: str) -> bool:
    """Return True if the query is purely a greeting / chitchat."""
    q = query.strip().lower().rstrip("!?.")
    if q in _GREETING_WORDS:
        return True
    words = q.split()
    if len(words) <= 4 and any(w in _GREETING_WORDS for w in words):
        return True
    return False


def _detect_temporal(query: str) -> tuple[bool, str]:
    """Return (is_time_sensitive, year_enriched_query)."""
    q_lower = query.lower()
    found = any(kw in q_lower for kw in _TEMPORAL_WORDS)
    if found:
        year = datetime.now().year
        if str(year) not in query:
            return True, f"{query} {year}"
        return True, query
    return False, query


def _detect_spatial(query: str) -> Optional[str]:
    """Return first detected spatial/location string or None."""
    for pat in _SPATIAL_PATTERNS:
        m = re.search(pat, query)
        if m:
            return m.group(1)
    return None


def _decompose(query: str) -> tuple[bool, List[str]]:
    """
    Attempt to split a compound query into 2-4 atomic sub-queries.
    Returns (is_complex, sub_queries).
    """
    for pattern, kind in _SPLIT_PATTERNS:
        m = re.match(pattern, query.strip())
        if not m:
            continue

        if kind in ("vs", "compare", "diff"):
            a = m.group(1).strip()
            b = m.group(2).strip()
            logger.debug("[LocalAnalyzer] Decomposed [%s]: '%s' | '%s'", kind, a, b)
            return True, [a, b]

        if kind == "pros_cons":
            topic = m.group(1).strip()
            return True, [f"advantages pros of {topic}", f"disadvantages cons of {topic}"]

        if kind == "cause_effect":
            topic = m.group(1).strip()
            return True, [f"causes of {topic}", f"effects of {topic}"]

    return False, []


def _simple_pronoun_resolve(query: str, context: str) -> str:
    """
    Very lightweight: replace 'it'/'this'/'that'/'they'/'them' with the
    last capitalised noun phrase found in the context tail (last 200 chars).
    """
    if not context:
        return query
    ctx_tail = context[-200:]
    topic_matches = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", ctx_tail)
    if topic_matches:
        last_topic = topic_matches[-1]
        query = re.sub(r"\b(it|this|that|they|them)\b", last_topic, query, flags=re.IGNORECASE)
    return query


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LocalQueryAnalyzer:
    """
    Drop-in local replacement for QueryAnalyzer.

    Returns a ``QueryAnalysis`` object with the same schema as the LLM version,
    computed with pure Python — **no API calls, no model loading**.

    Typical latency: < 1 ms
    """

    def analyze(self, query: str, conversation_context: str = "") -> QueryAnalysis:
        """
        Analyze the query locally and return a QueryAnalysis object.

        Parameters
        ----------
        query : str
            Raw user query.
        conversation_context : str
            Optional prior conversation text for basic pronoun resolution.
        """
        # --- Greeting / chitchat detection ---
        if _is_greeting(query):
            logger.info("[LocalAnalyzer] Greeting detected: '%s'", query)
            return QueryAnalysis(
                optimized_search_query=query,
                is_time_sensitive=False,
                spatial_context=None,
                requires_web_search=False,
                is_complex=False,
                sub_queries=[],
            )

        # --- Lightweight pronoun resolution ---
        resolved = _simple_pronoun_resolve(query, conversation_context)

        # --- Temporal enrichment ---
        is_time_sensitive, enriched = _detect_temporal(resolved)

        # --- Spatial context ---
        spatial = _detect_spatial(resolved)

        # --- Compound decomposition ---
        is_complex, sub_queries = _decompose(resolved)

        logger.info(
            "[LocalAnalyzer] query='%s' time=%s complex=%s subs=%s",
            query, is_time_sensitive, is_complex, sub_queries,
        )

        return QueryAnalysis(
            optimized_search_query=enriched,
            is_time_sensitive=is_time_sensitive,
            spatial_context=spatial,
            requires_web_search=True,
            is_complex=is_complex,
            sub_queries=sub_queries,
        )
