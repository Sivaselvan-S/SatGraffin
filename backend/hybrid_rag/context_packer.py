"""
Context Packer
==============
De-duplicates near-identical chunks (Jaccard similarity >= threshold) and
greedily fills a context window up to a token budget.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("satgraffin.context_packer")


def _token_count(text: str) -> int:
    """Approximate token count via whitespace splitting."""
    return len(text.split())


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between two texts (word-level)."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class ContextPacker:
    """
    De-duplicate and greedily pack chunks into a context window.

    Parameters
    ----------
    max_tokens : int
        Maximum tokens in packed context (default 3000).
    jaccard_threshold : float
        Similarity threshold for dedup (default 0.85).
    """

    def __init__(self, max_tokens: int = 3000, jaccard_threshold: float = 0.85):
        self.max_tokens = max_tokens
        self.jaccard_threshold = jaccard_threshold

    def pack(
        self,
        chunks: list[dict[str, Any]],
        max_tokens: int | None = None,
        jaccard_threshold: float | None = None,
    ) -> list[dict[str, Any]]:
        """De-duplicate and pack *chunks* into the context window."""
        budget = max_tokens if max_tokens is not None else self.max_tokens
        threshold = jaccard_threshold if jaccard_threshold is not None else self.jaccard_threshold

        packed: list[dict[str, Any]] = []
        tokens_used = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue

            chunk_tokens = _token_count(text)

            if tokens_used + chunk_tokens > budget:
                break

            # Check for near-duplicate
            is_dup = any(
                _jaccard_similarity(text, a["text"]) >= threshold
                for a in packed
            )
            if is_dup:
                continue

            packed.append(chunk)
            tokens_used += chunk_tokens

        logger.debug("Packed %d/%d chunks (%d/%d tokens)", len(packed), len(chunks), tokens_used, budget)
        return packed
