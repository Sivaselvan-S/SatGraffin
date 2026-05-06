"""
Sentence-Aware Chunker
======================
Splits text into chunks of 256–512 tokens while respecting sentence boundaries.
Each chunk carries full metadata (source, page, chunk_id, title, quality_score).

Token counting uses whitespace splitting (~90 % accurate for English) to avoid
an extra dependency on tiktoken.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("satgraffin.chunker")

# ---------------------------------------------------------------------------
# Try to use NLTK sentence tokeniser; fall back to regex if unavailable
# ---------------------------------------------------------------------------
try:
    import nltk

    # Download punkt data silently if missing
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize

    def _sent_tokenize(text: str) -> list[str]:
        return _nltk_sent_tokenize(text)

    logger.info("Using NLTK sentence tokeniser")

except ImportError:
    logger.warning("NLTK not available – falling back to regex sentence splitter")

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def _sent_tokenize(text: str) -> list[str]:  # type: ignore[misc]
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Approximate token count via whitespace splitting."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SentenceAwareChunker:
    """
    Produces chunks of *min_tokens*–*max_tokens* words by accumulating full
    sentences.  Metadata from the caller is attached to every chunk.

    Parameters
    ----------
    min_tokens : int
        Minimum token count before a chunk is considered "full" (default 256).
    max_tokens : int
        Hard upper limit per chunk (default 512).  A single sentence that
        exceeds this is kept as-is (never mid-sentence split).
    overlap_sentences : int
        Number of trailing sentences from the previous chunk to prepend to the
        next chunk for context continuity (default 1).
    """

    def __init__(
        self,
        min_tokens: int = 256,
        max_tokens: int = 512,
        overlap_sentences: int = 1,
    ) -> None:
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences

    # --------------------------------------------------------------------- #

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Split *text* into sentence-aware chunks.

        Returns
        -------
        list[dict]
            Each dict has keys ``"text"`` and ``"metadata"`` (which includes a
            unique ``chunk_id``).
        """
        if not text or not text.strip():
            return []

        metadata = dict(metadata) if metadata else {}
        sentences = _sent_tokenize(text)

        if not sentences:
            return []

        chunks: list[dict[str, Any]] = []
        current_sentences: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sent_tokens = _token_count(sentence)

            # Would adding this sentence exceed the hard limit?
            if current_sentences and (current_tokens + sent_tokens) > self.max_tokens:
                # Flush the current chunk
                chunks.append(self._make_chunk(current_sentences, metadata, chunk_index))
                chunk_index += 1

                # Carry over overlap sentences for continuity
                overlap = current_sentences[-self.overlap_sentences:] if self.overlap_sentences else []
                current_sentences = list(overlap)
                current_tokens = sum(_token_count(s) for s in current_sentences)

            current_sentences.append(sentence)
            current_tokens += sent_tokens

            # If we've reached a comfortable size, flush
            if current_tokens >= self.min_tokens and current_tokens <= self.max_tokens:
                # Only flush if the *next* sentence would push us over
                pass  # let the loop continue; we flush at the top

        # Flush remaining
        if current_sentences:
            chunks.append(self._make_chunk(current_sentences, metadata, chunk_index))

        logger.debug(
            "Chunked %d sentences into %d chunks (avg %d tokens)",
            len(sentences),
            len(chunks),
            sum(_token_count(c["text"]) for c in chunks) // max(len(chunks), 1),
        )
        return chunks

    # --------------------------------------------------------------------- #

    @staticmethod
    def _make_chunk(
        sentences: list[str],
        base_metadata: dict[str, Any],
        chunk_index: int,
    ) -> dict[str, Any]:
        chunk_text = " ".join(sentences)
        meta = {**base_metadata, "chunk_id": chunk_index}
        return {"text": chunk_text, "metadata": meta}
