"""
BM25 Retriever
==============
Lexical / keyword retrieval using the ``rank_bm25`` library.

Maintains an in-memory inverted index of all chunks.  The index can be
persisted to disk as a pickle file alongside the FAISS vector store.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger("satgraffin.bm25")

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

# Common English stop-words (small set to keep things lightweight)
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "it", "its",
})


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stop-words, split on whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    tokens = _WHITESPACE_RE.split(text.strip())
    return [t for t in tokens if t and t not in _STOP_WORDS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Wraps ``rank_bm25.BM25Okapi`` with add / search / persist semantics.

    Each "document" is a dict with at least ``"text"`` and ``"metadata"`` keys,
    matching the output of :class:`SentenceAwareChunker`.
    """

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._documents: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = False  # True when docs added but index not yet rebuilt

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def add_documents(self, chunks: list[dict[str, Any]]) -> None:
        """
        Add chunked documents to the corpus.

        Parameters
        ----------
        chunks : list[dict]
            Each dict must have ``"text"`` (str) and ``"metadata"`` (dict).
        """
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue
            tokens = tokenize(text)
            self._corpus_tokens.append(tokens)
            self._documents.append(chunk)

        self._dirty = True
        logger.debug("Added %d chunks to BM25 corpus (total: %d)", len(chunks), len(self._documents))

    def _rebuild_index(self) -> None:
        """(Re)build the BM25Okapi index from the current corpus."""
        if not self._corpus_tokens:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self._dirty = False
        logger.info("BM25 index rebuilt with %d documents", len(self._documents))

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def search(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """
        Search the BM25 index and return the top-*k* documents.

        Returns
        -------
        list[dict]
            Each dict is a copy of the stored chunk dict with an extra
            ``"bm25_score"`` key.
        """
        if self._dirty or self._bm25 is None:
            self._rebuild_index()

        if self._bm25 is None or not self._documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices (sorted descending by score)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] <= 0:
                continue  # skip zero-score documents
            doc = {**self._documents[idx], "bm25_score": float(scores[idx]), "bm25_rank": rank}
            results.append(doc)

        return results

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist the full BM25 state (corpus + documents) to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "corpus_tokens": self._corpus_tokens,
            "documents": self._documents,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to %s (%d docs)", path, len(self._documents))

    def load(self, path: str | Path) -> bool:
        """
        Load BM25 state from *path*.  Returns ``True`` on success.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("BM25 index file not found: %s", path)
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self._corpus_tokens = state["corpus_tokens"]
            self._documents = state["documents"]
            self._dirty = True  # will rebuild on next search
            logger.info("BM25 index loaded from %s (%d docs)", path, len(self._documents))
            return True
        except Exception as e:
            logger.error("Failed to load BM25 index: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        self._corpus_tokens.clear()
        self._documents.clear()
        self._bm25 = None
        self._dirty = False
