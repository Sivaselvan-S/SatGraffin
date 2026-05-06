"""
Cross-Encoder Re-Ranker
=======================
Re-ranks a candidate set by scoring each (query, chunk-text) pair with a
HuggingFace cross-encoder model.  The model runs **locally** on CPU — no
external API calls.

Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (~80 MB).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("satgraffin.reranker")


class CrossEncoderReranker:
    """
    Lazy-loads a ``sentence_transformers.CrossEncoder`` on first use, then
    re-ranks candidate chunks against the query.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ms-marco-MiniLM-L-6-v2).
    device : str
        Torch device string (default ``"cpu"``).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None  # lazy loaded

    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s (device=%s)", self.model_name, self.device)
        self._model = CrossEncoder(self.model_name, device=self.device)
        logger.info("Cross-encoder model loaded successfully")

    # ------------------------------------------------------------------ #

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Score and re-rank *candidates* against *query*.

        Parameters
        ----------
        query : str
            The user's search query.
        candidates : list[dict]
            Chunks with at least a ``"text"`` key.
        top_k : int
            Number of top results to return after re-ranking (default 5).

        Returns
        -------
        list[dict]
            Re-ranked chunks with an additional ``"ce_score"`` key, sorted
            by descending cross-encoder score.
        """
        if not candidates:
            return []

        if self._model is None:
            self._load_model()

        # Build (query, passage) pairs
        pairs = [(query, c.get("text", "")) for c in candidates]

        # Score all pairs in a single batch
        scores = self._model.predict(pairs)

        # Attach scores and sort
        scored: list[dict[str, Any]] = []
        for candidate, score in zip(candidates, scores):
            scored.append({**candidate, "ce_score": float(score)})

        scored.sort(key=lambda c: c["ce_score"], reverse=True)

        logger.debug(
            "Cross-encoder re-ranked %d candidates → top-%d (best=%.4f, worst=%.4f)",
            len(candidates),
            top_k,
            scored[0]["ce_score"] if scored else 0,
            scored[min(top_k, len(scored)) - 1]["ce_score"] if scored else 0,
        )
        return scored[:top_k]
