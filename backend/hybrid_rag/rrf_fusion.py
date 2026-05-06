"""
Reciprocal Rank Fusion (RRF)
============================
Merges multiple ranked result lists into a single fused ranking.

    score(d) = Σ  1 / (k + rank_i(d))

where *k* is a constant (default 60) and *rank_i(d)* is the 1-based rank of
document *d* in the *i*-th list.  Documents that appear in multiple lists
accumulate score from each.

Reference: Cormack, Clarke & Buettcher – "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods" (SIGIR 2009).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("satgraffin.rrf")


def _chunk_key(chunk: dict[str, Any]) -> str:
    """
    Produce a stable identity key for a chunk so we can detect duplicates
    across ranked lists.  We prefer ``(source, chunk_id)`` when available,
    falling back to the first 200 characters of the text.
    """
    meta = chunk.get("metadata", {})
    source = meta.get("source", "")
    chunk_id = meta.get("chunk_id", "")

    if source and chunk_id != "":
        return f"{source}:::{chunk_id}"

    # Fallback: hash-like key from text prefix
    text = chunk.get("text", "")[:200]
    return f"text:::{text}"


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Fuse multiple ranked result lists using RRF.

    Parameters
    ----------
    ranked_lists : list[list[dict]]
        Each inner list is a ranked result set (position 0 = rank 1).
        Each dict must have ``"text"`` and ``"metadata"`` keys.
    k : int
        RRF constant (default 60).
    top_n : int
        Number of fused results to return (default 20).

    Returns
    -------
    list[dict]
        Fused results sorted by descending RRF score.  Each dict has an
        additional ``"rrf_score"`` key.
    """
    fused_scores: dict[str, float] = {}
    chunk_by_key: dict[str, dict[str, Any]] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        for rank_0, chunk in enumerate(ranked_list):
            key = _chunk_key(chunk)
            rank_1 = rank_0 + 1  # 1-based rank
            score = 1.0 / (k + rank_1)

            fused_scores[key] = fused_scores.get(key, 0.0) + score

            # Keep the first occurrence of each chunk (preserves metadata)
            if key not in chunk_by_key:
                chunk_by_key[key] = chunk

    # Sort by fused score descending
    sorted_keys = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_n]

    results: list[dict[str, Any]] = []
    for key in sorted_keys:
        doc = {**chunk_by_key[key], "rrf_score": fused_scores[key]}
        results.append(doc)

    logger.debug(
        "RRF fused %d lists (%s docs) → %d results",
        len(ranked_lists),
        "+".join(str(len(rl)) for rl in ranked_lists),
        len(results),
    )
    return results
