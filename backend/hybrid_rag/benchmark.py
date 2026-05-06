"""
Benchmark: Old Pipeline vs Hybrid RAG Pipeline
===============================================
Runs 5 sample queries through both the legacy FAISS-only retriever and the
new hybrid pipeline (BM25 + FAISS → RRF → cross-encoder → context packer).

Usage:
    cd backend
    python -m hybrid_rag.benchmark
"""

from __future__ import annotations

import os
import sys
import time

from pathlib import Path
from dotenv import load_dotenv

# Ensure backend dir is on path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VECTOR_STORE_PATH = str(BACKEND_DIR / "vector_store")
BM25_INDEX_PATH = str(BACKEND_DIR / "vector_store" / "bm25_index.pkl")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SAMPLE_QUERIES = [
    "What is quantum computing and how do qubits work?",
    "Explain the Pallava dynasty and their contributions",
    "How does surface tension affect water?",
    "What are the strongest anime characters?",
    "Tell me about exoplanets and how they are detected",
]


def _format_preview(text: str, max_len: int = 80) -> str:
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


# ---------------------------------------------------------------------------
# Legacy pipeline (FAISS-only MMR)
# ---------------------------------------------------------------------------

def run_legacy_pipeline(query: str) -> dict:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    store = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )

    retriever = store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})

    t0 = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = []
    for doc in docs:
        results.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "?"),
        })

    sources = list({r["source"] for r in results})
    return {"results": results, "sources": sources, "time_ms": elapsed_ms}


# ---------------------------------------------------------------------------
# Hybrid pipeline (BM25 + FAISS → RRF → cross-encoder → packer)
# ---------------------------------------------------------------------------

def run_hybrid_pipeline(query: str) -> dict:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    from hybrid_rag.bm25_retriever import BM25Retriever
    from hybrid_rag.rrf_fusion import reciprocal_rank_fusion
    from hybrid_rag.cross_encoder_reranker import CrossEncoderReranker
    from hybrid_rag.context_packer import ContextPacker

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    # FAISS
    store = FAISS.load_local(
        VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
    )

    # BM25
    bm25 = BM25Retriever()
    if not bm25.load(BM25_INDEX_PATH):
        print("  [!] BM25 index not found — run the server once to build it.")
        print("      Skipping BM25 for this benchmark run.\n")

    t0 = time.perf_counter()

    # --- Dense retrieval (FAISS) ---
    faiss_docs = store.similarity_search_with_score(query, k=20)
    faiss_results = []
    for doc, score in faiss_docs:
        faiss_results.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "faiss_score": float(score),
        })

    # --- Sparse retrieval (BM25) ---
    bm25_results = bm25.search(query, top_k=20)

    # --- RRF fusion ---
    fused = reciprocal_rank_fusion([bm25_results, faiss_results], k=60, top_n=20)

    # --- Cross-encoder re-rank ---
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(query, fused, top_k=5)

    # --- Context packing ---
    packer = ContextPacker(max_tokens=3000, jaccard_threshold=0.85)
    packed = packer.pack(reranked)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    sources = list({c.get("metadata", {}).get("source", "?") for c in packed})
    return {
        "results": packed,
        "sources": sources,
        "time_ms": elapsed_ms,
        "bm25_count": len(bm25_results),
        "faiss_count": len(faiss_results),
        "fused_count": len(fused),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("  SatGraffin Benchmark: Legacy FAISS-only  vs  Hybrid RAG Pipeline")
    print("=" * 80)

    # Pre-warm the cross-encoder (one-time load)
    print("\n[*] Pre-warming cross-encoder model...")
    from hybrid_rag.cross_encoder_reranker import CrossEncoderReranker
    _warmup = CrossEncoderReranker()
    _warmup.rerank("warmup", [{"text": "warmup text", "metadata": {}}], top_k=1)
    print("[*] Cross-encoder ready.\n")

    for i, query in enumerate(SAMPLE_QUERIES, 1):
        print(f"\n{'─' * 80}")
        print(f"  Query {i}: {query}")
        print(f"{'─' * 80}")

        # Legacy
        print("\n  ▸ Legacy (FAISS MMR):")
        legacy = run_legacy_pipeline(query)
        print(f"    Time:    {legacy['time_ms']:.1f} ms")
        print(f"    Sources: {len(legacy['sources'])}")
        for j, r in enumerate(legacy["results"][:3], 1):
            print(f"    [{j}] {_format_preview(r['text'])}")

        # Hybrid
        print("\n  ▸ Hybrid RAG:")
        hybrid = run_hybrid_pipeline(query)
        print(f"    Time:    {hybrid['time_ms']:.1f} ms")
        print(f"    BM25 hits: {hybrid.get('bm25_count', 0)}")
        print(f"    FAISS hits: {hybrid.get('faiss_count', 0)}")
        print(f"    Fused:   {hybrid.get('fused_count', 0)} → packed {len(hybrid['results'])}")
        print(f"    Sources: {len(hybrid['sources'])}")
        for j, r in enumerate(hybrid["results"][:3], 1):
            score_info = ""
            if "rrf_score" in r:
                score_info += f" RRF={r['rrf_score']:.4f}"
            if "ce_score" in r:
                score_info += f" CE={r['ce_score']:.4f}"
            print(f"    [{j}]{score_info} {_format_preview(r['text'])}")

    print(f"\n{'=' * 80}")
    print("  Benchmark complete.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
