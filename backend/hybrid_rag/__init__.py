"""
SatGraffin Hybrid RAG Pipeline
================================
Modular components for hybrid retrieval-augmented generation:
  1. SentenceAwareChunker  – sentence-boundary chunking (256–512 tokens)
  2. BM25Retriever         – lexical / keyword retrieval via rank_bm25
  3. reciprocal_rank_fusion – RRF merge of ranked lists (k=60)
  4. CrossEncoderReranker  – local cross-encoder re-ranking
  5. ContextPacker         – Jaccard dedup + greedy token packing
  6. PromptBuilder         – citation-aware prompt formatting
"""

from hybrid_rag.chunker import SentenceAwareChunker
from hybrid_rag.bm25_retriever import BM25Retriever
from hybrid_rag.rrf_fusion import reciprocal_rank_fusion
from hybrid_rag.cross_encoder_reranker import CrossEncoderReranker
from hybrid_rag.context_packer import ContextPacker
from hybrid_rag.prompt_builder import PromptBuilder

__all__ = [
    "SentenceAwareChunker",
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "ContextPacker",
    "PromptBuilder",
]
