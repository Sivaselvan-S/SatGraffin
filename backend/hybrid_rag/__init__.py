"""
SatGraffin Industrial Agentic RAG Pipeline
===========================================
Modular components for agentic retrieval-augmented generation:
  1. SentenceAwareChunker  – sentence-boundary chunking (256–512 tokens)
  2. BM25Retriever         – lexical / keyword retrieval via rank_bm25
  3. reciprocal_rank_fusion – RRF merge of ranked lists (k=60)
  4. CrossEncoderReranker  – local cross-encoder re-ranking
  5. ContextPacker         – Jaccard dedup + greedy token packing
  6. PromptBuilder         – citation-aware prompt formatting
  7. HyDEGenerator         – hypothetical document embeddings
  8. CRAGValidator         – corrective RAG relevance grader
  9. MultimodalProcessor   – Gemini Vision image/diagram analysis
 10. KnowledgeGraphStore   – NetworkX entity relationship graph
 11. QueryTracer           – pipeline performance & audit tracing
"""

from hybrid_rag.chunker import SentenceAwareChunker
from hybrid_rag.bm25_retriever import BM25Retriever
from hybrid_rag.rrf_fusion import reciprocal_rank_fusion
from hybrid_rag.cross_encoder_reranker import CrossEncoderReranker
from hybrid_rag.context_packer import ContextPacker
from hybrid_rag.prompt_builder import PromptBuilder
from hybrid_rag.hyde import HyDEGenerator
from hybrid_rag.crag_validator import CRAGValidator
from hybrid_rag.multimodal_processor import MultimodalProcessor
from hybrid_rag.knowledge_graph import KnowledgeGraphStore
from hybrid_rag.tracer import QueryTracer

__all__ = [
    "SentenceAwareChunker",
    "BM25Retriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "ContextPacker",
    "PromptBuilder",
    "HyDEGenerator",
    "CRAGValidator",
    "MultimodalProcessor",
    "KnowledgeGraphStore",
    "QueryTracer",
]
